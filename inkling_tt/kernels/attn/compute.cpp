#define LLK_ZEROFLAG_OUTLINE 1

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

using namespace ckernel;
using namespace attn;

namespace {
constexpr uint32_t cache_tiles = get_compile_time_arg_val(0);
constexpr bool global_cache = get_compile_time_arg_val(1);
constexpr uint32_t grid_width = get_compile_time_arg_val(2);
constexpr uint32_t split_tiles = (cache_tiles + splits - 1) / splits;
constexpr uint32_t max_chunk_tiles = split_tiles < 4 ? split_tiles : 4;
constexpr bool streaming = split_tiles > max_chunk_tiles;
constexpr uint32_t vectors_per_tile = 32;

#ifdef TRISC_MATH
using namespace sfpi;
constexpr float scale = __builtin_bit_cast(float, scale_bits);

// FP32 DEST vectors hold one column parity of four 16-wide face rows.
// Only the first three rows are live. The shuffles stay within each face row.
constexpr uint32_t lanes[] = {0, 1, 8, 9};

template <int N>
sfpi_inline vFloat rotate(vFloat v) {
#pragma GCC unroll 4
    for (int i = 0; i < N; ++i) {
        v = subvec_shflror1(v);
    }
    return v;
}

sfpi_inline vFloat vmax(vFloat a, vFloat b) {
    v_if(b > a) { a = b; }
    v_endif;
    return a;
}

sfpi_inline vFloat row_max(vFloat v) {
    v = vmax(v, rotate<1>(v));
    v = vmax(v, rotate<2>(v));
    return vmax(v, rotate<4>(v));
}

sfpi_inline vFloat row_sum(vFloat v) {
    v += rotate<1>(v);
    v += rotate<2>(v);
    return v + rotate<4>(v);
}

sfpi_inline vFloat reciprocal(vFloat value) {
    vFloat r = approx_recip(value);
    r += r * (1.0f - value * r);
    return r + r * (1.0f - value * r);
}

void add_bias(uint32_t tile, uint32_t bias, uint32_t valid) {
#pragma GCC unroll 4
    for (uint32_t i : lanes) {
        vFloat b = dst_reg[bias * vectors_per_tile + i];
        const vInt key = (vConstTileId & 15) + int((i >= 8 ? 16 : 0) + (i & 1));
        v_if(key >= int(valid)) { b += __builtin_bit_cast(float, mask_bits); }
        v_endif;
        dst_reg[tile * vectors_per_tile + i] =
            vFloat(dst_reg[tile * vectors_per_tile + i]) + ::float32_to_bf16_rne(b);
    }
}

void local_softmax(uint32_t count, uint32_t running, uint32_t) {
    vFloat max = -1e30f;
#pragma GCC unroll 4
    for (uint32_t t = 0; t < count; ++t) {
#pragma GCC unroll 4
        for (uint32_t i : lanes) {
            max = vmax(max, dst_reg[t * vectors_per_tile + i]);
        }
    }
    max = row_max(max);
    vFloat sum = 0.0f;
    if constexpr (streaming) {
        if (running) {
            const vFloat prev_max = dst_reg[max_chunk_tiles * vectors_per_tile];
            max = vmax(max, prev_max);
            // The previous sum has already been reduced across the row.
            // Add it once, after reducing the current chunk.
        }
    }
#pragma GCC unroll 4
    for (uint32_t t = 0; t < count; ++t) {
#pragma GCC unroll 4
        for (uint32_t i : lanes) {
            vFloat x = (vFloat(dst_reg[t * vectors_per_tile + i]) - max) * scale;
            vFloat e = sfpu::_sfpu_exp_fp32_accurate_<false>(x);
            if constexpr (!streaming) {
                dst_reg[t * vectors_per_tile + i] = e;
            }
            sum += e;
        }
    }
    sum = row_sum(sum);
    // Carry the previous total into the current max's scale.
    if constexpr (streaming) {
        if (running) {
            const vFloat prev_max = dst_reg[max_chunk_tiles * vectors_per_tile];
            const vFloat prev_sum = dst_reg[max_chunk_tiles * vectors_per_tile + 1];
            sum += prev_sum * sfpu::_sfpu_exp_fp32_accurate_<false>((prev_max - max) * scale);
        }
    }
    dst_reg[max_chunk_tiles * vectors_per_tile] = max;
    dst_reg[max_chunk_tiles * vectors_per_tile + 1] = sum;
}

// Tile 0 contains per-split maxima and the sink. Tile 1 contains the
// corresponding sums (one for the sink). Return one FP32 scale per split.
void group_softmax(uint32_t, uint32_t, uint32_t) {
    const vFloat a = dst_reg[0], b = dst_reg[1], sink = dst_reg[8];
    vFloat max = row_max(vmax(vmax(a, b), sink));
    vFloat ca = sfpu::_sfpu_exp_fp32_accurate_<false>((a - max) * scale);
    vFloat cb = sfpu::_sfpu_exp_fp32_accurate_<false>((b - max) * scale);
    vFloat cs = sfpu::_sfpu_exp_fp32_accurate_<false>((sink - max) * scale);
    vFloat denom = row_sum(ca * vFloat(dst_reg[32]) + cb * vFloat(dst_reg[33]) + cs * vFloat(dst_reg[40]));
    vFloat inv = reciprocal(denom);
    dst_reg[0] = streaming ? max : ca * inv;
    dst_reg[1] = streaming ? inv : cb * inv;
}

void normalize(uint32_t split, uint32_t count, uint32_t) {
    vFloat factor;
    if constexpr (streaming) {
        factor = dst_reg[max_chunk_tiles * vectors_per_tile + 1];
    } else {
        factor = dst_reg[max_chunk_tiles * vectors_per_tile + (split & 1)];
        v_if((vConstTileId & 15) != int(split & ~1u)) { factor = 0.0f; }
        v_endif;
        factor = row_sum(factor);
    }
#pragma GCC unroll 4
    for (uint32_t t = 0; t < count; ++t) {
#pragma GCC unroll 4
        for (uint32_t i : lanes) {
            vFloat e = dst_reg[t * vectors_per_tile + i];
            if constexpr (streaming) {
                const vFloat max = dst_reg[max_chunk_tiles * vectors_per_tile];
                e = sfpu::_sfpu_exp_fp32_accurate_<false>((e - max) * scale);
            }
            vFloat p = e * factor;
            dst_reg[t * vectors_per_tile + i] = ::float32_to_bf16_rne(p);
        }
    }
}

void add_partial(uint32_t tile, uint32_t dest, uint32_t) {
#pragma GCC unroll 4
    for (uint32_t i : lanes) {
        dst_reg[dest * vectors_per_tile + i] = vFloat(dst_reg[dest * vectors_per_tile + i]) + vFloat(dst_reg[tile * vectors_per_tile + i]);
    }
}

void gate_output(uint32_t, uint32_t, uint32_t) {
#pragma GCC unroll 4
    for (uint32_t i : lanes) {
        vFloat value = ::float32_to_bf16_rne(vFloat(dst_reg[i]));
        vFloat sigmoid = sfpu::_sfpu_sigmoid_(dst_reg[vectors_per_tile + i]);
        dst_reg[i] = ::float32_to_bf16_rne(value * sigmoid);
    }
}

#define ATTN_MATH(fn, a, b, c) \
    _llk_math_eltwise_binary_sfpu_params_(fn, a, b, c, VectorMode::None)
#endif

void load_tiles(uint32_t cb, uint32_t count, uint32_t first) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    for (uint32_t i = 0; i < count; ++i) {
        copy_tile(cb, i, first + i);
    }
}

void finish(uint32_t cb, uint32_t first, uint32_t count) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < count; ++i) {
        pack_tile(first + i, cb);
    }
    tile_regs_release();
    cb_push_back(cb, count);
}

void score(uint32_t pos, const Work& work, bool running = false, bool last = true) {
    DeviceZoneScopedN("INK_ATTN_SCORE");
    const uint32_t count = work.tile_count;
    cb_wait_front(q_cb, dim_tiles);
    cb_wait_front(k_cb, count * dim_tiles);
    cb_reserve_back(exps_cb, count);
    const uint32_t stats_out = last ? local_stats_cb : running_stats_cb;
    tile_regs_acquire();
    reconfig_data_format(k_cb, q_cb);
    matmul_init(q_cb, k_cb, /*transpose=*/1);
    for (uint32_t t = 0; t < count; ++t) {
        for (uint32_t d = 0; d < dim_tiles; ++d) {
            matmul_tiles(q_cb, k_cb, d, t * dim_tiles + d, t);
        }
    }
    cb_pop_front(q_cb, dim_tiles);
    cb_pop_front(k_cb, count * dim_tiles);

    const bool relative = !global_cache || !relative_tile_is_zero(pos, work.first_tile + count - 1);
    cb_wait_front(table_cb, count);
    if (relative) {
        cb_wait_front(r_cb, 1);
        for (uint32_t t = 0; t < count; ++t) {
            const uint32_t key_tile = work.first_tile + t;
            if (global_cache && relative_tile_is_zero(pos, key_tile)) {
                continue;
            }
            const uint32_t bias = max_chunk_tiles + t;
            matmul_tiles(r_cb, table_cb, 0, t, bias);
            const uint32_t valid = work.valid_keys - key_tile * 32;
            MATH((ATTN_MATH(add_bias, t, bias, valid)));
        }
        cb_pop_front(r_cb, 1);
    }
    cb_pop_front(table_cb, count);
    if (running) {
        cb_wait_front(running_stats_cb, 1);
        load_tiles(running_stats_cb, 1, max_chunk_tiles);
        cb_pop_front(running_stats_cb, 1);
    }
    MATH((ATTN_MATH(local_softmax, count, running, 0)));
    cb_reserve_back(stats_out, 1);

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(stats_out);
    pack_tile(max_chunk_tiles, stats_out);
    for (uint32_t t = 0; t < count; ++t) {
        pack_tile(t, exps_cb);
    }
    tile_regs_release();
    cb_push_back(stats_out, 1);
    cb_push_back(exps_cb, count);
}

void combine_stats() {
    DeviceZoneScopedN("INK_ATTN_NORMALIZER");
    cb_wait_front(group_stats_cb, 2);
    cb_reserve_back(factor_out_cb, 1);
    tile_regs_acquire();
    load_tiles(group_stats_cb, 2, 0);
    MATH((ATTN_MATH(group_softmax, 0, 0, 0)));
    finish(factor_out_cb, 0, 1);
    cb_pop_front(group_stats_cb, 2);
}

void value_product(uint32_t split, uint32_t count, bool running = false, bool last = true) {
    DeviceZoneScopedN("INK_ATTN_VALUE");
    cb_wait_front(exps_cb, count);
    cb_wait_front(factor_in_cb, 1);
    cb_reserve_back(probs_cb, count);
    tile_regs_acquire();
    load_tiles(exps_cb, count, 0);
    load_tiles(factor_in_cb, 1, max_chunk_tiles);
    MATH((ATTN_MATH(normalize, split, count, 0)));
    finish(probs_cb, 0, count);
    cb_pop_front(exps_cb, count);
    if (last) {
        cb_pop_front(factor_in_cb, 1);
    }

    cb_wait_front(probs_cb, count);
    cb_wait_front(v_cb, count * dim_tiles);
    const uint32_t output = last ? partials_out_cb : running_values_cb;
    tile_regs_acquire();
    reconfig_data_format(v_cb, probs_cb);
    matmul_init(probs_cb, v_cb);
    for (uint32_t t = 0; t < count; ++t) {
        for (uint32_t d = 0; d < dim_tiles; ++d) {
            matmul_tiles(probs_cb, v_cb, t, t * dim_tiles + d, d);
        }
    }
    if (running) {
        cb_wait_front(running_values_cb, dim_tiles);
        load_tiles(running_values_cb, dim_tiles, dim_tiles);
        cb_pop_front(running_values_cb, dim_tiles);
        for (uint32_t d = 0; d < dim_tiles; ++d) {
            MATH((ATTN_MATH(add_partial, dim_tiles + d, d, 0)));
        }
    }
    cb_reserve_back(output, dim_tiles);
    finish(output, 0, dim_tiles);
    cb_pop_front(v_cb, count * dim_tiles);
    cb_pop_front(probs_cb, count);
}

void combine_values(uint32_t active) {
    DeviceZoneScopedN("INK_ATTN_OUTPUT");
    cb_wait_front(partials_in_cb, active);
    cb_wait_front(gate_cb, 1);
    cb_reserve_back(gated_cb, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(partials_in_cb);
    copy_init(partials_in_cb);
    copy_tile(partials_in_cb, 0, 0);
    for (uint32_t split = 1; split < active; ++split) {
        copy_tile(partials_in_cb, split, 1);
        MATH((ATTN_MATH(add_partial, 1, 0, 0)));
    }
    load_tiles(gate_cb, 1, 1);
    MATH((sfpu::sigmoid_init<false>()));
    MATH((ATTN_MATH(gate_output, 0, 0, 0)));
    finish(gated_cb, 0, 1);
    cb_pop_front(partials_in_cb, active);
    cb_pop_front(gate_cb, 1);
}
}  // namespace

void kernel_main() {
    const uint32_t split = (get_absolute_logical_y() * grid_width + get_absolute_logical_x()) % splits;
    cb_wait_front(compute_position_cb, 1);
    const uint32_t pos = read_tile_value(compute_position_cb, 0, 0);
    cb_pop_front(compute_position_cb, 1);
    const Work work(pos, cache_tiles, split);
    if (work.tile_count || split < dim_tiles) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(q_cb, k_cb, exps_cb);
    }
    if (work.tile_count) {
        if constexpr (streaming) {
            for (uint32_t offset = 0; offset < work.tile_count; offset += max_chunk_tiles) {
                const auto chunk = work.chunk(offset, max_chunk_tiles);
                score(pos, chunk, offset != 0, offset + chunk.tile_count == work.tile_count);
            }
        } else {
            score(pos, work);
        }
        if (split == 0) {
            combine_stats();
        }
        if constexpr (streaming) {
            for (uint32_t offset = 0; offset < work.tile_count; offset += max_chunk_tiles) {
                const auto chunk = work.chunk(offset, max_chunk_tiles);
                value_product(split, chunk.tile_count, offset != 0, offset + chunk.tile_count == work.tile_count);
            }
        } else {
            value_product(split, work.tile_count);
        }
    }
    if (split < dim_tiles) {
        combine_values(work.active_splits);
    }
}
