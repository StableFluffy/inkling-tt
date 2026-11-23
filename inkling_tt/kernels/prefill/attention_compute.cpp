#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

namespace {
constexpr uint32_t q = 0;
constexpr uint32_t k = 1;
constexpr uint32_t v = 2;
constexpr uint32_t bias = 3;
constexpr uint32_t odd_bias = 14;
constexpr uint32_t control = 7;
constexpr uint32_t stats = 8;
constexpr uint32_t sink = 9;
constexpr uint32_t score = 10;
constexpr uint32_t prob = 11;
constexpr uint32_t accum = 12;
constexpr uint32_t saved = 13;
constexpr uint32_t saved_tiles = get_compile_time_arg_val(4);
constexpr uint32_t out = 16;
constexpr uint32_t pv_keys = 8;
constexpr uint32_t head_tiles = 4;
constexpr uint32_t scale = 0x3db504f3;  // 1 / sqrt(128)
constexpr uint32_t one = 0x3f800000;

#ifdef TRISC_MATH
void round_bf16(uint32_t tile, uint32_t, uint32_t) {
    for (uint32_t i = 0; i < 32; ++i) {
        const auto lane = tile * 32 + i;
        sfpi::dst_reg[lane] = ::float32_to_bf16_rne(sfpi::vFloat(sfpi::dst_reg[lane]));
    }
}
#endif

inline void bf16_rne(uint32_t tile) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, tile, 0, 0, VectorMode::None)));
}

inline void load(uint32_t cb, uint32_t page, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, page, dst);
}

inline void pack(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    cb_reserve_back(cb, n);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(i, cb);
    }
    tile_regs_release();
    cb_push_back(cb, n);
}

void seed() {
    cb_wait_front(sink, 1);
    tile_regs_acquire();
    load(sink, 0, 0);
    load(sink, 0, 1);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale);
    fill_tile_init();
    fill_tile_bitcast(1, one);
    cb_pop_front(sink, 1);
    pack(stats, 2);
}

void qk(bool biased, bool keep, uint32_t page, uint32_t lag) {
    cb_wait_front(k, head_tiles);
    tile_regs_acquire();
    // The preceding FP32 datacopy leaves a different source tile stride.
    reconfig_full_operand<SrcOrder::Reverse>(q, k);
    matmul_init(q, k, 1);
    for (uint32_t d = 0; d < head_tiles; ++d) {
        matmul_tiles(q, k, d, d, 0);
    }
    cb_pop_front(k, head_tiles);
    if (biased) {
        const uint32_t cb = lag % 2 == 0 ? bias : odd_bias;
        cb_wait_front(cb, 1);
        load(cb, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        cb_pop_front(cb, 1);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale);
    tile_regs_commit();
    cb_reserve_back(score, 1);
    tile_regs_wait();
    pack_reconfig_data_format(score);
    pack_tile(0, score);
    if (keep) {
        pack_reconfig_data_format(saved);
        pack_tile<true>(0, saved, page);
    }
    tile_regs_release();
    cb_push_back(score, 1);
}

void update_stats() {
    cb_wait_front(score, 1);
    cb_wait_front(stats, 2);
    tile_regs_acquire();
    load(score, 0, 2);
    load(score, 0, 0);
    load(stats, 0, 3);
    load(stats, 1, 1);
    sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
    sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binary_max_tile_init();
    binary_max_tile(0, 3, 0, VectorMode::C);
    sub_binary_tile_init();
    sub_binary_tile(3, 0, 3, VectorMode::C);
    sfpu_sub_bcast_col_init();
    sfpu_sub_bcast_col(2, 0);
    exp_tile_init<false>();
    exp_tile<false>(2);
    exp_tile<false>(3, VectorMode::C);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2);
    mul_binary_tile_init();
    mul_binary_tile(1, 3, 1, VectorMode::C);
    add_binary_tile_init();
    add_binary_tile(1, 2, 1, VectorMode::C);
    cb_pop_front(score, 1);
    cb_pop_front(stats, 2);
    pack(stats, 2);
}

void invert_denominator() {
    cb_wait_front(stats, 2);
    tile_regs_acquire();
    load(stats, 0, 0);
    load(stats, 1, 1);
    recip_tile_init();
    recip_tile(1, VectorMode::C);
    cb_pop_front(stats, 2);
    pack(stats, 2);
}

void probability(bool kept, uint32_t page) {
    if (!kept) {
        cb_wait_front(score, 1);
    }
    cb_wait_front(stats, 2);
    tile_regs_acquire();
    load(kept ? saved : score, kept ? page : 0, 0);
    load(stats, 0, 1);
    load(stats, 1, 2);
    sfpu_sub_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);
    exp_tile_init<false>();
    exp_tile<false>(0);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 2);
    bf16_rne(0);
    if (!kept) {
        cb_pop_front(score, 1);
    }
    pack(prob, 1);
}

// Accumulate P @ V for a group of keys in Dest. The FP32 accumulator round
// trip between groups is exact, so the sum order matches one key at a time.
#if PREFILL_DISTANCE
constexpr uint32_t relative = 18;
constexpr uint32_t table = 19;
constexpr uint32_t distances = 5;

// Distance-domain logits of one query tile: [32 x 32] @ [32 x 1024] in FP32,
// one K tile per output tile as in the separate projection.
void project_distances() {
    cb_wait_front(relative, 1);
    cb_reserve_back(distances, 32);
    for (uint32_t j = 0; j < 32; j += head_tiles) {
        tile_regs_acquire();
        reconfig_full_operand<SrcOrder::Reverse>(relative, table);
        matmul_init(relative, table);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            matmul_tiles(relative, table, 0, j + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(distances);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            pack_tile<true>(d, distances, j + d);
        }
        tile_regs_release();
    }
    cb_push_back(distances, 32);
    cb_pop_front(relative, 1);
}
#endif

void pv(bool first, bool last, uint32_t keys) {
    tile_regs_acquire();
    if (!first) {
        cb_wait_front(accum, head_tiles);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            load(accum, d, d);
        }
        cb_pop_front(accum, head_tiles);
    }
    reconfig_full_operand<SrcOrder::Reverse>(prob, v);
    matmul_init(prob, v);
    // Consume each probability tile from the CB front: a group can wrap the ring.
    for (uint32_t j = 0; j < keys; ++j) {
        cb_wait_front(prob, 1);
        cb_wait_front(v, head_tiles);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            matmul_tiles(prob, v, 0, d, d);
        }
        cb_pop_front(v, head_tiles);
        cb_pop_front(prob, 1);
    }
    if (last) {
        for (uint32_t d = 0; d < head_tiles; ++d) {
            bf16_rne(d);
        }
    }
    pack(last ? out : accum, head_tiles);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(q, k, out);
    constexpr uint32_t query_tiles = get_compile_time_arg_val(0);
    constexpr bool global = get_compile_time_arg_val(2);
    constexpr uint32_t window_tiles = 32;
#if PREFILL_DISTANCE
    cb_wait_front(table, 32);
#endif
    cb_wait_front(control, 1);
    const uint32_t start = read_tile_value(control, 0, 0) / 32;
    const uint32_t valid_tiles = (read_tile_value(control, 0, 1) + 31) / 32;
    cb_pop_front(control, 1);
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        const uint32_t i = task % query_tiles;
        const uint32_t qt = i % 2 == 0 ? i / 2 : query_tiles - 1 - i / 2;
        if (qt >= valid_tiles) {
            continue;
        }
        const uint32_t q_abs = start + qt;
        const uint32_t first = global || q_abs <= window_tiles ? 0 : q_abs - window_tiles;
        const uint32_t count = q_abs + 1 - first;
        const uint32_t saved_start = count > saved_tiles ? count - saved_tiles : 0;
#if PREFILL_DISTANCE
        project_distances();
#endif
        cb_wait_front(q, head_tiles);
        seed();
        if constexpr (saved_tiles != 0) {
            cb_reserve_back(saved, saved_tiles);
        }
        for (uint32_t i = 0; i < count; ++i) {
            qk(count - i <= 33, i >= saved_start, i - saved_start, count - 1 - i);
            update_stats();
        }
        invert_denominator();
        if constexpr (saved_tiles != 0) {
            cb_push_back(saved, saved_tiles);
            cb_wait_front(saved, saved_tiles);
        }
        for (uint32_t group = 0; group < count; group += pv_keys) {
            const uint32_t keys = count - group < pv_keys ? count - group : pv_keys;
            for (uint32_t i = group; i < group + keys; ++i) {
                const bool kept = i >= saved_start;
                if (!kept) {
                    qk(count - i <= 33, false, 0, count - 1 - i);
                }
                probability(kept, i - saved_start);
            }
            pv(group == 0, group + keys == count, keys);
        }
        cb_pop_front(q, head_tiles);
        cb_pop_front(stats, 2);
        if constexpr (saved_tiles != 0) {
            cb_pop_front(saved, saved_tiles);
        }
    }
}
