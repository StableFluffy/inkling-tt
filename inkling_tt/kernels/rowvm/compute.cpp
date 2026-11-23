#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef ROWVM_ZONES
#define RV_ZONE(name) DeviceZoneScopedN(name)
#else
#define RV_ZONE(name)
#endif
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_conversions.h"
#endif

using namespace ckernel;

#ifdef ROWVM_TASK_BASE
inline uint32_t rowvm_task() {
    return get_absolute_logical_y() * ROWVM_GRID_X + get_absolute_logical_x() - ROWVM_TASK_BASE;
}
#else
inline uint32_t rowvm_task() { return get_arg_val<uint32_t>(0); }
#endif
namespace {
enum Program : uint32_t { RMS, L2, ADD, GATE, CONV, MOE_HIDDEN, MOE_COMBINE, GATE_INPUT, MESSAGE, MIX, ROUTER, RMS_STRIPED, GATE_INPUT_STRIPED, MESSAGE_STRIPED, SINK_TILE, CONV_ADD, CONV_MIX, MESSAGE_UP, CONV_K, CONV_V };
constexpr uint32_t op = get_compile_time_arg_val(0);
constexpr uint32_t width = get_compile_time_arg_val(1);
constexpr uint32_t flags = get_compile_time_arg_val(2);
constexpr uint32_t vectors = 2 * ((width + 63) / 64);
constexpr uint32_t tiles = (width + 1023) / 1024;
constexpr uint32_t out_cb = 16;
constexpr uint32_t state_cb = 17;
constexpr uint32_t mix_rows = get_compile_time_arg_val(3);
constexpr uint32_t mix_width = get_compile_time_arg_val(4);
constexpr uint32_t stripes = get_compile_time_arg_val(5);
#ifdef ROWVM_INDEXED_CONV
constexpr uint32_t state_rows = 1;
#else
constexpr uint32_t state_rows = 4;
#endif
constexpr bool conv_mix = op == CONV_MIX;
constexpr bool conv_kv = op == CONV_K || op == CONV_V;
constexpr bool clip = op == L2 || op == CONV_K;
constexpr bool fused_conv = op == CONV_ADD || conv_mix;
constexpr bool striped_message = op == MESSAGE_STRIPED || op == MESSAGE_UP;
constexpr uint32_t aggregate_pages = (op == MIX || conv_mix) ? (stripes + 15) / 16 : 1;
constexpr uint32_t aggregate_cb = 8;
constexpr uint32_t broadcast_cb = 9;
constexpr uint32_t partial_cb = 18;
constexpr uint32_t coeff_cb = 19;
constexpr uint32_t mix_cb = 20;
constexpr uint32_t conv_partial_cb = 10;
constexpr uint32_t bank_scratch_cb = 11;
constexpr uint32_t conv_output_cb = 21;
constexpr uint32_t bank_output_cb = 22;
constexpr bool weighted = flags & 1;
constexpr bool bf_out = flags & 2;
constexpr bool extra = flags & 4;
constexpr bool conv_residual = flags & 16;
constexpr bool append_bank = flags & 32;

#ifdef TRISC_MATH
namespace math_impl {
using namespace sfpi;
constexpr uint32_t vectors_per_tile = 32;
constexpr uint32_t scratch = 6 * vectors_per_tile;
constexpr uint32_t factor_slot = 7 * vectors_per_tile;

template <int N>
sfpi_inline vFloat rotate(vFloat x) {
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        x = subvec_shflror1(x);
    }
    return x;
}

sfpi_inline vFloat sum32(vFloat x) {
    x += rotate<1>(x);
    x += rotate<2>(x);
    x += rotate<4>(x);
    vFloat b = x, c = x, d = x;
    subvec_transp(x, b, c, d);
    return (x + b) + (c + d);
}

// In fp32 DEST a vector addresses one parity of four physical 16-wide rows.
sfpi_inline vInt scalar_lane(int element) {
    return vInt((element / 16) * 16 + (element % 16 & ~1));
}

sfpi_inline vFloat splat(uint32_t base, int element) {
    vFloat x = dst_reg[base + (element & 1)];
    v_if(vConstTileId != scalar_lane(element)) { x = 0.0f; }
    v_endif;
    return sum32(x);
}

sfpi_inline vFloat reciprocal(vFloat x) {
    vFloat r = approx_recip(x);
    r += r * (1.0f - x * r);
    r += r * (1.0f - x * r);
    return r;
}

void init_tanh() { sfpu::tanh_init<false, true>(); }
void init_sqrt() { sfpu::sqrt_init<false>(); }

void init() {
    if constexpr (op == RMS || op == L2 || op == GATE_INPUT || op == MIX || conv_mix || conv_kv || op == RMS_STRIPED || op == GATE_INPUT_STRIPED) {
        sfpu::sqrt_init<false>();
    }
    if constexpr (op == GATE || op == MOE_HIDDEN || op == ROUTER) {
        sfpu::sigmoid_init<false>();
    }
}

template <bool Round = bf_out>
sfpi_inline vFloat rounded(vFloat x) {
    if constexpr (Round) {
        // Blackhole's conversion instruction resolves ties away from zero.
        return ::float32_to_bf16_rne(x);
    }
    return x;
}

void factor(uint32_t, uint32_t, uint32_t) {
    vFloat sum = 0.0f;
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat x = dst_reg[v];
        sum = x * x + sum;
    }
    sum = sum32(sum);
    if constexpr (clip) {
        constexpr float threshold = __builtin_bit_cast(float, get_compile_time_arg_val(3));
        vFloat norm = sfpu::_calculate_sqrt_body_<false, false, false>(sum);
        vFloat f = threshold * reciprocal(norm + 1e-6f);
        v_if(f > 1.0f) { f = 1.0f; }
        v_endif;
        dst_reg[factor_slot] = f;
    } else {
        dst_reg[factor_slot] = sfpu::_calculate_sqrt_body_<false, true, false>(
            sum * (1.0f / width) + 1e-6f);
    }
}

void scale(uint32_t tile, uint32_t use_weight, uint32_t unnormalized) {
    vFloat inv = unnormalized ? vFloat(1.0f) : vFloat(dst_reg[factor_slot]);
    const uint32_t base = tile * vectors_per_tile;
    for (uint32_t v = base; v < base + vectors_per_tile && v < vectors; ++v) {
        vFloat x = dst_reg[v] * inv;
        if (use_weight) {
            x *= dst_reg[scratch + v - base];
        }
        dst_reg[v] = rounded(x);
    }
}

void pre_scale(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < vectors; ++v) {
        dst_reg[v] = vFloat(dst_reg[v]) * vFloat(dst_reg[scratch + v]);
    }
}

void apply_tau(uint32_t, uint32_t, uint32_t) {
    vFloat tau = splat(scratch, 0);
    vFloat inv = dst_reg[factor_slot];
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat x = rounded<(flags & 8) != 0>(vFloat(dst_reg[v]) * inv);
        dst_reg[v] = rounded(x * tau);
    }
}

void sigmoid_operand(uint32_t tile, uint32_t, uint32_t) {
    const uint32_t base = tile * vectors_per_tile;
    for (uint32_t v = 0; v < vectors; ++v) {
        dst_reg[2 * vectors_per_tile + v] = sfpu::_sfpu_sigmoid_(dst_reg[base + v]);
    }
}

void softcap_operand(uint32_t tile, uint32_t, uint32_t) {
    const uint32_t base = tile * vectors_per_tile;
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat x = dst_reg[base + v];
        x *= tile == 0 ? vFloat(0.25f) : vFloat(0.04f);
        x = sfpu::_sfpu_tanh_fp32_accurate_(x);
        dst_reg[base + v] = x * (tile == 0 ? vFloat(4.0f) : vFloat(25.0f));
    }
}

// Write a stripe's result into destination tile 0.
void elementwise(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat a = dst_reg[v];
        if constexpr (op == ADD) {
            a += dst_reg[vectors_per_tile + v];
        } else if constexpr (op == GATE) {
            a *= dst_reg[2 * vectors_per_tile + v];
        } else if constexpr (op == MOE_HIDDEN) {
            a *= dst_reg[2 * vectors_per_tile + v];
            a *= dst_reg[vectors_per_tile + v];
        }
        dst_reg[v] = rounded(a);
    }
}

void tanh_tile(uint32_t src, uint32_t count, uint32_t) {
    const uint32_t base = src * vectors_per_tile;
    for (uint32_t v = 0; v < count; ++v) {
        dst_reg[base + v] = sfpu::_sfpu_tanh_fp32_accurate_(dst_reg[base + v]);
    }
}

void message(uint32_t tile, uint32_t, uint32_t) {
    const uint32_t base = tile * vectors_per_tile;
    for (uint32_t v = 0; v < vectors_per_tile && base + v < vectors; ++v) {
        vFloat u = dst_reg[base + v];
        vFloat m = dst_reg[5 * vectors_per_tile + v];
        vFloat f = dst_reg[scratch + v];
        vFloat g = dst_reg[factor_slot + v];
        vFloat value = (1.0f - f) * u + g * m;
        dst_reg[base + v] = weighted ? value : rounded(value);
    }
}

void conv_lag(uint32_t lag, uint32_t head, uint32_t) {
    vFloat dynamic = 0.0f;
    if constexpr (extra && (op == CONV || conv_kv)) {
        dynamic = splat(scratch, 4 * head + lag) * (1.0f / 3.0f);
    }
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat sample = dst_reg[lag * vectors_per_tile + v];
        vFloat tap = dst_reg[5 * vectors_per_tile + v];
        if constexpr (extra && (op == CONV || conv_kv)) {
            tap += dynamic;
        }
        vFloat product = sample * tap;
        vFloat value = lag == 0 ? product : vFloat(dst_reg[4 * vectors_per_tile + v]) + product;
        dst_reg[4 * vectors_per_tile + v] = lag == 3 ? rounded<bf_out || fused_conv>(value) : value;
    }
}

void conv_add(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat residual = dst_reg[6 * vectors_per_tile + v];
        residual += dst_reg[4 * vectors_per_tile + v];
        dst_reg[4 * vectors_per_tile + v] = rounded<!conv_mix || (flags & 64)>(residual);
    }
}

void combine_row(uint32_t row, uint32_t, uint32_t) {
    vFloat gamma = splat(scratch, row);
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat d = dst_reg[vectors_per_tile + v];
        vFloat term = d * gamma;
        vFloat value = row == 0 ? term : vFloat(dst_reg[v]) + term;
        dst_reg[v] = row == 7 ? rounded(value) : value;
    }
}
sfpi_inline void set_scalar(uint32_t base, int element, vFloat value) {
    v_if(vConstTileId == scalar_lane(element)) {
        dst_reg[base + (element & 1)] = value;
    }
    v_endif;
}

void mix_metric(uint32_t row, uint32_t square, uint32_t dest) {
    vFloat sum = 0.0f;
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat x = dst_reg[v];
        vFloat q = square ? x : vFloat(dst_reg[vectors_per_tile + v]);
        sum = x * q + sum;
    }
    vFloat total = sum32(sum);
    set_scalar(dest * vectors_per_tile, row, total);
}

sfpi_inline vFloat vmax(vFloat x, vFloat y) {
    v_if(y > x) { x = y; }
    v_endif;
    return x;
}

sfpi_inline vFloat max32(vFloat x) {
    x = vmax(x, rotate<1>(x));
    x = vmax(x, rotate<2>(x));
    x = vmax(x, rotate<4>(x));
    vFloat b = x, c = x, d = x;
    subvec_transp(x, b, c, d);
    return vmax(vmax(x, b), vmax(c, d));
}

void stripe_rms(uint32_t, uint32_t, uint32_t) {
    vFloat sum = 0.0f;
    for (uint32_t i = 0; i < (stripes + 15) / 16; ++i) {
        vFloat partials = dst_reg[2 * i];
        v_if((vConstTileId & 3) != 0 || vConstTileId >= int(4 * stripes - 64 * i)) { partials = 0.0f; }
        v_endif;
        sum += partials;
    }
    dst_reg[factor_slot] = sfpu::_calculate_sqrt_body_<false, true, false>(
        sum32(sum) * (1.0f / (mix_width ? mix_width : 1)) + 1e-6f);
}

[[gnu::noinline]] void mix_sum(uint32_t, uint32_t, uint32_t) {
    vFloat even = 0.0f;
    vFloat odd = 0.0f;
    for (uint32_t i = 0; i < stripes; ++i) {
        even += dst_reg[2 * i];
        odd += dst_reg[2 * i + 1];
    }
    dst_reg[0] = even;
    dst_reg[1] = odd;
}

[[gnu::noinline]] void mix_rms(uint32_t, uint32_t, uint32_t) {
    vFloat sum = splat(0, 0);
    dst_reg[2] = sfpu::_calculate_sqrt_body_<false, true, false>(
        sum * (1.0f / (mix_width ? mix_width : 1)) + 1e-6f);
}

[[gnu::noinline]] void mix_mask(uint32_t, uint32_t, uint32_t) {
    vFloat even = dst_reg[0];
    vFloat odd = dst_reg[1];
    vFloat inv = dst_reg[2];
    v_if(vConstTileId == 0) { odd *= inv; }
    v_endif;
    v_if(vConstTileId == 0 || vConstTileId >= int(mix_rows + 2)) { even = -__builtin_inff(); }
    v_endif;
    v_if(vConstTileId >= int(mix_rows + 1)) { odd = -__builtin_inff(); }
    v_endif;
    dst_reg[0] = even;
    dst_reg[1] = odd;
}

[[gnu::noinline]] void mix_max(uint32_t, uint32_t, uint32_t) {
    vFloat even = dst_reg[0];
    vFloat odd = dst_reg[1];
    dst_reg[3] = max32(vmax(even, odd));
}

[[gnu::noinline]] void mix_exp(uint32_t parity, uint32_t, uint32_t) {
    vFloat x = dst_reg[parity];
    x -= dst_reg[3];
    dst_reg[parity] = sfpu::_sfpu_exp_fp32_accurate_<false>(x);
}

[[gnu::noinline]] void mix_denominator(uint32_t, uint32_t, uint32_t) {
    vFloat sum = vFloat(dst_reg[0]) + vFloat(dst_reg[1]);
    dst_reg[3] = reciprocal(sum32(sum));
}

[[gnu::noinline]] void mix_normalize(uint32_t, uint32_t, uint32_t) {
    vFloat r = dst_reg[3];
    vFloat even = vFloat(dst_reg[0]) * r;
    vFloat odd = vFloat(dst_reg[1]) * r;
    const vFloat inv = dst_reg[2];
    // The first mixture term is the raw partial times its RMS factor.
    v_if(vConstTileId == 0) { odd *= inv; }
    v_endif;
    dst_reg[factor_slot] = even;
    dst_reg[factor_slot + 1] = odd;
}

void mix_row(uint32_t metric, uint32_t last, uint32_t) {
    const vFloat coeff = splat(factor_slot, metric);
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat value = vFloat(dst_reg[vectors_per_tile + v]) * coeff;
        if (metric != 1) {
            value += dst_reg[v];
        }
        dst_reg[v] = !weighted && last ? rounded(value) : value;
    }
}

void scale_mix(uint32_t skip, uint32_t, uint32_t bank_row) {
    const vFloat inv = skip ? vFloat(1.0f) : splat(factor_slot, 0);
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat x = vFloat(dst_reg[v]) * inv;
        if constexpr (weighted) {
            if (!bank_row) {
                x *= dst_reg[scratch + v];
            }
        }
        if (bank_row) {
            // The stock Blackhole selector matmul cleared the low 14 bits
            // before storing the FP32 bank row. Preserve that writeback.
            const vUInt bits = as<vUInt>(x) & 0xffffc000U;
            dst_reg[v] = as<vFloat>(bits);
        } else {
            dst_reg[v] = rounded(x);
        }
    }
}
void router_scores(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < vectors; ++v) {
        vFloat score = vFloat(dst_reg[2 * vectors_per_tile + v]) + vFloat(dst_reg[vectors_per_tile + v]);
        dst_reg[vectors_per_tile + v] = score;
    }
}

[[gnu::noinline]] void router_factor(uint32_t, uint32_t, uint32_t) {
    vFloat sum = vFloat(dst_reg[0]) + vFloat(dst_reg[1]);
    dst_reg[2] = 8.0f * reciprocal(sum32(sum) + 1e-20f);
}

void router_gamma(uint32_t, uint32_t, uint32_t) {
    vFloat coeff = dst_reg[2];
    dst_reg[0] = vFloat(dst_reg[0]) * coeff;
    dst_reg[1] = vFloat(dst_reg[1]) * coeff;
}
}  // namespace math_impl

#define ROWVM_MATH(FN, A, B, C) \
    _llk_math_eltwise_binary_sfpu_params_(math_impl::FN, A, B, C, VectorMode::None)
#endif

void start_load(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

void load(uint32_t cb, uint32_t src, uint32_t dest) {
    start_load(cb);
    copy_tile(cb, src, dest);
}

void finish(uint32_t first, uint32_t count, uint32_t cb = out_cb) {
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = first; t < first + count; ++t) {
        pack_tile(t, cb);
    }
    tile_regs_release();
}

void rms_dest(uint32_t weight_cb, bool use_weight, bool skip = false) {
    if (!skip) {
        MATH((ROWVM_MATH(factor, 0, 0, 0)));
    }
    for (uint32_t t = 0; t < tiles; ++t) {
        if (use_weight) {
            load(weight_cb, t, 6);
        }
        MATH((ROWVM_MATH(scale, t, use_weight, skip)));
    }
}

void run_norm() {
    constexpr uint32_t source = conv_kv ? conv_partial_cb : 0;
    cb_wait_front(source, tiles);
    if constexpr (weighted) {
        cb_wait_front(1, tiles);
    }
    if constexpr (op == L2 && extra) {
        cb_wait_front(weighted ? 2 : 1, 1);
    }
    cb_reserve_back(out_cb, tiles);
    tile_regs_acquire();
    start_load(source);
    copy_block(source, 0, 0, tiles);
    if constexpr (clip) {
        if constexpr (weighted) {
            load(1, 0, 6);
            MATH((ROWVM_MATH(pre_scale, 0, 0, 0)));
        }
        MATH((ROWVM_MATH(factor, 0, 0, 0)));
        if constexpr (op == L2 && extra) {
            load(weighted ? 2 : 1, 0, 6);
            MATH((ROWVM_MATH(apply_tau, 0, 0, 0)));
        } else {
            MATH((ROWVM_MATH(scale, 0, 0, 0)));
        }
    } else {
        rms_dest(1, weighted);
    }
    finish(0, tiles);
    cb_push_back(out_cb, tiles);
    cb_pop_front(source, tiles);
    if constexpr (weighted) {
        cb_pop_front(1, tiles);
    }
    if constexpr (op == L2 && extra) {
        cb_pop_front(weighted ? 2 : 1, 1);
    }
}

void run_elementwise() {
    cb_wait_front(0, 1);
    cb_wait_front(1, 1);
    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    copy_init(0);
    copy_tile(0, 0, 0);
    load(1, 0, 1);
    if constexpr (op == GATE || op == MOE_HIDDEN) {
        MATH((ROWVM_MATH(sigmoid_operand, op == GATE ? 1 : 0, 0, 0)));
    }
    if constexpr (op == MOE_HIDDEN) {
        MATH((math_impl::init_tanh()));
        MATH((ROWVM_MATH(softcap_operand, 0, 0, 0)));
        MATH((ROWVM_MATH(softcap_operand, 1, 0, 0)));
    }
    MATH((ROWVM_MATH(elementwise, 0, 0, 0)));
    finish(0, 1);
    cb_push_back(out_cb, 1);
    cb_pop_front(0, 1);
    cb_pop_front(1, 1);
}

void run_conv() {
    RV_ZONE("RV_CONV");
    constexpr uint32_t first = conv_mix ? 4 : 0;
    constexpr uint32_t target = conv_mix ? conv_output_cb : (conv_kv ? conv_partial_cb : out_cb);
    cb_wait_front(first, 1);
    cb_wait_front(first + 1, 3);
    cb_wait_front(first + 2, 4);
    if constexpr (((op == CONV || conv_kv) && extra) || conv_residual) {
        cb_wait_front(first + 3, 1);
    }
    cb_reserve_back(target, 1);
    cb_reserve_back(state_cb, state_rows);
    if constexpr (conv_mix) {
        cb_reserve_back(conv_partial_cb, 1);
    }
    tile_regs_acquire();
    load(first, 0, 0);
    start_load(first + 1);
    copy_block(first + 1, 0, 1, 3);
    if constexpr (((op == CONV || conv_kv) && extra) || conv_residual) {
        load(first + 3, 0, 6);
    }
    start_load(first + 2);
    [[maybe_unused]] const uint32_t task = rowvm_task();
    for (uint32_t lag = 0; lag < 4; ++lag) {
        copy_tile(first + 2, lag, 5);
        MATH((ROWVM_MATH(conv_lag, lag, task * width / 128, 0)));
    }
    if constexpr (conv_residual) {
        MATH((ROWVM_MATH(conv_add, 0, 0, 0)));
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(target);
    pack_tile(4, target);
    if constexpr (conv_mix) {
        pack_reconfig_data_format(conv_partial_cb);
        pack_tile(4, conv_partial_cb);
        cb_push_back(conv_partial_cb, 1);
    }
    pack_reconfig_data_format(state_cb);
    for (uint32_t lag = 0; lag < state_rows; ++lag) {
        pack_tile(lag, state_cb);
    }
    tile_regs_release();
    cb_push_back(target, 1);
    cb_push_back(state_cb, state_rows);
    cb_pop_front(first, 1);
    cb_pop_front(first + 1, 3);
    cb_pop_front(first + 2, 4);
    if constexpr (((op == CONV || conv_kv) && extra) || conv_residual) {
        cb_pop_front(first + 3, 1);
    }
}

void run_combine() {
    cb_wait_front(0, 8);
    cb_wait_front(1, 1);
    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    load(1, 0, 6);
    start_load(0);
    for (uint32_t r = 0; r < 8; ++r) {
        copy_tile(0, r, 1);
        MATH((ROWVM_MATH(combine_row, r, 0, 0)));
    }
    finish(0, 1);
    cb_push_back(out_cb, 1);
    cb_pop_front(0, 8);
    cb_pop_front(1, 1);
}

void run_gate_input() {
    cb_reserve_back(out_cb, 2 * tiles);
    for (uint32_t input = 0; input < 2; ++input) {
        cb_wait_front(input, tiles);
        tile_regs_acquire();
        start_load(input);
        copy_block(input, 0, 0, tiles);
        rms_dest(0, false, extra && input == 0);
        finish(0, tiles);
        cb_pop_front(input, tiles);
    }
    cb_push_back(out_cb, 2 * tiles);
}

void run_message() {
    RV_ZONE("RV_MSG");
    constexpr bool norm = weighted && !striped_message;
    constexpr uint32_t target = striped_message ? mix_cb : out_cb;
    constexpr uint32_t count = norm ? 5 : 4;
    for (uint32_t cb = 0; cb < count; ++cb) {
        cb_wait_front(cb, tiles);
    }
    cb_reserve_back(target, tiles);
    tile_regs_acquire();
    copy_init(0);
    copy_block(0, 0, 0, tiles);
    MATH((math_impl::init_tanh()));
    for (uint32_t t = 0; t < tiles; ++t) {
        [[maybe_unused]] const uint32_t count = t + 1 == tiles ? vectors - 32 * t : 32;
        load(1, t, 5);
        load(2, t, 6);
        load(3, t, 7);
        MATH((ROWVM_MATH(tanh_tile, 6, count, 0)));
        MATH((ROWVM_MATH(tanh_tile, 7, count, 0)));
        MATH((ROWVM_MATH(message, t, 0, 0)));
    }
    if constexpr (norm) {
        MATH((math_impl::init_sqrt()));
        rms_dest(4, true);
    }
    pack_reconfig_data_format(target);
    finish(0, tiles, target);
    cb_push_back(target, tiles);
    for (uint32_t cb = 0; cb < count; ++cb) {
        cb_pop_front(cb, tiles);
    }
}

void run_message_matmuls() {
    RV_ZONE("RV_MSGMM");
    constexpr uint32_t controller = 12;
    constexpr uint32_t logits = 15;
    constexpr uint32_t k_tiles = 2;
    constexpr uint32_t n_tiles = 2;
    cb_wait_front(controller, k_tiles);
    cb_reserve_back(logits, 2 * n_tiles);
    pack_reconfig_data_format(logits);
    for (uint32_t weights : {13u, 14u}) {
        cb_wait_front(weights, k_tiles * n_tiles);
        reconfig_data_format<SrcOrder::Reverse>(controller, weights);
        matmul_block_init(controller, weights, 0, n_tiles, 1, k_tiles);
        tile_regs_acquire();
        matmul_block_k(controller, weights, 0, 0, 0, 0, n_tiles, 1, k_tiles, k_tiles, n_tiles);
        finish(0, n_tiles, logits);
        cb_pop_front(weights, k_tiles * n_tiles);
    }
    cb_push_back(logits, 2 * n_tiles);
    cb_pop_front(controller, k_tiles);
}
// Helpers for the two stripe collectives (logits and optional post-mix RMS).
void pack_scratch(uint32_t dest, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(dest, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}

void leader_coeff(bool rms_pass) {
    RV_ZONE("RV_LEAD");
    constexpr uint32_t pages = aggregate_pages ? aggregate_pages : 1;
    cb_wait_front(aggregate_cb, pages);
    cb_reserve_back(coeff_cb, 1);
    tile_regs_acquire();
    start_load(aggregate_cb);
    copy_block(aggregate_cb, 0, 0, pages);
    if (rms_pass) {
        MATH((ROWVM_MATH(stripe_rms, 0, 0, 0)));
    } else {
        MATH((ROWVM_MATH(mix_sum, 0, 0, 0)));
        MATH((ROWVM_MATH(mix_rms, 0, 0, 0)));
        MATH((ROWVM_MATH(mix_mask, 0, 0, 0)));
        MATH((ROWVM_MATH(mix_max, 0, 0, 0)));
        MATH((ROWVM_MATH(mix_exp, 0, 0, 0)));
        MATH((ROWVM_MATH(mix_exp, 1, 0, 0)));
        MATH((ROWVM_MATH(mix_denominator, 0, 0, 0)));
        MATH((ROWVM_MATH(mix_normalize, 0, 0, 0)));
    }
    cb_pop_front(aggregate_cb, pages);
    pack_scratch(7, coeff_cb);
}

void run_bank_append() {
    constexpr uint32_t entry_cb = 7;
    cb_wait_front(entry_cb, 1);
    cb_reserve_back(partial_cb, 1);
    tile_regs_acquire();
    load(entry_cb, 0, 0);
    MATH((ROWVM_MATH(mix_metric, 0, 1, 2)));
    pack_scratch(2, partial_cb);
    if (rowvm_task() == 0) {
        leader_coeff(true);
    }
    cb_wait_front(broadcast_cb, 1);
    cb_reserve_back(bank_scratch_cb, 1);
    cb_reserve_back(bank_output_cb, 1);
    tile_regs_acquire();
    load(entry_cb, 0, 0);
    load(broadcast_cb, 0, 7);
    // This row remains FP32 even when the final mix output is BF16.
    MATH((ROWVM_MATH(scale_mix, 0, 0, 1)));
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(bank_scratch_cb);
    pack_tile(0, bank_scratch_cb);
    pack_tile(0, bank_output_cb);
    tile_regs_release();
    cb_push_back(bank_scratch_cb, 1);
    cb_push_back(bank_output_cb, 1);
    cb_pop_front(entry_cb, 1);
    cb_pop_front(broadcast_cb, 1);
}

void run_mix() {
    constexpr uint32_t source = conv_mix ? conv_partial_cb : 0;
    constexpr uint32_t bank_rows = mix_rows - append_bank;
    constexpr uint32_t rows = bank_rows ? bank_rows : 1;
    const bool leader = rowvm_task() % (stripes ? stripes : 1) == 0;
    {
    RV_ZONE("RV_MIXIN");
    cb_wait_front(source, 1);
    cb_wait_front(1, 1);
    cb_wait_front(2, rows);
    }
    if constexpr (append_bank) {
        cb_wait_front(bank_scratch_cb, 1);
    }
    cb_reserve_back(partial_cb, 1);
    tile_regs_acquire();
    load(source, 0, 0);
    load(1, 0, 1);
    MATH((ROWVM_MATH(mix_metric, 0, 1, 2)));
    MATH((ROWVM_MATH(mix_metric, 1, 0, 2)));
    start_load(2);
    for (uint32_t j = 0; j < bank_rows; ++j) {
        copy_tile(2, j, 0);
        MATH((ROWVM_MATH(mix_metric, j + 2, 0, 2)));
    }
    if constexpr (append_bank) {
        load(bank_scratch_cb, 0, 0);
        MATH((ROWVM_MATH(mix_metric, bank_rows + 2, 0, 2)));
    }
    cb_pop_front(1, 1);
    pack_scratch(2, partial_cb);
    if (leader) {
        leader_coeff(false);
    }
    {
    RV_ZONE("RV_BWAIT1");
    cb_wait_front(broadcast_cb, 1);
    }
    if constexpr (weighted) {
        cb_reserve_back(mix_cb, 1);
        cb_reserve_back(partial_cb, 1);
    } else {
        cb_reserve_back(out_cb, 1);
    }
    tile_regs_acquire();
    load(broadcast_cb, 0, 7);
    load(source, 0, 1);
    MATH((ROWVM_MATH(mix_row, 1, mix_rows == 0, 0)));
    start_load(2);
    for (uint32_t j = 0; j < bank_rows; ++j) {
        copy_tile(2, j, 1);
        MATH((ROWVM_MATH(mix_row, j + 2, j + 1 == mix_rows, 0)));
    }
    if constexpr (append_bank) {
        load(bank_scratch_cb, 0, 1);
        MATH((ROWVM_MATH(mix_row, bank_rows + 2, 1, 0)));
        cb_pop_front(bank_scratch_cb, 1);
    }
    cb_pop_front(source, 1);
    cb_pop_front(2, rows);
    cb_pop_front(broadcast_cb, 1);
    if constexpr (weighted) {
        MATH((ROWVM_MATH(mix_metric, 0, 1, 2)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(mix_cb);
        pack_tile(0, mix_cb);
        pack_tile(2, partial_cb);
        tile_regs_release();
        cb_push_back(mix_cb, 1);
        cb_push_back(partial_cb, 1);
        if (leader) {
            leader_coeff(true);
        }
        {
        RV_ZONE("RV_BWAIT2");
        cb_wait_front(broadcast_cb, 1);
        }
        cb_wait_front(mix_cb, 1);
        cb_wait_front(3, 1);
        cb_reserve_back(out_cb, 1);
        tile_regs_acquire();
        load(mix_cb, 0, 0);
        load(broadcast_cb, 0, 7);
        load(3, 0, 6);
        MATH((ROWVM_MATH(scale_mix, 0, 0, 0)));
        cb_pop_front(mix_cb, 1);
        cb_pop_front(broadcast_cb, 1);
        cb_pop_front(3, 1);
    }
    pack_reconfig_data_format(out_cb);
    finish(0, 1);
    cb_push_back(out_cb, 1);
}
void run_striped_rms() {
    const uint32_t task = rowvm_task();
    constexpr uint32_t stripe_count = stripes ? stripes : 1;
    const bool leader = task % stripe_count == 0;
    const uint32_t group = task / stripe_count;
    const uint32_t source = op == GATE_INPUT_STRIPED ? group : (striped_message ? mix_cb : 0);
    constexpr uint32_t weight_cb = striped_message ? 4 : 1;
    [[maybe_unused]] const bool skip = op == GATE_INPUT_STRIPED && extra && group == 0;
    cb_wait_front(source, 1);
    cb_reserve_back(partial_cb, 1);
    tile_regs_acquire();
    load(source, 0, 0);
    MATH((ROWVM_MATH(mix_metric, 0, 1, 2)));
    pack_scratch(2, partial_cb);
    if (leader) {
        leader_coeff(true);
    }

    cb_wait_front(broadcast_cb, 1);
    if constexpr (weighted) {
        cb_wait_front(weight_cb, 1);
    }
    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    load(source, 0, 0);
    load(broadcast_cb, 0, 7);
    if constexpr (weighted) {
        load(weight_cb, 0, 6);
    }
    MATH((ROWVM_MATH(scale_mix, skip, 0, 0)));
    cb_pop_front(source, 1);
    cb_pop_front(broadcast_cb, 1);
    if constexpr (weighted) {
        cb_pop_front(weight_cb, 1);
    }
    pack_scratch(0, out_cb);
}

void run_router() {
    cb_wait_front(0, 1);
    cb_wait_front(1, 1);
    cb_reserve_back(partial_cb, 2);
    tile_regs_acquire();
    copy_init(0);
    copy_tile(0, 0, 0);
    load(1, 0, 1);
    MATH((ROWVM_MATH(sigmoid_operand, 0, 0, 0)));
    MATH((ROWVM_MATH(router_scores, 0, 0, 0)));
    cb_pop_front(0, 1);
    cb_pop_front(1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(partial_cb);
    pack_tile(2, partial_cb);
    pack_tile(1, partial_cb);
    tile_regs_release();
    cb_push_back(partial_cb, 2);

    cb_wait_front(aggregate_cb, 1);
    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    load(aggregate_cb, 0, 0);
    MATH((ROWVM_MATH(router_factor, 0, 0, 0)));
    MATH((ROWVM_MATH(router_gamma, 0, 0, 0)));
    cb_pop_front(aggregate_cb, 1);
    pack_scratch(0, out_cb);
}
}  // namespace

void kernel_main() {
    if constexpr (op == MESSAGE_UP) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(12, 13, out_cb);
    } else {
        compute_kernel_hw_startup(0, out_cb);
    }
    MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, math_impl::init)));
    if constexpr (op == RMS || op == L2) {
        run_norm();
    } else if constexpr (op == ADD || op == GATE || op == MOE_HIDDEN) {
        run_elementwise();
    } else if constexpr (op == CONV || op == CONV_ADD) {
        run_conv();
    } else if constexpr (conv_kv) {
        run_conv();
        pack_reconfig_data_format(out_cb);
        run_norm();
    } else if constexpr (op == MOE_COMBINE) {
        run_combine();
    } else if constexpr (op == GATE_INPUT) {
        run_gate_input();
    } else if constexpr (op == MESSAGE) {
        run_message();
    } else if constexpr (op == MIX) {
        run_mix();
    } else if constexpr (conv_mix) {
        run_conv();
        if constexpr (append_bank) {
            run_bank_append();
        }
        run_mix();
    } else if constexpr (op == ROUTER) {
        run_router();
    } else if constexpr (op == RMS_STRIPED || op == GATE_INPUT_STRIPED) {
        run_striped_rms();
    } else if constexpr (striped_message) {
        if constexpr (op == MESSAGE_UP) {
            run_message_matmuls();
        }
        run_message();
        MATH((math_impl::init_sqrt()));
        run_striped_rms();
    }
}
