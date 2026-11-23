#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_conversions.h"
#endif

using namespace ckernel;
namespace {
constexpr uint32_t half_stripes = get_compile_time_arg_val(0);
constexpr uint32_t grid_x = get_compile_time_arg_val(2);
constexpr uint32_t width = get_compile_time_arg_val(3);
constexpr uint32_t stripes = 2 * half_stripes;
constexpr uint32_t k_tiles = 8, n_tiles = 2;
constexpr uint32_t x_cb = 0, w_cb = 1, agg_h_cb = 3, row_cb = 4, agg_cb = 8, bcast_cb = 9;
constexpr uint32_t p2_cb = 17, partial_cb = 18, coeff_cb = 19, xdense_cb = 20, hrow_cb = 21;

#ifdef TRISC_MATH
namespace math_impl {
using namespace sfpi;
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

void init() { sfpu::sqrt_init<false>(); }

// The RMS steps repeat rowvm's striped RMSNorm arithmetic exactly (compute.cpp mix_metric,
// stripe_rms and scale_mix), so the BF16 controller input is bit-identical to message_input.
sfpi_inline vInt scalar_lane(int element) {
    return vInt((element / 16) * 16 + (element % 16 & ~1));
}

sfpi_inline vFloat splat(uint32_t base, int element) {
    vFloat x = dst_reg[base + (element & 1)];
    v_if(vConstTileId != scalar_lane(element)) { x = 0.0f; }
    v_endif;
    return sum32(x);
}

void sumsq(uint32_t, uint32_t, uint32_t) {
    vFloat sum = 0.0f;
    for (uint32_t v = 0; v < 8; ++v) {
        vFloat x = dst_reg[v];
        sum = x * x + sum;
    }
    const vFloat total = sum32(sum);
    v_if(vConstTileId == scalar_lane(0)) { dst_reg[64] = total; }
    v_endif;
}

void factor(uint32_t, uint32_t, uint32_t) {
    vFloat partials = dst_reg[0];
    v_if((vConstTileId & 3) != 0 || vConstTileId >= int(4 * half_stripes)) { partials = 0.0f; }
    v_endif;
    dst_reg[0] = sfpu::_calculate_sqrt_body_<false, true, false>(sum32(partials) * (1.0f / width) + 1e-6f);
}

void scale(uint32_t, uint32_t, uint32_t) {
    const vFloat inv = splat(7 * 32, 0);
    for (uint32_t v = 0; v < 8; ++v) {
        dst_reg[v] = ::float32_to_bf16_rne(vFloat(dst_reg[v]) * inv);
    }
}

// Stripe partial s of the 64-wide projection occupies vectors 2s and 2s+1.
void reduce(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < 2; ++v) {
        vFloat sum = dst_reg[v];
        for (uint32_t s = 1; s < stripes; ++s) {
            sum += dst_reg[2 * s + v];
        }
        dst_reg[v] = ::float32_to_bf16_rne(sum);
    }
}
}  // namespace math_impl
#define MSG_MATH(FN) _llk_math_eltwise_binary_sfpu_params_(math_impl::FN, 0, 0, 0, VectorMode::None)
#endif

void load(uint32_t cb, uint32_t src, uint32_t dest) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, src, dest);
}

void pack_out(uint32_t dest, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(dest, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    const uint32_t stripe = (get_absolute_logical_y() * grid_x + get_absolute_logical_x()) % stripes;
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, w_cb, p2_cb);
    MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, math_impl::init)));
    cb_wait_front(row_cb, 1);
    cb_reserve_back(partial_cb, 1);
    tile_regs_acquire();
    load(row_cb, 0, 0);
    MATH((MSG_MATH(sumsq)));
    pack_out(2, partial_cb);
    if (stripe % half_stripes == 0) {
        cb_wait_front(agg_cb, 1);
        cb_reserve_back(coeff_cb, 1);
        tile_regs_acquire();
        load(agg_cb, 0, 0);
        MATH((MSG_MATH(factor)));
        cb_pop_front(agg_cb, 1);
        pack_out(0, coeff_cb);
    }
    cb_wait_front(bcast_cb, 1);
    cb_reserve_back(xdense_cb, 1);
    tile_regs_acquire();
    load(row_cb, 0, 0);
    load(bcast_cb, 0, 7);
    MATH((MSG_MATH(scale)));
    cb_pop_front(bcast_cb, 1);
    cb_pop_front(row_cb, 1);
    pack_out(0, xdense_cb);

    cb_wait_front(x_cb, k_tiles);
    cb_wait_front(w_cb, k_tiles * n_tiles);
    cb_reserve_back(p2_cb, n_tiles);
    reconfig_data_format<SrcOrder::Reverse>(x_cb, w_cb);
    matmul_block_init(x_cb, w_cb, 0, n_tiles, 1, k_tiles);
    tile_regs_acquire();
    matmul_block_k(x_cb, w_cb, 0, 0, 0, 0, n_tiles, 1, k_tiles, k_tiles, n_tiles);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(p2_cb);
    for (uint32_t n = 0; n < n_tiles; ++n) {
        pack_tile(n, p2_cb);
    }
    tile_regs_release();
    cb_push_back(p2_cb, n_tiles);
    cb_pop_front(x_cb, k_tiles);
    cb_pop_front(w_cb, k_tiles * n_tiles);
    if (stripe != 0) {
        return;
    }
    cb_wait_front(agg_h_cb, 2);
    cb_reserve_back(hrow_cb, 1);
    tile_regs_acquire();
    load(agg_h_cb, 0, 0);
    load(agg_h_cb, 1, 1);
    MATH((MSG_MATH(reduce)));
    cb_pop_front(agg_h_cb, 2);
    pack_out(0, hrow_cb);
}
