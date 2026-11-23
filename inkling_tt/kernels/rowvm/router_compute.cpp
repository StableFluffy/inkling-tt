#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

using namespace ckernel;
namespace {
constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
constexpr uint32_t stripes = get_compile_time_arg_val(1);
constexpr uint32_t grid_x = get_compile_time_arg_val(2);
constexpr uint32_t n_tiles = 4;
constexpr uint32_t x_cb = 0, w_cb = 1, bias_cb = 2, aggregate_cb = 3, selection_cb = 8;
constexpr uint32_t partial_cb = 16, gamma_cb = 17, scores_cb = 18;
constexpr uint32_t aggregate_pages = (stripes * 512 + 4095) / 4096;

#ifdef TRISC_MATH
namespace math_impl {
using namespace sfpi;
constexpr uint32_t vectors_per_tile = 32;
constexpr uint32_t bias_slot = aggregate_pages * vectors_per_tile;
constexpr uint32_t scores_slot = bias_slot + vectors_per_tile;
constexpr uint32_t biased_slot = scores_slot + vectors_per_tile;
static_assert(aggregate_pages + 3 <= 8, "router aggregate exceeds FP32 DEST");

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

sfpi_inline vFloat reciprocal(vFloat x) {
    vFloat r = approx_recip(x);
    r += r * (1.0f - x * r);
    r += r * (1.0f - x * r);
    return r;
}

void init() { sfpu::sigmoid_init<false>(); }

// Dense stripe partial k occupies vectors 4k..4k+3 of the aggregate tiles.
void scores(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < 4; ++v) {
        vFloat logit = dst_reg[v];
        for (uint32_t k = 1; k < stripes; ++k) {
            logit += dst_reg[4 * k + v];
        }
        const vFloat score = sfpu::_sfpu_sigmoid_(logit);
        dst_reg[scores_slot + v] = score;
        dst_reg[biased_slot + v] = score + vFloat(dst_reg[bias_slot + v]);
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

#define ROUTER_MATH(FN) _llk_math_eltwise_binary_sfpu_params_(math_impl::FN, 0, 0, 0, VectorMode::None)
#endif

void load(uint32_t cb, uint32_t src, uint32_t dest) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, src, dest);
}

void pack_one(uint32_t dest, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(dest, cb);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    const uint32_t task = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, w_cb, partial_cb);
    MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, math_impl::init)));
    cb_wait_front(x_cb, k_tiles);
    cb_wait_front(w_cb, k_tiles * n_tiles);
    cb_reserve_back(partial_cb, n_tiles);
    pack_reconfig_data_format(partial_cb);
    reconfig_data_format<SrcOrder::Reverse>(x_cb, w_cb);
    matmul_block_init(x_cb, w_cb, 0, n_tiles, 1, k_tiles);
    tile_regs_acquire();
    matmul_block_k(x_cb, w_cb, 0, 0, 0, 0, n_tiles, 1, k_tiles, k_tiles, n_tiles);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t n = 0; n < n_tiles; ++n) {
        pack_tile(n, partial_cb);
    }
    tile_regs_release();
    cb_push_back(partial_cb, n_tiles);
    cb_pop_front(x_cb, k_tiles);
    cb_pop_front(w_cb, k_tiles * n_tiles);
    if (task != 0) {
        return;
    }
    cb_wait_front(aggregate_cb, aggregate_pages);
    cb_wait_front(bias_cb, 1);
    cb_reserve_back(scores_cb, 2);
    tile_regs_acquire();
    reconfig_data_format_srca(aggregate_cb);
    copy_init(aggregate_cb);
    for (uint32_t p = 0; p < aggregate_pages; ++p) {
        copy_tile(aggregate_cb, p, p);
    }
    load(bias_cb, 0, aggregate_pages);
    MATH((ROUTER_MATH(scores)));
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(scores_cb);
    pack_tile(aggregate_pages + 1, scores_cb);
    pack_tile(aggregate_pages + 2, scores_cb);
    tile_regs_release();
    cb_push_back(scores_cb, 2);
    cb_pop_front(aggregate_cb, aggregate_pages);
    cb_pop_front(bias_cb, 1);

    cb_wait_front(selection_cb, 1);
    cb_reserve_back(gamma_cb, 1);
    tile_regs_acquire();
    load(selection_cb, 0, 0);
    MATH((ROUTER_MATH(router_factor)));
    MATH((ROUTER_MATH(router_gamma)));
    cb_pop_front(selection_cb, 1);
    pack_one(0, gamma_cb);
    cb_push_back(gamma_cb, 1);
}
