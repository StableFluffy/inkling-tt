#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_conversions.h"
#endif

using namespace ckernel;
namespace {
constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
constexpr uint32_t stripes = get_compile_time_arg_val(1);
constexpr uint32_t grid_x = get_compile_time_arg_val(3);
constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
constexpr uint32_t x_cb = 0, w_cb = 1, aggregate_cb = 3, partial_cb = 16, row_cb = 17;
constexpr uint32_t row_vectors = n_tiles;  // 32 FP32 values per SFPU vector
constexpr uint32_t aggregate_pages = (stripes * n_tiles * 128 + 4095) / 4096;
static_assert(aggregate_pages <= 8, "matvec aggregate exceeds FP32 DEST");

#ifdef TRISC_MATH
namespace math_impl {
using namespace sfpi;
void init() {}

// Stripe partial s occupies vectors row_vectors*s .. row_vectors*(s+1)-1.
void reduce(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < row_vectors; ++v) {
        vFloat sum = dst_reg[v];
        for (uint32_t s = 1; s < stripes; ++s) {
            sum += dst_reg[row_vectors * s + v];
        }
        dst_reg[v] = ::float32_to_bf16_rne(sum);
    }
}
}  // namespace math_impl
#endif
}  // namespace

void kernel_main() {
    const uint32_t stripe = (get_absolute_logical_y() * grid_x + get_absolute_logical_x()) % stripes;
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
    if (stripe != 0) {
        return;
    }
    cb_wait_front(aggregate_cb, aggregate_pages);
    cb_reserve_back(row_cb, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(aggregate_cb);
    copy_init(aggregate_cb);
    for (uint32_t p = 0; p < aggregate_pages; ++p) {
        copy_tile(aggregate_cb, p, p);
    }
    MATH((_llk_math_eltwise_binary_sfpu_params_(math_impl::reduce, 0, 0, 0, VectorMode::None)));
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(row_cb);
    pack_tile(0, row_cb);
    tile_regs_release();
    cb_push_back(row_cb, 1);
    cb_pop_front(aggregate_cb, aggregate_pages);
}
