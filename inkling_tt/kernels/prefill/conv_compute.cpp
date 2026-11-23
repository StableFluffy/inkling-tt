#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
void round_bf16(uint32_t, uint32_t, uint32_t) {
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::dst_reg[i] = ::float32_to_bf16_rne(sfpi::vFloat(sfpi::dst_reg[i]));
    }
}
#endif

void load(uint32_t cb, uint32_t page, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, page, dst);
}

void kernel_main() {
    constexpr bool dynamic = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(0, 16);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);
    for (uint32_t col = get_arg_val<uint32_t>(0); col < get_arg_val<uint32_t>(1); ++col) {
        cb_wait_front(1, 4);
        for (uint32_t t = 0; t < tiles; ++t) {
            cb_wait_front(0, 4);
            if constexpr (dynamic) {
                cb_wait_front(2, 4);
            }
            tile_regs_acquire();
            for (uint32_t lag = 0; lag < 4; ++lag) {
                const uint32_t sum = lag == 0 ? 0 : 1;
                load(0, lag, sum);
                if constexpr (dynamic) {
                    copy_tile(0, lag, 3);
                }
                load(1, lag, 2);
                sfpu_mul_bcast_row_init();
                sfpu_mul_bcast_row(sum, 2);
                if constexpr (dynamic) {
                    load(2, lag, 2);
                    binop_with_scalar_tile_init();
                    mul_unary_tile(2, 0x3eaaaaab, VectorMode::C);
                    sfpu_mul_bcast_col_init();
                    sfpu_mul_bcast_col(3, 2);
                    add_binary_tile_init();
                    add_binary_tile(sum, 3, sum);
                }
                if (lag != 0) {
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
            }
            MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, 0, 0, 0, VectorMode::None)));
            tile_regs_commit();
            cb_reserve_back(16, 1);
            tile_regs_wait();
            pack_reconfig_data_format(16);
            pack_tile(0, 16);
            tile_regs_release();
            cb_push_back(16, 1);
            cb_pop_front(0, 4);
            if constexpr (dynamic) {
                cb_pop_front(2, 4);
            }
        }
        cb_pop_front(1, 4);
    }
}
