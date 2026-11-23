#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reduce.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
void round_bf16(uint32_t tile, uint32_t, uint32_t) {
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t lane = tile * 32 + i;
        sfpi::dst_reg[lane] = ::float32_to_bf16_rne(sfpi::vFloat(sfpi::dst_reg[lane]));
    }
}
#endif

void kernel_main() {
    constexpr uint32_t cols = get_compile_time_arg_val(0);
    constexpr bool weighted = get_compile_time_arg_val(1);
    constexpr bool bf16 = get_compile_time_arg_val(2);
    constexpr uint32_t inv_width = get_compile_time_arg_val(5);
    constexpr uint32_t block = 4;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(0, 0, 3);
    cb_wait_front(1, 1);
    cb_wait_front(2, 1);
    for (uint32_t row = start; row < end; ++row) {
        reconfig_data_format(0, 0);
        pack_reconfig_data_format(3);
        mul_init(0, 0);
        for (uint32_t col = 0; col < cols; col += block) {
            cb_wait_front(0, col + block);
            cb_reserve_back(3, block);
            tile_regs_acquire();
            for (uint32_t c = 0; c < block; ++c) {
                mul_tiles(0, 0, col + c, col + c, c);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < block; ++c) {
                pack_tile(c, 3);
            }
            tile_regs_release();
            cb_push_back(3, block);
        }

        cb_wait_front(3, cols);
        reconfig_data_format(1, 3);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(3, 1, 4);
        tile_regs_acquire();
        for (uint32_t col = 0; col < cols; ++col) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(3, 1, col, 0, 0);
        }
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_width);
        tile_regs_commit();
        cb_reserve_back(4, 1);
        tile_regs_wait();
        pack_reconfig_data_format(4);
        pack_tile(0, 4);
        tile_regs_release();
        cb_push_back(4, 1);
        cb_pop_front(3, cols);

        cb_wait_front(4, 1);
        reconfig_data_format(4, 2);
        add_init(4, 2);
        tile_regs_acquire();
        add_tiles(4, 2, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        cb_reserve_back(5, 1);
        tile_regs_wait();
        pack_tile(0, 5);
        tile_regs_release();
        cb_push_back(5, 1);
        cb_pop_front(4, 1);

        cb_wait_front(5, 1);
        pack_reconfig_data_format(16);
        for (uint32_t col = 0; col < cols; col += block) {
            cb_reserve_back(16, block);
            reconfig_data_format(0, 5);
            mul_bcast_cols_init(0, 5);
            tile_regs_acquire();
            for (uint32_t c = 0; c < block; ++c) {
                mul_tiles_bcast_cols(0, 5, col + c, 0, c);
            }
            if constexpr (weighted) {
                cb_wait_front(6, block);
                reconfig_data_format_srca(6);
                copy_init(6);
                for (uint32_t c = 0; c < block; ++c) {
                    copy_tile(6, c, block + c);
                }
                sfpu_mul_bcast_row_init();
                for (uint32_t c = 0; c < block; ++c) {
                    sfpu_mul_bcast_row(c, block + c);
                }
                cb_pop_front(6, block);
            }
            if constexpr (bf16) {
                for (uint32_t c = 0; c < block; ++c) {
                    MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, c, 0, 0, VectorMode::None)));
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < block; ++c) {
                pack_tile(c, 16);
            }
            tile_regs_release();
            cb_push_back(16, block);
        }
        cb_pop_front(0, cols);
        cb_pop_front(5, 1);
    }
    cb_pop_front(1, 1);
    cb_pop_front(2, 1);
}
