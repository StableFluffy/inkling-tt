// u * (1 - tanh(h @ Wf)) + m * tanh(h @ Wg) in FP32 for one output tile.
// The two K tiles accumulate in Dest in the order of the separate projection.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t one = 0x3f800000;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t row = task % row_tiles;
        if (task == start || row == 0) {
            cb_wait_front(1, 4);
        }
        cb_wait_front(0, 2);
        cb_wait_front(2, 2);
        tile_regs_acquire();
        reconfig_full_operand<SrcOrder::Reverse>(0, 1);
        matmul_init(0, 1);
        for (uint32_t k = 0; k < 2; ++k) {
            matmul_tiles(0, 1, k, k, 0);
        }
        for (uint32_t k = 0; k < 2; ++k) {
            matmul_tiles(0, 1, k, 2 + k, 1);
        }
        cb_pop_front(0, 2);
        reconfig_data_format_srca(2);
        copy_init(2);
        copy_tile(2, 0, 2);
        copy_tile(2, 1, 3);
        cb_pop_front(2, 2);
        tanh_tile_init<false>();
        tanh_tile<false>(0);
        tanh_tile<false>(1);
        binop_with_scalar_tile_init();
        rsub_unary_tile(0, one);
        mul_binary_tile_init();
        mul_binary_tile(2, 0, 2);
        mul_binary_tile(3, 1, 3);
        add_binary_tile_init();
        add_binary_tile(2, 3, 2);
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(2, 16);
        tile_regs_release();
        cb_push_back(16, 1);
        if (task + 1 == end || (task + 1) % row_tiles == 0) {
            cb_pop_front(1, 4);
        }
    }
}
