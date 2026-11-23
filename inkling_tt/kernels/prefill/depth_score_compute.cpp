#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cols = get_compile_time_arg_val(0);
    constexpr uint32_t block = 8;
    constexpr uint32_t k_block = 2;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_block_init(0, 1, false, 1, 1, k_block);
    cb_wait_front(1, cols);
    for (uint32_t task = first; task < end; ++task) {
        // Match the stock projection's two-tile partials and final DEST reload.
        pack_reconfig_data_format(3);
        for (uint32_t col = 0; col < cols; col += block) {
            cb_wait_front(0, block);
            for (uint32_t k = 0; k < block; k += k_block) {
                const bool last = col + k + k_block == cols;
                const bool accumulate = col + k != 0;
                tile_regs_acquire();
                if (last) {
                    cb_wait_front(3, 1);
                    reconfig_data_format_srca(3);
                    copy_init(3);
                    copy_tile(3, 0, 0);
                    cb_pop_front(3, 1);
                    reconfig_full_operand<SrcOrder::Reverse>(0, 1);
                    matmul_block_init(0, 1, false, 1, 1, k_block);
                }
                for (uint32_t i = 0; i < k_block; ++i) {
                    matmul_block(0, 1, k + i, col + k + i, 0, false, 1, 1, k_block);
                }
                tile_regs_commit();
                const uint32_t target = last ? 16 : 3;
                cb_reserve_back(target, 1);
                tile_regs_wait();
                if (last) {
                    pack_reconfig_data_format(16);
                }
                PACK((llk_pack_reconfig_l1_acc(!last && accumulate)));
                pack_tile(0, target);
                tile_regs_release();
                cb_push_back(target, 1);
                if (!last && col + k + 2 * k_block != cols) {
                    cb_wait_front(3, 1);
                    cb_pop_front(3, 1);
                }
            }
            cb_pop_front(0, block);
        }
    }
    cb_pop_front(1, cols);
}
