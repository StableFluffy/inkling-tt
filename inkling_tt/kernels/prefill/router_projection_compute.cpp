#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_block_init(0, 1, false, n_tiles, 1, 1);
    for (uint32_t task = first; task < end; ++task) {
        if (task == first || task % rows == 0) {
            cb_wait_front(1, k_tiles * n_tiles);
        }
        cb_wait_front(0, k_tiles);
        // Keep the stock one-tile K partials and final FP32 reload.
        pack_reconfig_data_format(3);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            const bool last = k + 1 == k_tiles;
            tile_regs_acquire();
            if (last) {
                cb_wait_front(3, n_tiles);
                reconfig_data_format_srca(3);
                copy_init(3);
                for (uint32_t n = 0; n < n_tiles; ++n) {
                    copy_tile(3, n, n);
                }
                cb_pop_front(3, n_tiles);
                reconfig_full_operand<SrcOrder::Reverse>(0, 1);
                matmul_block_init(0, 1, false, n_tiles, 1, 1);
            }
            matmul_block(0, 1, k, k * n_tiles, 0, false, n_tiles, 1, 1);
            const uint32_t target = last ? 16 : 3;
            tile_regs_commit();
            cb_reserve_back(target, n_tiles);
            tile_regs_wait();
            if (last) {
                pack_reconfig_data_format(16);
            }
            PACK((llk_pack_reconfig_l1_acc(k != 0 && !last)));
            for (uint32_t n = 0; n < n_tiles; ++n) {
                pack_tile(n, target);
            }
            tile_regs_release();
            cb_push_back(target, n_tiles);
            if (k + 2 < k_tiles) {
                cb_wait_front(3, n_tiles);
                cb_pop_front(3, n_tiles);
            }
        }
        cb_pop_front(0, k_tiles);
        if (task + 1 == end || (task + 1) % rows == 0) {
            cb_pop_front(1, k_tiles * n_tiles);
        }
    }
}
