#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t x = 0, w = 1, partial = 2, out = 16;
    constexpr uint32_t block = get_compile_time_arg_val(0);
    constexpr uint32_t blocks = 72 / block;
    compute_kernel_hw_startup<SrcOrder::Reverse>(x, w, partial);
    matmul_block_init(x, w, 0, 1, 1, block);
    pack_reconfig_data_format(partial);
    for (uint32_t i = 0; i < blocks; ++i) {
        cb_wait_front(x, block);
        cb_wait_front(w, block);
        tile_regs_acquire();
        if (i == blocks - 1) {
            cb_wait_front(partial, 1);
            reconfig_data_format_srca(w, partial);
            copy_init(partial);
            copy_tile(partial, 0, 0);
            cb_pop_front(partial, 1);
            reconfig_data_format_srca(partial, w);
            matmul_block_init(x, w, 0, 1, 1, block);
        }
        for (uint32_t k = 0; k < block; ++k) {
            matmul_block(x, w, k, k, 0, 0, 1, 1, block);
        }
        tile_regs_commit();
        const uint32_t target = i == blocks - 1 ? out : partial;
        cb_reserve_back(target, 1);
        tile_regs_wait();
        if (i == blocks - 1) {
            pack_reconfig_data_format(out);
            pack_reconfig_l1_acc(0);
        } else if (i < 2) {
            pack_reconfig_l1_acc(i);
        }
        pack_tile(0, target);
        tile_regs_release();
        cb_push_back(target, 1);
        if (i < blocks - 2) {
            cb_wait_front(partial, 1);
            cb_pop_front(partial, 1);
        }
        cb_pop_front(x, block);
        cb_pop_front(w, block);
    }
}
