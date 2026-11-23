#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void load(uint32_t cb, uint32_t page, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, page, dst);
}

void kernel_main() {
    constexpr uint32_t cols = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(2);
    constexpr uint32_t queries = get_compile_time_arg_val(3);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0, 16);
    for (uint32_t tile = first; tile < end; ++tile) {
        cb_wait_front(0, depth);
        cb_wait_front(1, queries * depth);
        cb_reserve_back(16, queries);
        for (uint32_t q = 0; q < queries; ++q) {
            tile_regs_acquire();
            for (uint32_t d = 0; d < depth; ++d) {
                const uint32_t sum = d == 0 ? 0 : 1;
                load(0, d, sum);
                load(1, q * depth + d, 2);
                sfpu_mul_bcast_col_init();
                sfpu_mul_bcast_col(sum, 2);
                if (d) {
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 16);
            tile_regs_release();
        }
        cb_push_back(16, queries);
        cb_pop_front(0, depth);
        if (tile + 1 == end || (tile + 1) % cols == 0) {
            cb_pop_front(1, queries * depth);
        }
    }
}
