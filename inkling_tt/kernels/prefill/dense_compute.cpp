#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(2);
    constexpr uint32_t kb = get_compile_time_arg_val(3);
    const uint32_t width = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2) * kb;
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_block_init(0, 1, false, width, 1, 1);
    cb_wait_front(1, kt * width);
    for (uint32_t m = 0; m < mt; ++m) {
        cb_wait_front(0, kt);
        tile_regs_acquire();
        // The decode pipeline rotates K by whole blocks to balance DRAM.
        // Keep that order and the full accumulation in Dest for every row.
        for (uint32_t k = first; k < kt; ++k) {
            matmul_block(0, 1, k, k * width, 0, false, width, 1, 1);
        }
        for (uint32_t k = 0; k < first; ++k) {
            matmul_block(0, 1, k, k * width, 0, false, width, 1, 1);
        }
        cb_pop_front(0, kt);
        tile_regs_commit();
        cb_reserve_back(16, width);
        tile_regs_wait();
        for (uint32_t n = 0; n < width; ++n) {
            pack_tile(n, 16);
        }
        tile_regs_release();
        cb_push_back(16, width);
    }
    cb_pop_front(1, kt * width);
}
