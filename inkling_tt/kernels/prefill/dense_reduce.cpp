#if REDUCE_COMPUTE
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t splits = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 16);
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
        cb_wait_front(0, splits);
        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, 0, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0);
        for (uint32_t k = 1; k < splits; ++k) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, k, 0);
        }
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
        cb_pop_front(0, splits);
    }
}
#else
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t splits = get_compile_time_arg_val(1);
    constexpr auto in_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto partials = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), 2048);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    auto write = [&](uint32_t tile) {
        cb_wait_front(16, 1);
        noc_async_write<2048>(get_read_ptr(16), out.get_noc_addr(tile), 2048);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    };
    for (uint32_t tile = start; tile < end; ++tile) {
        cb_reserve_back(0, splits);
        const uint32_t addr = get_write_ptr(0);
        for (uint32_t k = 0; k < splits; ++k) {
            noc_async_read<4096>(partials.get_noc_addr(k * tiles + tile), addr + k * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(0, splits);
        if (tile != start) {
            write(tile - 1);
        }
    }
    write(end - 1);
}
#endif
