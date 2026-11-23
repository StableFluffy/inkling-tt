// Stage the live data once before it fans out to the expert readers.
#include "../common/read_vc.h"

void kernel_main() {
    select_read_vc();
    constexpr uint32_t workers = get_compile_time_arg_val(0);
    constexpr uint32_t pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t live_bytes = get_compile_time_arg_val(3);
    constexpr bool two_faces = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1), page_bytes);
    const uint32_t scratch = get_write_ptr(0);

    for (uint32_t page = get_arg_val<uint32_t>(0); page < pages; page += workers) {
        const uint64_t from = src.get_noc_addr(page);
        const uint64_t to = dst.get_noc_addr(page);
        noc_async_read<live_bytes>(from, scratch, live_bytes);
        if constexpr (two_faces) {
            // Each short DRAM read retains source/destination alignment modulo 64.
            noc_async_read<live_bytes>(from + 512, scratch + 64, live_bytes);
        }
        noc_async_read_barrier();
        noc_async_write<live_bytes>(scratch, to, live_bytes);
        if constexpr (two_faces) {
            noc_async_write<live_bytes>(scratch + 64, to + 512, live_bytes);
        }
        noc_async_write_barrier();
    }
}
