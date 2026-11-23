#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t src_stride = get_compile_time_arg_val(1);
    constexpr uint32_t dst_stride = get_compile_time_arg_val(2);
    constexpr uint32_t dst_start = get_compile_time_arg_val(3);
    constexpr uint32_t bytes = 2048;
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0), bytes);
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1), bytes);
    const uint32_t buf = get_write_ptr(0);
    for (uint32_t i = get_arg_val<uint32_t>(0); i < get_arg_val<uint32_t>(1); ++i) {
        const uint32_t group = i / tiles;
        const uint32_t tile = i % tiles;
#ifdef DRAFT_HEADS
        const uint32_t dest = (group % dst_start * dst_stride + group / dst_start) * tiles + tile;
#else
        const uint32_t dest = group * dst_stride + dst_start + tile;
#endif
        noc_async_read<bytes>(src.get_noc_addr(group * src_stride + tile), buf, bytes);
        noc_async_read_barrier();
        noc_async_write<bytes>(buf, dst.get_noc_addr(dest), bytes);
        noc_async_write_barrier();
    }
}
