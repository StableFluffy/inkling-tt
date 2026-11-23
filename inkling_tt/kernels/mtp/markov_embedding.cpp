#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = width * 2;
    constexpr uint32_t start_tile = get_compile_time_arg_val(1);
    constexpr auto index_args = TensorAccessorArgs<2>();
    constexpr auto table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    const auto index = TensorAccessor(index_args, get_common_arg_val<uint32_t>(0), 4);
    const auto table = TensorAccessor(table_args, get_common_arg_val<uint32_t>(1), row_bytes);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(2), 2048);
    const uint32_t buf = get_write_ptr(0);
    noc_async_read<32>(index.get_noc_addr(0), buf, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t token = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    noc_async_read<row_bytes>(table.get_noc_addr(token), buf, row_bytes);
    noc_async_read_barrier();
    for (uint32_t i = 0; i < width / 32; ++i) {
        noc_async_write<32>(buf + i * 64, output.get_noc_addr(start_tile + i), 32);
        noc_async_write<32>(buf + i * 64 + 32, output.get_noc_addr(start_tile + i, 512), 32);
    }
    noc_async_write_barrier();
}
