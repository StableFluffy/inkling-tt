#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t lmax = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = 4608;
    constexpr auto pos_args = TensorAccessorArgs<2>();
    constexpr auto embed_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<embed_args.next_compile_time_args_offset()>();
    constexpr auto id_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr auto step_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto tau_args = TensorAccessorArgs<step_args.next_compile_time_args_offset()>();
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const uint32_t row = get_arg_val<uint32_t>(0);
    const auto position = TensorAccessor(pos_args, addr(0), 4);
    const auto embed = TensorAccessor(embed_args, addr(1), row_bytes);
    const auto table = TensorAccessor(table_args, addr(2), lmax * 4);
    const auto id = TensorAccessor(id_args, addr(3 + row), 4);
    const uint32_t outputs = 3 + rows + 3 * row;
    const auto x = TensorAccessor(x_args, addr(outputs), row_bytes);
    const auto step = TensorAccessor(step_args, addr(outputs + 1), 4);
    const auto tau = TensorAccessor(tau_args, addr(outputs + 2), 4096);
    const uint32_t scratch = get_write_ptr(0);
    const auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);

    noc_async_read<4>(position.get_noc_addr(0), scratch, 4);
    noc_async_read<4>(id.get_noc_addr(0), scratch + 64, 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t p = words[0] + row;
    const uint32_t token = words[16];
    if (p >= lmax) {
        return;
    }
    const uint32_t offset = p * 4;
    noc_async_read<row_bytes>(embed.get_noc_addr(token), scratch + 128, row_bytes);
    noc_async_read<64>(table.get_noc_addr(0, offset & ~63u), scratch + 64, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t scale = words[16 + ((offset & 63) / 4)];
    words[0] = p;
    words[16] = scale;
    noc_async_write<row_bytes>(scratch + 128, x.get_noc_addr(0), row_bytes);
    noc_async_write<4>(scratch, step.get_noc_addr(0), 4);
    noc_async_write<4>(scratch + 64, tau.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
