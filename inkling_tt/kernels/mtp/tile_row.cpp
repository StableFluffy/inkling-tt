#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tile = get_compile_time_arg_val(0);
constexpr uint32_t src_row = get_compile_time_arg_val(1);
constexpr uint32_t dst_row = get_compile_time_arg_val(2);
constexpr uint32_t bytes = tile / 64;

uint32_t face_row(uint32_t row, uint32_t half) {
    return (row / 16 * 2 + half) * (tile / 4) + row % 16 * bytes;
}
}

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<3>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0), tile);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1), tile);
    const uint32_t buf = get_write_ptr(0);
    for (uint32_t i = get_arg_val<uint32_t>(0); i < get_arg_val<uint32_t>(1); ++i) {
        noc_async_read<tile>(input.get_noc_addr(i), buf, tile);
        noc_async_read_barrier();
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write<bytes>(buf + face_row(src_row, f), output.get_noc_addr(i, face_row(dst_row, f)), bytes);
        }
        noc_async_write_barrier();
    }
}
