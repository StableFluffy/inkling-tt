#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_row = 32;
    constexpr uint32_t face_bytes = 512;
    constexpr auto matrix_args = TensorAccessorArgs<1>();
    constexpr auto row_args = TensorAccessorArgs<matrix_args.next_compile_time_args_offset()>();
    const auto matrix = TensorAccessor(matrix_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t tile = first; tile < end; ++tile) {
        for (uint32_t i = 0; i < rows; ++i) {
            const auto row = TensorAccessor(row_args, get_common_arg_val<uint32_t>(i + 1), tile_bytes);
            for (uint32_t face = 0; face < 2; ++face) {
                noc_async_read<face_row>(
                    row.get_noc_addr(tile, face * face_bytes), scratch + (i * 2 + face) * 64, face_row);
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t matrix_offset = face * face_bytes + i * face_row;
                noc_async_write<face_row>(
                    scratch + (i * 2 + face) * 64, matrix.get_noc_addr(tile, matrix_offset), face_row);
            }
        }
        noc_async_write_barrier();
    }
}
