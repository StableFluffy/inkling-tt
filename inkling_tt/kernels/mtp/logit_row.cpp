#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t vocab = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr bool fp32 = get_compile_time_arg_val(2);
    constexpr uint32_t face_bytes = fp32 ? 64 : 32;
    constexpr uint32_t tile_bytes = 64 * face_bytes;
    constexpr uint32_t bf_row = 64;
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1), row_bytes);
    const uint32_t buf = get_write_ptr(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    for (uint32_t tile = first; tile < end; ++tile) {
        noc_async_read<face_bytes>(src.get_noc_addr(tile), buf, face_bytes);
        noc_async_read<face_bytes>(src.get_noc_addr(tile, tile_bytes / 4), buf + bf_row, face_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        if constexpr (fp32) {
            auto values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + 2 * bf_row);
            for (uint32_t i = 0; i < 32; ++i) {
                const uint32_t bits = words[i];
                values[i] = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
            }
        } else {
            for (uint32_t i = 0; i < face_bytes / 4; ++i) {
                words[face_bytes / 4 + i] = words[bf_row / 4 + i];
            }
        }
        const uint32_t output = fp32 ? buf + 2 * bf_row : buf;
        if ((tile + 1) * 32 > vocab) {
            auto values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(output);
            for (uint32_t j = 0; j < 32; ++j) {
                if (tile * 32 + j >= vocab) {
                    values[j] = 0xff80u;
                }
            }
        }
        noc_async_write<bf_row>(output, dst.get_noc_addr(0, tile * bf_row), bf_row);
        noc_async_write_barrier();
    }
}
