#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t vocab = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_bytes = 32;
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1), row_bytes);
    const uint32_t buf = get_write_ptr(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    for (uint32_t tile = first; tile < end; ++tile) {
        noc_async_read<face_bytes>(src.get_noc_addr(tile), buf, face_bytes);
        noc_async_read<face_bytes>(src.get_noc_addr(tile, 512), buf + 2 * face_bytes, face_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        for (uint32_t i = 0; i < face_bytes / 4; ++i) {
            words[face_bytes / 4 + i] = words[2 * face_bytes / 4 + i];
        }
        if ((tile + 1) * 32 > vocab) {
            auto values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
            for (uint32_t j = 0; j < 32; ++j) {
                if (tile * 32 + j >= vocab) {
                    values[j] = 0xff80u;
                }
            }
        }
        noc_async_write<2 * face_bytes>(buf, dst.get_noc_addr(0, tile * 2 * face_bytes), 2 * face_bytes);
        noc_async_write_barrier();
    }
}
