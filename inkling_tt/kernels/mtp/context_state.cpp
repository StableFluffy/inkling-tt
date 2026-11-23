#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t face_bytes = 32;
constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
constexpr uint32_t capacity = get_compile_time_arg_val(1);
constexpr uint32_t cols = 4;

uint32_t face_row(uint32_t row, uint32_t half) {
    return (row / 16 * 2 + half) * 512 + row % 16 * face_bytes;
}
}  // namespace

void kernel_main() {
    constexpr auto meta_args = TensorAccessorArgs<2>();
    constexpr auto src_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(0), 32);
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(2), tile_bytes);
    const uint32_t buf = get_write_ptr(0);
    noc_async_read<32>(meta.get_noc_addr(0), buf, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t start = header[0];
    const uint32_t valid = header[1];
    const uint32_t first = valid > capacity ? valid - capacity : 0;
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
        const uint32_t base_row = (tile / cols % row_tiles) * 32;
        if (base_row >= valid || base_row + 32 <= first) {
            continue;
        }
        const uint32_t head = tile / (cols * row_tiles);
        const uint32_t col = tile % cols;
        noc_async_read<tile_bytes>(src.get_noc_addr(tile), buf, tile_bytes);
        noc_async_read_barrier();
        const uint32_t end = valid < base_row + 32 ? valid : base_row + 32;
        for (uint32_t row = first > base_row ? first : base_row; row < end; ++row) {
            const uint32_t slot = (start + row) % capacity;
            const uint32_t dest = (head * (capacity / 32 + 1) + slot / 32) * cols + col;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t src_offset = face_row(row - base_row, half);
                const uint32_t dst_offset = face_row(slot % 32, half);
                noc_async_write<face_bytes>(buf + src_offset, dst.get_noc_addr(dest, dst_offset), face_bytes);
            }
        }
        noc_async_write_barrier();
    }
}
