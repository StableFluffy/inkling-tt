#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tile_bytes = 4096;
constexpr uint32_t face_bytes = 64;

uint32_t face_row(uint32_t row, uint32_t half) {
    return (row / 16 * 2 + half) * 1024 + row % 16 * face_bytes;
}
}

void kernel_main() {
    constexpr uint32_t capacity = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr auto pos_args = TensorAccessorArgs<2>();
    const auto pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(0), 32);
    uint32_t buf = get_write_ptr(0);
    noc_async_read<32>(pos.get_noc_addr(0), buf, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t start = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    constexpr auto first_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef DRAFT_MASK
    const auto mask = TensorAccessor(first_args, get_common_arg_val<uint32_t>(1), tile_bytes);
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < end; ++tile) {
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t j = tile * 32 + half * 16 + i;
                const bool active = j < capacity ? j < start : j < capacity + rows;
                words[i] = active ? 0 : 0xff800000u;
            }
            noc_async_write<face_bytes>(buf, mask.get_noc_addr(tile, face_row(0, half)), face_bytes);
            noc_async_write_barrier();
        }
    }
#else
    constexpr uint32_t rope_cols = 2;
    constexpr auto cos_args = TensorAccessorArgs<first_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(first_args, get_common_arg_val<uint32_t>(1), 512);
    const auto cos = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(2), tile_bytes);
    const auto sin = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(3), tile_bytes);
    for (uint32_t task = get_arg_val<uint32_t>(0); task < end; ++task) {
        const uint32_t segment = task / (rows / 32 * rope_cols);
        const uint32_t tile = task % (rows / 32 * rope_cols);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t p = start + tile / rope_cols * 32 + i;
            const uint32_t row = p < capacity ? p : capacity - 1;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t col = segment * 256 + tile % rope_cols * 128 + half * face_bytes;
                noc_async_read<face_bytes>(table.get_noc_addr(row, col), buf + face_row(i, half), face_bytes);
            }
        }
        noc_async_read_barrier();
        const uint64_t dst = segment ? sin.get_noc_addr(tile) : cos.get_noc_addr(tile);
        noc_async_write<tile_bytes>(buf, dst, tile_bytes);
        noc_async_write_barrier();
    }
#endif
}
