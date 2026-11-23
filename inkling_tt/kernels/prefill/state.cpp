// Commit the valid K/V rows after all attention reads, or extract the last token.
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cache_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cols = get_compile_time_arg_val(2);

uint32_t face_row(uint32_t row, uint32_t half) {
    return (row / 16 * 2 + half) * 512 + row % 16 * 32;
}

void kernel_main() {
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr auto a0 = TensorAccessorArgs<3>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(a0, addr(0), 32);
    const auto src = TensorAccessor(a1, addr(1), tile_bytes);
    const auto dst = TensorAccessor(a2, addr(2), tile_bytes);
    const uint32_t input = get_write_ptr(0);
    const uint32_t output = input + tile_bytes;
    noc_async_read<32>(meta.get_noc_addr(0), input, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto descriptor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input);
    const uint32_t start = descriptor[0] / 32;
    const uint32_t length = descriptor[1];

#ifdef PREFILL_LAST
    const uint32_t final_tile = (length - 1) / 32;
    const uint32_t final_row = (length - 1) % 32;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t col = get_arg_val<uint32_t>(0); col < get_arg_val<uint32_t>(1); ++col) {
        noc_async_read<tile_bytes>(src.get_noc_addr(final_tile * cols + col), input, tile_bytes);
        for (uint32_t i = 0; i < tile_bytes; i += MEM_ZEROS_SIZE) {
            noc_async_read<MEM_ZEROS_SIZE>(zeros, output + i, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t half = 0; half < 2; ++half) {
            noc_async_read<32>(get_noc_addr(input + face_row(final_row, half)),
                               output + face_row(0, half), 32);
        }
        noc_async_read_barrier();
        noc_async_write<tile_bytes>(output, dst.get_noc_addr(col), tile_bytes);
        noc_async_write_barrier();
    }
#else
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    const auto src_v = TensorAccessor(a3, addr(3), tile_bytes);
    const auto dst_v = TensorAccessor(a4, addr(4), tile_bytes);
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        const uint32_t row = task / cols % q_tiles;
        const uint32_t head = task / (cols * q_tiles);
        const uint32_t col = task % cols;
        // A chunk longer than a ring keeps only its newest rows: skip a tile that a
        // later valid tile of the same chunk overwrites.
        if (row * 32 >= length || (row + cache_tiles) * 32 < length) {
            continue;
        }
        const uint32_t valid = length - row * 32;
        const uint32_t dest_tile = (head * cache_tiles + (start + row) % cache_tiles) * cols + col;
        for (uint32_t i = 0; i < 2; ++i) {
            const uint64_t from = i ? src_v.get_noc_addr(task) : src.get_noc_addr(task);
            const uint64_t to = i ? dst_v.get_noc_addr(dest_tile) : dst.get_noc_addr(dest_tile);
            noc_async_read<tile_bytes>(from, input, tile_bytes);
            if (valid < 32) {
                // Rows past the end keep the newest earlier value: the same chunk's
                // tile one ring length back if there is one, else the cache.
                const uint32_t earlier = task - cache_tiles * cols;
                const uint64_t keep = row >= cache_tiles
                    ? (i ? src_v.get_noc_addr(earlier) : src.get_noc_addr(earlier)) : to;
                noc_async_read<tile_bytes>(keep, output, tile_bytes);
            }
            noc_async_read_barrier();
            if (valid < 32) {
                for (uint32_t half = 0; half < 2; ++half) {
                    for (uint32_t row = valid; row < 32;) {
                        const uint32_t n = 16 - row % 16;
                        const uint32_t offset = face_row(row, half);
                        noc_async_read(get_noc_addr(output + offset), input + offset, n * 32);
                        row += n;
                    }
                }
                noc_async_read_barrier();
            }
            noc_async_write<tile_bytes>(input, to, tile_bytes);
            noc_async_write_barrier();
        }
    }
#endif
}
