#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t tiles_per_head = 4;
constexpr uint32_t group_heads = 3;
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr bool scatter = get_compile_time_arg_val(1);

uint32_t row_offset(uint32_t row, uint32_t face) {
    return (row / 16 * 2 + face) * 512 + row % 16 * 32;
}
}

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), tile_bytes);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
        const uint32_t col = tile % tiles_per_head;
        const uint32_t group = tile / tiles_per_head;
        const uint32_t first = group * group_heads * tiles_per_head + col;
        if constexpr (scatter) {
            noc_async_read<tile_bytes>(src.get_noc_addr(tile), scratch, tile_bytes);
        } else {
            for (uint32_t h = 0; h < group_heads; ++h) {
                noc_async_read<tile_bytes>(
                    src.get_noc_addr(first + h * tiles_per_head), scratch + h * tile_bytes, tile_bytes);
            }
        }
        noc_async_read_barrier();
        for (uint32_t h = 0; h < group_heads; ++h) {
            for (uint32_t i = 0; i < rows; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t compact = row_offset(h * rows + i, f);
                    const uint32_t wide = row_offset(i, f);
                    if constexpr (scatter) {
                        noc_async_write<32>(scratch + compact, out.get_noc_addr(first + h * tiles_per_head, wide), 32);
                    } else {
                        noc_async_write<32>(scratch + h * tile_bytes + wide, out.get_noc_addr(tile, compact), 32);
                    }
                }
            }
        }
        noc_async_write_barrier();
    }
}
