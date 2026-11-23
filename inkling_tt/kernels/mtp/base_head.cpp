#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t total_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t grid_width = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
#ifdef DRAFT_BASE_ROWS
    constexpr uint32_t rows = DRAFT_BASE_ROWS;
#endif
    const uint32_t addr = get_common_arg_val<uint32_t>(0);
#ifndef DRAFT_BASE_ROWS
    const auto output = TensorAccessor(TensorAccessorArgs<3>(), get_common_arg_val<uint32_t>(1), tile_bytes);
#endif
    const uint32_t first = (get_absolute_logical_y() * grid_width + get_absolute_logical_x()) * shard_tiles;
    for (uint32_t i = 0; i < shard_tiles && first + i < total_tiles; ++i) {
#ifdef DRAFT_BASE_ROWS
        for (uint32_t r = 0; r < rows; ++r) {
            const auto out = TensorAccessor(TensorAccessorArgs<3>(), get_common_arg_val<uint32_t>(r + 1), tile_bytes);
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t offset = face * 512;
                noc_async_write<32>(addr + i * tile_bytes + offset + (r + 1) * 32,
                                    out.get_noc_addr(first + i, offset), 32);
            }
        }
#else
        noc_async_write<tile_bytes>(addr + i * tile_bytes, output.get_noc_addr(first + i), tile_bytes);
#endif
    }
    noc_async_write_barrier();
}
