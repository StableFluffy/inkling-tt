#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t total_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t grid_width = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(TensorAccessorArgs<3>(), get_common_arg_val<uint32_t>(1), tile_bytes);
    const uint32_t first = (get_absolute_logical_y() * grid_width + get_absolute_logical_x()) * shard_tiles;
    for (uint32_t i = 0; i < shard_tiles && first + i < total_tiles; ++i) {
        noc_async_write<tile_bytes>(addr + i * tile_bytes, output.get_noc_addr(first + i), tile_bytes);
    }
    noc_async_write_barrier();
}
