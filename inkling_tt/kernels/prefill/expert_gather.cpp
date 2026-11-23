// Gather routed activation rows into full [32,32] tiles.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t map_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t token_tiles = get_compile_time_arg_val(1);
    constexpr auto map_args = TensorAccessorArgs<2>();
    constexpr auto ends_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<ends_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto map = TensorAccessor(map_args, get_common_arg_val<uint32_t>(0), map_bytes);
    const auto ends = TensorAccessor(ends_args, get_common_arg_val<uint32_t>(1), 512);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(2), 2048);
#if EXPERTS_WIDE_L1_BLOCKS
    constexpr auto tail_args = out_args;
    constexpr auto wide_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
    const auto tail = TensorAccessor(tail_args, get_common_arg_val<uint32_t>(3), 2048);
    const auto out = TensorAccessor(wide_args, get_common_arg_val<uint32_t>(4), 2048);
#else
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(3), 2048);
#endif

    const uint32_t scratch = get_write_ptr(0);
    const uint32_t map_l1 = get_write_ptr(2);
    noc_async_read<32>(ends.get_noc_addr(0) + 480, scratch, 32);
    noc_async_read(map.get_noc_addr(0), map_l1, map_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t blocks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[7];
    const auto tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(map_l1);
    const uint32_t col = get_arg_val<uint32_t>(0);
    // Cache this feature column in L1. DRAM reads require matching
    // 64-byte offsets, while row gathers use 32-byte face rows.
    const uint32_t cache = (get_write_ptr(3) + 63) & ~63u;
    for (uint32_t t = 0; t < token_tiles; ++t) {
        noc_async_read<2048>(x.get_noc_addr(t * 72 + col), cache + t * 2048, 2048);
    }
    noc_async_read_barrier();

    const uint64_t zero = get_noc_addr(MEM_ZEROS_BASE);
    // Two scratch tiles: gather the next block while the previous write drains.
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t tile = scratch + block % 2 * 2048;
        noc_async_writes_flushed();
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t token = tokens[block * 32 + r];
            uint64_t left = zero;
            uint64_t right = zero;
            if (token != 0) {
                const uint32_t t = token - 1;
                left = get_noc_addr(cache + t / 32 * 2048 + (t / 16 % 2) * 1024 + (t % 16) * 32);
                right = left + 512;
            }
            const uint32_t row = r / 16 * 1024 + r % 16 * 32;
            noc_async_read<32>(left, tile + row, 32);
            noc_async_read<32>(right, tile + row + 512, 32);
        }
        noc_async_read_barrier();
#if EXPERTS_WIDE_L1_BLOCKS
        const uint64_t addr = block < EXPERTS_WIDE_L1_BLOCKS
            ? out.get_noc_addr(block * 72 + col)
            : tail.get_noc_addr((block - EXPERTS_WIDE_L1_BLOCKS) * 72 + col);
#else
        const uint64_t addr = out.get_noc_addr(block * 72 + col);
#endif
        noc_async_write<2048>(tile, addr, 2048);
    }
    noc_async_write_barrier();
}
