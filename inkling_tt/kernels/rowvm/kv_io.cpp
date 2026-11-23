#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t block = get_compile_time_arg_val(0);
    constexpr uint32_t block_bytes = block * 2048;
    constexpr uint32_t blocks = 72 / block;
    const uint32_t column = get_arg_val<uint32_t>(0);
    const uint32_t projection = column / 32;
#ifdef KV_WRITE
    constexpr auto in_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(in_args, get_common_arg_val<uint32_t>(projection), 2048);
    const auto y = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2 + projection), 2048);
    for (uint32_t k = 0; k < blocks; ++k) {
        cb_reserve_back(0, block);
        const uint32_t dest = get_write_ptr(0);
        for (uint32_t j = 0; j < block; ++j) {
            const uint32_t tile = k * block + j;
            const uint32_t local = dest + j * 2048;
            noc_async_read<32>(x.get_noc_addr(tile), local, 32);
            noc_async_read<32>(x.get_noc_addr(tile, 512), local + 512, 32);
        }
        noc_async_read_barrier();
        cb_push_back(0, block);
    }
    cb_wait_front(16, 1);
    noc_async_write_tile(column % 32, y, get_read_ptr(16));
    noc_async_write_barrier();
    cb_pop_front(16, 1);
#else
    const uint32_t bank = column / 8;
    const uint32_t base = get_common_arg_val<uint32_t>(0) + (column % 8) * 72 * 2048;
    const uint64_t remote = get_noc_addr_from_bank_id<true>(bank, base);
    for (uint32_t k = 0; k < blocks; ++k) {
        cb_reserve_back(1, block);
        const uint32_t dst = get_write_ptr(1);
        noc_async_read<block_bytes>(remote + k * block_bytes, dst, block_bytes);
        noc_async_read_barrier();
        cb_push_back(1, block);
    }
#endif
}
