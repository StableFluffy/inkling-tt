#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t vocab = get_compile_time_arg_val(0);
    constexpr uint32_t padded = get_compile_time_arg_val(1);
    constexpr uint32_t bf_tile = 2048;
    constexpr uint32_t fp_tile = 4096;
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr auto first_args = TensorAccessorArgs<2>();
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef DRAFT_WRITE
    const auto output = TensorAccessor(first_args, addr(0), padded * 2);
    const uint32_t scratch = get_write_ptr(3);
    for (uint32_t tile = first; tile < end; ++tile) {
        cb_wait_front(16, 1);
        const auto left = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(16));
        const auto right = left + 256;
        auto logits = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        invalidate_l1_cache();
        for (uint32_t i = 0; i < 16; ++i) {
            logits[i] = tile * 32 + i < vocab ? left[i] : 0xff80u;
            logits[i + 16] = tile * 32 + i + 16 < vocab ? right[i] : 0xff80u;
        }
        noc_async_write<64>(scratch, output.get_noc_addr(0, tile * 64), 64);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
#else
    constexpr auto second_args = TensorAccessorArgs<first_args.next_compile_time_args_offset()>();
    constexpr auto third_args = TensorAccessorArgs<second_args.next_compile_time_args_offset()>();
    const auto correction = TensorAccessor(first_args, addr(0), fp_tile);
    const auto bias = TensorAccessor(second_args, addr(1), bf_tile);
    const auto base = TensorAccessor(third_args, addr(2), bf_tile);
    for (uint32_t tile = first; tile < end; ++tile) {
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        cb_reserve_back(2, 1);
        noc_async_read_tile(tile, correction, get_write_ptr(0));
        noc_async_read_tile(tile, bias, get_write_ptr(1));
        noc_async_read_tile(tile, base, get_write_ptr(2));
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
        cb_push_back(2, 1);
    }
#endif
}
