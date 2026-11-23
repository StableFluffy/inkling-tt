#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"

constexpr uint32_t kt = get_compile_time_arg_val(0);
constexpr uint32_t nt = get_compile_time_arg_val(1);
constexpr uint32_t mt = get_compile_time_arg_val(2);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
constexpr uint32_t out_bytes = get_compile_time_arg_val(5);
constexpr uint32_t splits = get_compile_time_arg_val(6);
constexpr bool interleaved = get_compile_time_arg_val(7);
constexpr uint32_t bank_width = (nt + 7) / 8;

void kernel_main() {
    const uint32_t col = get_arg_val<uint32_t>(0);
    const uint32_t width = get_arg_val<uint32_t>(1);
    const uint32_t split = splits > 1 ? get_arg_val<uint32_t>(7) : 0;
#if PREFILL_READ_WEIGHT
    const uint32_t address = get_common_arg_val<uint32_t>(0);
    cb_reserve_back(1, kt * width);
    const uint32_t dst = get_write_ptr(1);
    if constexpr (interleaved) {
        const auto weight = TensorAccessor(TensorAccessorArgs<8>(), address, tile_bytes);
        for (uint32_t k = 0; k < kt; ++k) {
            for (uint32_t n = 0; n < width; ++n) {
                noc_async_read<tile_bytes>(weight.get_noc_addr(k * nt + col + n),
                                           dst + (k * width + n) * tile_bytes, tile_bytes);
            }
        }
    } else {
        const uint32_t bank = col / bank_width;
        const uint32_t bank_col = col % bank_width;
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, address);
        for (uint32_t k = 0; k < kt; ++k) {
            noc_async_read(base + ((split * kt + k) * bank_width + bank_col) * tile_bytes,
                           dst + k * width * tile_bytes, width * tile_bytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(1, kt * width);
#else
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), out_bytes);
    PrefillBroadcast broadcast(3);
    auto write_row = [&](uint32_t m) {
        cb_wait_front(16, width);
        const uint32_t src = get_read_ptr(16);
        for (uint32_t n = 0; n < width; ++n) {
            noc_async_write(src + n * out_bytes, out.get_noc_addr((split * mt + m) * nt + col + n), out_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(16, width);
    };
    for (uint32_t m = 0; m < mt; ++m) {
        cb_reserve_back(0, kt);
        const uint32_t dst = get_write_ptr(0);
        if (broadcast.sender) {
            for (uint32_t k = 0; k < kt; ++k) {
                noc_async_read<tile_bytes>(x.get_noc_addr((m * splits + split) * kt + k),
                                           dst + k * tile_bytes, tile_bytes);
            }
            noc_async_read_barrier();
        }
        if constexpr (splits == 1) {
            broadcast.distribute(dst, kt * tile_bytes);
        }
        cb_push_back(0, kt);
        if (m != 0) {
            write_row(m - 1);
        }
    }
    write_row(mt - 1);
    if constexpr (splits == 1) {
        broadcast.finish();
    }
#endif
}
