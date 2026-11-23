#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto id_args = TensorAccessorArgs<1>();
    constexpr auto pos_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr auto meta_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto result_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const auto pos = TensorAccessor(pos_args, addr(0), 4);
    const auto meta = TensorAccessor(meta_args, addr(2 * rows), 32);
    const auto result = TensorAccessor(result_args, addr(2 * rows + 1), 64);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read<4>(pos.get_noc_addr(0), scratch, 4);
    for (uint32_t r = 0; r < 2 * rows - 1; ++r) {
        const auto id = TensorAccessor(id_args, addr(r + 1), 4);
        noc_async_read<4>(id.get_noc_addr(0), scratch + (r + 1) * 64, 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t start = in[0];
    uint32_t accepted = 0;
    while (accepted < rows - 1 && in[(1 + accepted) * 16] == in[(1 + rows + accepted) * 16]) {
        ++accepted;
    }
    const uint32_t count = accepted + 1;
    const uint32_t output = scratch + 2 * rows * 64;
    auto out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output);
    out[0] = start;
    out[1] = count;
    for (uint32_t i = 2; i < 8; ++i) {
        out[i] = 0;
    }
    noc_async_write<32>(output, meta.get_noc_addr(0), 32);
    noc_async_write_barrier();
    out[0] = count;
    out[1] = start + count;
    for (uint32_t r = 0; r < rows; ++r) {
        out[r + 2] = r < count ? in[(1 + r) * 16] : 0;
    }
    for (uint32_t i = rows + 2; i < 16; ++i) {
        out[i] = 0;
    }
    noc_async_write<64>(output, result.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
