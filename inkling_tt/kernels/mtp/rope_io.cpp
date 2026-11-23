#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const auto arg = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto a0 = TensorAccessorArgs<1>();
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef DRAFT_WRITE
    const auto output = TensorAccessor(a0, arg(0), 2048);
    for (uint32_t row = first; row < end; ++row) {
        const uint32_t index = row / 2 * 4 + row % 2;
        cb_wait_front(16, 2);
        const uint32_t data = get_read_ptr(16);
        noc_async_write<2048>(data, output.get_noc_addr(index), 2048);
        noc_async_write<2048>(data + 2048, output.get_noc_addr(index + 2), 2048);
        noc_async_write_barrier();
        cb_pop_front(16, 2);
    }
#else
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(a0, arg(0), 2048);
    const auto cos = TensorAccessor(a1, arg(1), 4096);
    const auto sin = TensorAccessor(a2, arg(2), 4096);
    for (uint32_t row = first; row < end; ++row) {
        const uint32_t col = row % 2;
        const uint32_t index = row / 2 * 4 + col;
        const uint32_t angle = row % (2 * rows);
        cb_reserve_back(0, 2);
        cb_reserve_back(1, 2);
        const uint32_t data = get_write_ptr(0);
        const uint32_t trig = get_write_ptr(1);
        noc_async_read<2048>(input.get_noc_addr(index), data, 2048);
        noc_async_read<2048>(input.get_noc_addr(index + 2), data + 2048, 2048);
        noc_async_read<4096>(cos.get_noc_addr(angle), trig, 4096);
        noc_async_read<4096>(sin.get_noc_addr(angle), trig + 4096, 4096);
        noc_async_read_barrier();
        cb_push_back(0, 2);
        cb_push_back(1, 2);
    }
#endif
}
