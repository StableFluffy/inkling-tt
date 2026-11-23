// Message-gate epilogue I/O. Tasks are column-major so each worker keeps one
// column's forget and message weights while it walks the rows.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(1);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef PREFILL_WRITE
    const auto out = TensorAccessor(TensorAccessorArgs<2>(), get_common_arg_val<uint32_t>(0), 4096);
    for (uint32_t task = start; task < end; ++task) {
        cb_wait_front(16, 1);
        noc_async_write<4096>(get_read_ptr(16), out.get_noc_addr(task % row_tiles * cols + task / row_tiles), 4096);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
#else
    constexpr uint32_t hidden_bytes = get_compile_time_arg_val(2) ? 2048 : 4096;
    constexpr auto h_args = TensorAccessorArgs<3>();
    constexpr auto f_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0), hidden_bytes);
    const auto forget = TensorAccessor(f_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto message = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), 4096);
    const auto u = TensorAccessor(u_args, get_common_arg_val<uint32_t>(3), 4096);
    const auto m = TensorAccessor(m_args, get_common_arg_val<uint32_t>(4), 4096);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t col = task / row_tiles;
        const uint32_t row = task % row_tiles;
        if (task == start || row == 0) {
            cb_reserve_back(1, 4);
            const uint32_t w = get_write_ptr(1);
            for (uint32_t k = 0; k < 2; ++k) {
                noc_async_read<4096>(forget.get_noc_addr(k * cols + col), w + k * 4096, 4096);
                noc_async_read<4096>(message.get_noc_addr(k * cols + col), w + (2 + k) * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(1, 4);
        }
        cb_reserve_back(0, 2);
        cb_reserve_back(2, 2);
        const uint32_t x = get_write_ptr(0);
        const uint32_t y = get_write_ptr(2);
        for (uint32_t k = 0; k < 2; ++k) {
            noc_async_read<hidden_bytes>(h.get_noc_addr(row * 2 + k), x + k * hidden_bytes, hidden_bytes);
        }
        noc_async_read<4096>(u.get_noc_addr(row * cols + col), y, 4096);
        noc_async_read<4096>(m.get_noc_addr(row * cols + col), y + 4096, 4096);
        noc_async_read_barrier();
        cb_push_back(0, 2);
        cb_push_back(2, 2);
    }
#endif
}
