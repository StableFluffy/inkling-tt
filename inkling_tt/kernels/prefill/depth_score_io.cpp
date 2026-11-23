#include "api/dataflow/dataflow_api.h"

constexpr uint32_t page = 4096;
constexpr uint32_t cols = get_compile_time_arg_val(0);
constexpr uint32_t block = 8;

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr auto args = TensorAccessorArgs<2>();
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef PREFILL_WRITE
    const auto out = TensorAccessor(args, get_common_arg_val<uint32_t>(0), page);
    for (uint32_t task = first; task < end; ++task) {
        cb_wait_front(16, 1);
        noc_async_write<page>(get_read_ptr(16), out.get_noc_addr(task), page);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
#else
    const auto q = TensorAccessor(args, get_common_arg_val<uint32_t>(0), page);
    cb_reserve_back(1, cols);
    const uint32_t q_dest = get_write_ptr(1);
    for (uint32_t col = 0; col < cols; ++col) {
        noc_async_read<page>(q.get_noc_addr(col), q_dest + col * page, page);
    }
    noc_async_read_barrier();
    cb_push_back(1, cols);
    for (uint32_t task = first; task < end; ++task) {
        const auto x = TensorAccessor(args, get_common_arg_val<uint32_t>(1 + task / rows), page);
        const uint32_t row = task % rows;
        for (uint32_t col = 0; col < cols; col += block) {
            cb_reserve_back(0, block);
            const uint32_t dest = get_write_ptr(0);
            for (uint32_t i = 0; i < block; ++i) {
                noc_async_read<page>(x.get_noc_addr(row * cols + col + i), dest + i * page, page);
            }
            noc_async_read_barrier();
            cb_push_back(0, block);
        }
    }
#endif
}
