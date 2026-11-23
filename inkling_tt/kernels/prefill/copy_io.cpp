// Copy tiles [start, end) of one interleaved tensor into another at a tile offset.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page = get_compile_time_arg_val(0);
    constexpr uint32_t offset = get_compile_time_arg_val(1);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef PREFILL_WRITE
    const auto dst = TensorAccessor(TensorAccessorArgs<2>(), get_common_arg_val<uint32_t>(0), page);
    for (uint32_t i = start; i < end; ++i) {
        cb_wait_front(0, 1);
        noc_async_write<page>(get_read_ptr(0), dst.get_noc_addr(offset + i), page);
        noc_async_write_barrier();
        cb_pop_front(0, 1);
    }
#else
    const auto src = TensorAccessor(TensorAccessorArgs<2>(), get_common_arg_val<uint32_t>(0), page);
    for (uint32_t i = start; i < end; ++i) {
        cb_reserve_back(0, 1);
        noc_async_read<page>(src.get_noc_addr(i), get_write_ptr(0), page);
        noc_async_read_barrier();
        cb_push_back(0, 1);
    }
#endif
}
