#include "api/dataflow/dataflow_api.h"

void kernel_main() {
#ifdef DRAFT_WRITE
    constexpr uint32_t cb = 16;
    constexpr uint32_t bytes = 2048;
#else
    constexpr uint32_t cb = 0;
    constexpr uint32_t bytes = 4096;
#endif
    const auto tensor = TensorAccessor(TensorAccessorArgs<0>(), get_common_arg_val<uint32_t>(0), bytes);
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
#ifdef DRAFT_WRITE
        cb_wait_front(cb, 1);
        noc_async_write<bytes>(get_read_ptr(cb), tensor.get_noc_addr(tile), bytes);
        noc_async_write_barrier();
        cb_pop_front(cb, 1);
#else
        cb_reserve_back(cb, 1);
        noc_async_read<bytes>(tensor.get_noc_addr(tile), get_write_ptr(cb), bytes);
        noc_async_read_barrier();
        cb_push_back(cb, 1);
#endif
    }
}
