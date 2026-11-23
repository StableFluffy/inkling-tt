#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    constexpr uint32_t tile = 2048;
    constexpr uint32_t columns = 72;
    constexpr uint32_t slots = 8;
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const auto rt = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef EXPERTS_VERIFY_WRITE
    const auto out = TensorAccessor(a0, rt(0), tile);
    const uint32_t scratch = get_write_ptr(3);
    uint32_t current_row = 32;
    for (uint32_t i = start; i < end; ++i) {
        const uint32_t row = i / columns;
        if (row != current_row) {
            current_row = row;
            const auto gamma = TensorAccessor(a1, rt(1 + row), 4096);
            noc_async_read<32>(gamma.get_noc_addr(0), scratch, 32);
            noc_async_read_barrier();
            invalidate_l1_cache();
        }
        const auto weights = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t s = 0; s < slots; ++s) {
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, weights[s]);
        }
        cb_wait_front(16, 1);
        const uint32_t source = get_read_ptr(16);
        const uint32_t col = i % columns;
        noc_async_write<32>(source, out.get_noc_addr(col, row * 32), 32);
        noc_async_write<32>(source + 512, out.get_noc_addr(col, 512 + row * 32), 32);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
#else
    const auto inverse = TensorAccessor(a0, rt(0), 1024);
    const auto down = TensorAccessor(a1, rt(1), 512);
    const uint32_t map = get_write_ptr(2);
    noc_async_read<1024>(inverse.get_noc_addr(0), map, 1024);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto compact = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(map);
    for (uint32_t i = start; i < end; ++i) {
        const uint32_t row = i / columns;
        const uint32_t col = i % columns;
        for (uint32_t s = 0; s < slots; ++s) {
            cb_reserve_back(0, 1);
            const uint32_t dst = get_write_ptr(0);
            const uint32_t entry = compact[row * slots + s] - 1;
            const uint64_t from = down.get_noc_addr(entry / 8 * columns + col, entry % 8 * 32);
            noc_async_read<32>(from, dst, 32);
            noc_async_read<32>(from + 256, dst + 512, 32);
            noc_async_read_barrier();
            cb_push_back(0, 1);
        }
    }
#endif
}
