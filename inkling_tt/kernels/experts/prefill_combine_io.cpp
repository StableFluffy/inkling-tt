#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#if PREFILL_WRITE
    const auto out = TensorAccessor(TensorAccessorArgs<0>(), get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t i = start; i < end; ++i) {
        cb_wait_front(16, 1);
        noc_async_write<2048>(get_read_ptr(16), out.get_noc_addr(i), 2048);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
#else
    constexpr uint32_t map_bytes = get_compile_time_arg_val(0);
    constexpr auto map_args = TensorAccessorArgs<2>();
    constexpr auto down_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<down_args.next_compile_time_args_offset()>();
    const auto map = TensorAccessor(map_args, get_common_arg_val<uint32_t>(0), map_bytes);
    const auto down = TensorAccessor(down_args, get_common_arg_val<uint32_t>(1), 512);
    const auto gamma = TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(2), 4096);
    const uint32_t map_l1 = get_write_ptr(2);
    const auto routed_rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(map_l1);
    const uint64_t zero = get_noc_addr(MEM_ZEROS_BASE);

    for (uint32_t i = start; i < end; ++i) {
        const uint32_t c = i % 72;
        if (i == start || c == 0) {
            noc_async_read<map_bytes>(map.get_noc_addr(i / 72), map_l1, map_bytes);
            cb_reserve_back(1, 8);
            uint32_t gamma_l1 = get_write_ptr(1);
            for (uint32_t slot = 0; slot < 8; ++slot) {
                noc_async_read<4096>(gamma.get_noc_addr(i / 72 * 8 + slot), gamma_l1 + slot * 4096, 4096);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            cb_push_back(1, 8);
        }
        for (uint32_t slot = 0; slot < 8; ++slot) {
            cb_reserve_back(0, 1);
            const uint32_t dst = get_write_ptr(0);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t row = routed_rows[r * 8 + slot];
                uint64_t left = zero;
                uint64_t right = zero;
                if (row != 0) {
                    const uint32_t compact = row - 1;
                    left = down.get_noc_addr(compact / 8 * 72 + c) + compact % 8 * 32;
                    right = left + 256;
                }
                const uint32_t face_row = r / 16 * 1024 + r % 16 * 32;
                noc_async_read<32>(left, dst + face_row, 32);
                noc_async_read<32>(right, dst + face_row + 512, 32);
            }
            noc_async_read_barrier();
            cb_push_back(0, 1);
        }
    }
#endif
}
