// Move head tiles from a token-major projection to head-major layout.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t src_cols = get_compile_time_arg_val(0);
    constexpr uint32_t col_offset = get_compile_time_arg_val(1);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(2);
    constexpr bool scaled = get_compile_time_arg_val(3);
    constexpr uint32_t head_width = get_compile_time_arg_val(4);
    constexpr uint32_t head_tiles = (head_width + 31) / 32;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef PREFILL_WRITE
    const auto out = TensorAccessor(TensorAccessorArgs<5>(), get_common_arg_val<uint32_t>(0), 2048);
    constexpr uint32_t cb = 16;
    for (uint32_t task = start; task < end; ++task) {
        cb_wait_front(cb, head_tiles);
        const uint32_t src = get_read_ptr(cb);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            noc_async_write<2048>(src + d * 2048, out.get_noc_addr(task * head_tiles + d), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(cb, head_tiles);
    }
#else
    constexpr auto x_args = TensorAccessorArgs<5>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    constexpr auto tau_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto tau = TensorAccessor(tau_args, get_common_arg_val<uint32_t>(1), 4096);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t head = task / row_tiles;
        const uint32_t row = task % row_tiles;
        cb_reserve_back(0, head_tiles);
        const uint32_t dest = get_write_ptr(0);
        if constexpr (head_width == 16) {
            constexpr uint32_t face_bytes = 512;
            const uint64_t src = x.get_noc_addr(row * src_cols + col_offset + head / 2) + (head % 2) * face_bytes;
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t f = 0; f < 4; f += 2) {
                noc_async_read<face_bytes>(src + f * face_bytes, dest + f * face_bytes, face_bytes);
                noc_async_read<face_bytes>(zeros, dest + (f + 1) * face_bytes, face_bytes);
            }
        } else {
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_read<2048>(x.get_noc_addr(row * src_cols + col_offset + head * head_tiles + d),
                                     dest + d * 2048, 2048);
            }
        }
        if constexpr (scaled) {
            cb_reserve_back(1, 1);
            noc_async_read<4096>(tau.get_noc_addr(row), get_write_ptr(1), 4096);
        }
        noc_async_read_barrier();
        cb_push_back(0, head_tiles);
        if constexpr (scaled) {
            cb_push_back(1, 1);
        }
    }
#endif
}
