#include "api/dataflow/dataflow_api.h"

constexpr uint32_t fp_bytes = 4096;
constexpr uint32_t cols = get_compile_time_arg_val(0);
constexpr uint32_t rows = get_compile_time_arg_val(1);
constexpr uint32_t depth = get_compile_time_arg_val(2);
constexpr uint32_t queries = get_compile_time_arg_val(3);

uint32_t face_index(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}

void kernel_main() {
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr auto a0 = TensorAccessorArgs<4>();
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef PREFILL_WRITE
    for (uint32_t tile = first; tile < end; ++tile) {
        cb_wait_front(16, queries);
        const uint32_t data = get_read_ptr(16);
        for (uint32_t q = 0; q < queries; ++q) {
            const auto out = TensorAccessor(a0, addr(q), fp_bytes);
            noc_async_write<fp_bytes>(data + q * fp_bytes, out.get_noc_addr(tile), fp_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(16, queries);
    }
#else
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const auto slabs = TensorAccessor(a0, addr(0), fp_bytes);
    const auto weights = TensorAccessor(a1, addr(1), fp_bytes);
    const uint32_t scratch = get_write_ptr(2);
    for (uint32_t tile = first; tile < end; ++tile) {
        const uint32_t row_block = tile / cols;
        if (tile == first || tile % cols == 0) {
            cb_reserve_back(1, queries * depth);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
            for (uint32_t d = 0; d < depth; ++d) {
                noc_async_read<fp_bytes>(weights.get_noc_addr(d * rows + row_block),
                                        scratch + d * fp_bytes, fp_bytes);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            auto p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            for (uint32_t q = 0; q < queries; ++q) {
                for (uint32_t d = 0; d < depth; ++d) {
                    for (uint32_t row = 0; row < 32; ++row) {
                        dst[(q * depth + d) * 1024 + face_index(row, 0)] = p[d * 1024 + face_index(row, q)];
                    }
                }
            }
            cb_push_back(1, queries * depth);
        }
        cb_reserve_back(0, depth);
        const uint32_t data = get_write_ptr(0);
        for (uint32_t d = 0; d < depth; ++d) {
            noc_async_read<fp_bytes>(slabs.get_noc_addr(d * rows * cols + tile), data + d * fp_bytes, fp_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(0, depth);
    }
#endif
}
