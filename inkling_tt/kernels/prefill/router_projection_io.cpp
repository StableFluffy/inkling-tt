#include "api/dataflow/dataflow_api.h"

constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t src_cols = get_compile_time_arg_val(1);
constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
constexpr uint32_t bf_bytes = 2048;
constexpr uint32_t fp_bytes = 4096;

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    constexpr auto args = TensorAccessorArgs<4>();
#ifdef PREFILL_WRITE
    const auto out = TensorAccessor(args, get_common_arg_val<uint32_t>(0), fp_bytes);
    for (uint32_t task = first; task < end; ++task) {
        cb_wait_front(16, n_tiles);
        const uint32_t data = get_read_ptr(16);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc_async_write<fp_bytes>(data + n * fp_bytes, out.get_noc_addr(task * n_tiles + n), fp_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(16, n_tiles);
    }
#else
    constexpr auto w_args = TensorAccessorArgs<args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(args, get_common_arg_val<uint32_t>(0), bf_bytes);
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), bf_bytes);
    for (uint32_t task = first; task < end; ++task) {
        const uint32_t split = task / rows;
        const uint32_t row = task % rows;
        if (task == first || row == 0) {
            cb_reserve_back(1, k_tiles * n_tiles);
            const uint32_t data = get_write_ptr(1);
            for (uint32_t t = 0; t < k_tiles * n_tiles; ++t) {
                noc_async_read<bf_bytes>(w.get_noc_addr(split * k_tiles * n_tiles + t), data + t * bf_bytes, bf_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(1, k_tiles * n_tiles);
        }
        cb_reserve_back(0, k_tiles);
        const uint32_t data = get_write_ptr(0);
        for (uint32_t k = 0; k < k_tiles; ++k) {
            noc_async_read<bf_bytes>(x.get_noc_addr(row * src_cols + split * k_tiles + k), data + k * bf_bytes, bf_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(0, k_tiles);
    }
#endif
}
