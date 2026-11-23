#include "api/dataflow/dataflow_api.h"

void constant_tile(uint32_t cb, uint32_t value) {
    cb_reserve_back(cb, 1);
    auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    for (uint32_t i = 0; i < 1024; ++i) {
        dst[i] = value;
    }
    cb_push_back(cb, 1);
}

void kernel_main() {
    constexpr uint32_t cols = get_compile_time_arg_val(0);
    constexpr uint32_t out_bytes = get_compile_time_arg_val(2) ? 2048 : 4096;
    constexpr uint32_t in_bytes = get_compile_time_arg_val(3) ? 2048 : 4096;
    constexpr uint32_t eps = get_compile_time_arg_val(4);
    constexpr uint32_t stride = get_compile_time_arg_val(6);
    constexpr uint32_t offset = get_compile_time_arg_val(7);
    constexpr uint32_t block = 4;
    constexpr uint32_t rows = get_compile_time_arg_val(8);
    constexpr bool paired = get_compile_time_arg_val(9);
    constexpr auto args = TensorAccessorArgs<10>();
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef PREFILL_WRITE
    const auto out = TensorAccessor(args, get_common_arg_val<uint32_t>(0), out_bytes);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t row = paired ? task % rows : task;
        const uint32_t col_offset = offset + (paired ? task / rows * cols : 0);
        for (uint32_t col = 0; col < cols; col += block) {
            cb_wait_front(16, block);
            const uint32_t data = get_read_ptr(16);
            for (uint32_t c = 0; c < block; ++c) {
                noc_async_write<out_bytes>(data + c * out_bytes, out.get_noc_addr(row * stride + col_offset + col + c), out_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(16, block);
        }
    }
#else
    constant_tile(1, 0x3f800000);
    constant_tile(2, eps);
    for (uint32_t task = start; task < end; ++task) {
        const uint32_t row = paired ? task % rows : task;
        const auto in = TensorAccessor(args, get_common_arg_val<uint32_t>(paired ? task / rows : 0), in_bytes);
        for (uint32_t col = 0; col < cols; col += block) {
            cb_reserve_back(0, block);
            const uint32_t data = get_write_ptr(0);
            for (uint32_t c = 0; c < block; ++c) {
                noc_async_read<in_bytes>(in.get_noc_addr(row * cols + col + c), data + c * in_bytes, in_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(0, block);
        }
#ifdef PREFILL_WEIGHT
        if (task == start) {
            constexpr auto weight_args = TensorAccessorArgs<args.next_compile_time_args_offset()>();
            const auto weight = TensorAccessor(weight_args, get_common_arg_val<uint32_t>(1), 4096);
            cb_reserve_back(6, cols);
            const uint32_t data = get_write_ptr(6);
            for (uint32_t col = 0; col < cols; ++col) {
                noc_async_read<4096>(weight.get_noc_addr(col), data + col * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(6, cols);
        }
#endif
    }
#endif
}
