#include "broadcast.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr auto ends_args = TensorAccessorArgs<3>();
    constexpr auto x_args = TensorAccessorArgs<ends_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto ends = TensorAccessor(ends_args, get_common_arg_val<uint32_t>(0), 512);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2), 2048);
#if EXPERTS_WIDE_L1_BLOCKS
    constexpr auto tail_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto tail = TensorAccessor(tail_args, get_common_arg_val<uint32_t>(3), 2048);
#endif
#if EXPERTS_GATE
    const uint32_t col = get_arg_val<uint32_t>(0);
    const uint32_t lane = get_arg_val<uint32_t>(3);
    constexpr uint32_t lanes = EXPERTS_GU_LANES;
    constexpr uint32_t out_k = 36;
    constexpr uint32_t out_cols = cols / 2;
#else
    const uint32_t col = get_arg_val<uint32_t>(0) + 8 * get_arg_val<uint32_t>(1);
    constexpr uint32_t lane = 0;
    constexpr uint32_t lanes = 1;
    constexpr uint32_t out_k = 72;
    constexpr uint32_t out_cols = 1;
#endif
#if EXPERTS_BROADCAST
    PrefillBroadcast broadcast(3);
#endif
    const uint32_t ends_l1 = get_write_ptr(3);
    noc_async_read<512>(ends.get_noc_addr(0), ends_l1, 512);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ends_l1);
    auto write = [&](uint32_t block) {
        cb_wait_front(16, out_cols);
        const uint32_t src = get_read_ptr(16);
        for (uint32_t i = 0; i < out_cols; ++i) {
#if EXPERTS_WIDE_L1_BLOCKS && !EXPERTS_GATE
            const uint64_t dst = block < EXPERTS_WIDE_L1_BLOCKS
                ? out.get_noc_addr(block * out_k + col + i)
                : tail.get_noc_addr((block - EXPERTS_WIDE_L1_BLOCKS) * out_k + col + i);
#else
            const uint64_t dst = out.get_noc_addr(block * out_k + col + i);
#endif
            noc_async_write<2048>(src + i * 2048, dst, 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(16, out_cols);
    };
    uint32_t pending = 0;
    bool have_pending = false;
    for (uint32_t expert = lane; expert < 128; expert += lanes) {
        const uint32_t begin = expert == 0 ? 0 : offsets[expert - 1];
        const uint32_t end = offsets[expert];
        for (uint32_t block = begin; block < end; ++block) {
            cb_reserve_back(0, k);
            const uint32_t dest = get_write_ptr(0);
#if EXPERTS_BROADCAST
            if (broadcast.sender) {
#endif
                for (uint32_t i = 0; i < k; ++i) {
#if EXPERTS_WIDE_L1_BLOCKS && EXPERTS_GATE
                    const uint64_t src = block < EXPERTS_WIDE_L1_BLOCKS
                        ? x.get_noc_addr(block * k + i)
                        : tail.get_noc_addr((block - EXPERTS_WIDE_L1_BLOCKS) * k + i);
#else
                    const uint64_t src = x.get_noc_addr(block * k + i);
#endif
                    noc_async_read<2048>(src, dest + i * 2048, 2048);
                }
                noc_async_read_barrier();
#if EXPERTS_BROADCAST
            }
            broadcast.distribute(dest, k * 2048);
#endif
            cb_push_back(0, k);
            if (have_pending) {
                write(pending);
            }
            pending = block;
            have_pending = true;
        }
    }
    if (have_pending) {
        write(pending);
    }
#if EXPERTS_BROADCAST
    broadcast.finish();
#endif
}
