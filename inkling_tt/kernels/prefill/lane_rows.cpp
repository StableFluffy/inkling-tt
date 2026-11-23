// Lane members take turns reading an activation block and multicasting it to
// the lane. Every member writes its own output columns for each block.
#include "api/dataflow/dataflow_api.h"
#include "lane.h"

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr auto ends_args = TensorAccessorArgs<3>();
    constexpr auto x_args = TensorAccessorArgs<ends_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto ends_acc = TensorAccessor(ends_args, get_common_arg_val<uint32_t>(0), 512);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2), 2048);
#if EXPERTS_WIDE_L1_BLOCKS
    constexpr auto tail_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto tail = TensorAccessor(tail_args, get_common_arg_val<uint32_t>(3), 2048);
#endif
#if EXPERTS_GATE
    constexpr uint32_t out_k = 36;
    constexpr uint32_t out_cols = cols / 2;
#else
    constexpr uint32_t out_k = 72;
    constexpr uint32_t out_cols = cols;
#endif
    const uint32_t col = get_arg_val<uint32_t>(lane_args::group) * out_cols;
    const uint32_t me = get_arg_val<uint32_t>(lane_args::me);
    const uint32_t members = get_arg_val<uint32_t>(lane_args::members);
    const uint32_t rects = get_arg_val<uint32_t>(lane_args::rects);
    const uint32_t rect_arg = lane_args::rects + 1;
    const uint32_t coord_arg = rect_arg + rects * 5;
    const uint32_t ready_addr = get_semaphore(0);
    const uint32_t valid_addr = get_semaphore(1);
    auto ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    auto valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);

    const uint32_t ends_l1 = get_write_ptr(3);
    noc_async_read<512>(ends_acc.get_noc_addr(0), ends_l1, 512);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ends_l1);
    const auto ends = [&](uint32_t e) { return offsets[e]; };
    const LaneRange range = lane_range(ends, get_arg_val<uint32_t>(lane_args::lane),
                                       get_arg_val<uint32_t>(lane_args::lanes));
    const uint32_t my_x = get_arg_val<uint32_t>(coord_arg + 2 * me);
    const uint32_t my_y = get_arg_val<uint32_t>(coord_arg + 2 * me + 1);

    auto multicast = [&](uint32_t src, uint32_t addr, uint32_t bytes, bool semaphore) {
        for (uint32_t r = 0; r < rects; ++r) {
            const uint32_t base = rect_arg + r * 5;
            const uint32_t x0 = get_arg_val<uint32_t>(base), y0 = get_arg_val<uint32_t>(base + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(base + 2), y1 = get_arg_val<uint32_t>(base + 3);
            const uint32_t inside = x0 <= my_x && my_x <= x1 && y0 <= my_y && my_y <= y1;
            const uint32_t n = get_arg_val<uint32_t>(base + 4) - inside;
            if (n == 0) {
                continue;
            }
            const uint64_t dst = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, addr)
                                                : get_noc_multicast_addr(x1, y1, x0, y0, addr);
            if (semaphore) {
                noc_semaphore_set_multicast(src, dst, n);
            } else {
                noc_async_write_multicast(src, dst, bytes, n);
            }
        }
    };
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
    for (uint32_t expert = 0; expert < 128; ++expert) {
        uint32_t begin, end;
        if (!lane_expert(ends, range, expert, begin, end)) {
            continue;
        }
        for (uint32_t block = begin; block < end; ++block) {
            cb_reserve_back(0, k);
            const uint32_t dest = get_write_ptr(0);
            const uint32_t sender = (block - range.lo) % members;
            if (sender == me) {
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
                if (members > 1) {
                    noc_semaphore_wait(ready, members - 1);
                    noc_semaphore_set(ready, 0);
                    multicast(dest, dest, k * 2048, false);
                    noc_async_write_barrier();
                    multicast(get_semaphore(2), valid_addr, 0, true);
                }
            } else {
                noc_semaphore_set(valid, 0);
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(coord_arg + 2 * sender),
                                               get_arg_val<uint32_t>(coord_arg + 2 * sender + 1), ready_addr), 1);
                noc_semaphore_wait(valid, 1);
            }
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
    noc_async_write_barrier();
}
