// Feed compact activation tiles and write back the two live BF16 faces.
#include "api/dataflow/dataflow_api.h"
#ifdef EXPERTS_VERIFY_DOWN_ROWS
#include "ckernel.h"
#include "ckernel_defs.h"
#endif

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr auto ends_args = TensorAccessorArgs<3>();
    constexpr auto x_args = TensorAccessorArgs<ends_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto ends = TensorAccessor(ends_args, get_common_arg_val<uint32_t>(0), 512);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), 512);
#ifndef EXPERTS_VERIFY_DOWN_ROWS
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2), 512);
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
    const uint32_t ends_l1 = get_write_ptr(3);
    noc_async_read<512>(ends.get_noc_addr(0), ends_l1, 512);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ends_l1);
#ifndef EXPERTS_VERIFY_DOWN_ROWS
    auto write = [&](uint32_t block) {
        cb_wait_front(16, out_cols);
        const uint32_t src = get_read_ptr(16);
        for (uint32_t i = 0; i < out_cols; ++i) {
            const uint64_t dst = out.get_noc_addr(block * out_k + col + i);
            noc_async_write<256>(src + i * 2048, dst, 256);
            noc_async_write<256>(src + i * 2048 + 512, dst + 256, 256);
        }
        noc_async_write_barrier();
        cb_pop_front(16, out_cols);
    };
    uint32_t pending = 0;
    bool have_pending = false;
#endif
    for (uint32_t expert = lane; expert < 128; expert += lanes) {
        const uint32_t begin = expert == 0 ? 0 : offsets[expert - 1];
        const uint32_t end = offsets[expert];
        for (uint32_t block = begin; block < end; ++block) {
            cb_reserve_back(0, k);
            const uint32_t dest = get_write_ptr(0);
            for (uint32_t i = 0; i < k; ++i) {
                noc_async_read<512>(x.get_noc_addr(block * k + i), dest + i * 512, 512);
            }
            noc_async_read_barrier();
            cb_push_back(0, k);
#ifndef EXPERTS_VERIFY_DOWN_ROWS
            if (have_pending) {
                write(pending);
            }
            pending = block;
            have_pending = true;
#endif
        }
    }
#ifdef EXPERTS_VERIFY_DOWN_ROWS
    constexpr auto inverse_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<inverse_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto inverse = TensorAccessor(inverse_args, get_common_arg_val<uint32_t>(3), 1024);
    const uint32_t map_addr = get_write_ptr(5);
    const uint32_t weights_addr = get_write_ptr(6);
    noc_async_read<1024>(inverse.get_noc_addr(0), map_addr, 1024);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto map = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(map_addr);
    cb_wait_front(16, offsets[127]);
    const uint32_t projected = get_read_ptr(16);
    for (uint32_t row = 0; row < EXPERTS_VERIFY_DOWN_ROWS; ++row) {
        const auto gamma = TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(4 + row), 4096);
        noc_async_read<32>(gamma.get_noc_addr(0), weights_addr, 32);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const auto weights = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_addr);
        for (uint32_t slot = 0; slot < 8; ++slot) {
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, weights[slot]);
        }
        for (uint32_t slot = 0; slot < 8; ++slot) {
            const uint32_t entry = map[row * 8 + slot] - 1;
            const uint32_t from = projected + (entry / 8) * 2048 + (entry % 8) * 32;
            cb_reserve_back(4, 1);
            const uint32_t dst = get_write_ptr(4);
            noc_async_read<32>(get_noc_addr(from), dst, 32);
            noc_async_read<32>(get_noc_addr(from + 512), dst + 512, 32);
            noc_async_read_barrier();
            cb_push_back(4, 1);
        }
        cb_wait_front(17, 1);
        const uint32_t source = get_read_ptr(17);
        noc_async_write<32>(source, out.get_noc_addr(col, row * 32), 32);
        noc_async_write<32>(source + 512, out.get_noc_addr(col, 512 + row * 32), 32);
        noc_async_write_barrier();
        cb_pop_front(17, 1);
    }
    cb_pop_front(16, offsets[127]);
#else
    if (have_pending) {
        write(pending);
    }
#endif
}
