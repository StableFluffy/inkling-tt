// BRISC gathers and scatters only the first two face rows of B1 tiles.
#include "../common/read_vc.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    select_read_vc();
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_gamma = 3;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face1 = 512;
    constexpr uint32_t half_row = 32;
    constexpr uint32_t x_page = 64 * EXPERTS_ACT_ROWS;
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), tile_bytes);
#if EXPERTS_GATE
    const uint32_t group = get_arg_val<uint32_t>(0);
    const uint32_t lane = get_arg_val<uint32_t>(3);
    constexpr uint32_t slots = 1;
#else
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t col = get_arg_val<uint32_t>(1);
    constexpr uint32_t slots = 8;
    constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto gamma = TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(2), 4096);
    const uint32_t gamma_l1 = get_write_ptr(cb_gamma);
    noc_async_read<32>(gamma.get_noc_addr(0), gamma_l1, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto scores = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gamma_l1);
    for (uint32_t slot = 0; slot < slots; ++slot) {
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, scores[slot]);
    }
#endif
    for (uint32_t slot_in = 0; slot_in < slots; ++slot_in) {
        cb_reserve_back(cb_x, k);
        uint32_t dst = get_write_ptr(cb_x);
        for (uint32_t tile = 0; tile < k; ++tile) {
            const uint64_t src = x.get_noc_addr(slot_in * k + tile);
            noc_async_read<half_row>(src, dst, half_row);
            noc_async_read<half_row>(src + face1, dst + x_page / 2, half_row);
            dst += x_page;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, k);
    }
#if EXPERTS_GATE
    constexpr uint32_t count = get_compile_time_arg_val(2) / 2;
    constexpr uint32_t out_slots = 8 / EXPERTS_GU_LANES;
#else
    constexpr uint32_t count = get_compile_time_arg_val(2);
    constexpr uint32_t out_slots = 1;
#endif
    for (uint32_t slot = 0; slot < out_slots; ++slot) {
#if EXPERTS_GATE
        const uint32_t first = (lane + slot * EXPERTS_GU_LANES) * 36 + group;
#else
        const uint32_t first = bank + 8 * col * count;
#endif
        constexpr uint32_t stride = EXPERTS_GATE ? 1 : 8;
        cb_wait_front(cb_out, count);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t tile = 0; tile < count; ++tile) {
            const uint64_t dst = out.get_noc_addr(first + stride * tile);
            noc_async_write<half_row>(src, dst, half_row);
            noc_async_write<half_row>(src + face1, dst + face1, half_row);
            src += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, count);
    }
}
