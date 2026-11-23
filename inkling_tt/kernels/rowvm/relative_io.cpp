#include "../common/read_vc.h"

void kernel_main() {
    select_read_vc();
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
#ifdef ROWVM_WRITE
    constexpr uint32_t out_cb = 16;
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    const auto out = TensorAccessor(TensorAccessorArgs<1>(), get_common_arg_val<uint32_t>(0), tile_bytes);
    for (uint32_t tile = first; tile < end; ++tile) {
        cb_wait_front(out_cb, 1);
        noc_async_write<tile_bytes>(get_read_ptr(out_cb), out.get_noc_addr(tile), tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(out_cb, 1);
    }
#else
    constexpr uint32_t q_cb = 0;
    constexpr uint32_t rows_cb = 1;
    constexpr uint32_t mask_cb = 2;
    constexpr uint32_t index_cb = 4;
    constexpr uint32_t bf_tile_bytes = 2048;
    constexpr uint32_t fp_tile_bytes = 4096;
    constexpr uint32_t stick_bytes = 64;
    constexpr uint32_t index_bytes = 128;
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto index_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
#ifndef ROWVM_POSITION
    constexpr auto mask_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(3), fp_tile_bytes);
#endif
    const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0), bf_tile_bytes);
    const auto indices = TensorAccessor(index_args, get_common_arg_val<uint32_t>(1), get_compile_time_arg_val(0));
    const auto table = TensorAccessor(table_args, get_common_arg_val<uint32_t>(2), stick_bytes);
    const uint32_t index_addr = get_write_ptr(index_cb);

    cb_reserve_back(q_cb, 1);
    noc_async_read<bf_tile_bytes>(q.get_noc_addr(0), get_write_ptr(q_cb), bf_tile_bytes);
#ifdef ROWVM_POSITION
    noc_async_read<4>(indices.get_noc_addr(0), index_addr, 4);
#endif
    noc_async_read_barrier();
    cb_push_back(q_cb, 1);
#ifdef ROWVM_POSITION
    const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_addr);
#endif
    for (uint32_t tile = first; tile < end; ++tile) {
        cb_reserve_back(rows_cb, 1);
        cb_reserve_back(mask_cb, 1);
#ifdef ROWVM_POSITION
        auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_addr);
        auto mask_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(mask_cb));
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t slot = tile * 32 + row;
            uint32_t distance = pos - slot;
            bool live;
            if constexpr (ROWVM_GLOBAL) {
                live = slot <= pos;
                ids[row] = live && distance < ROWVM_POSITION ? distance : ROWVM_POSITION;
            } else {
                distance &= ROWVM_POSITION - 1;
                live = distance <= pos;
                ids[row] = distance;
            }
            mask_row[(row / 16) * 256 + row % 16] = live ? 0u : ROWVM_MASK_BITS;
        }
#else
        noc_async_read<index_bytes>(indices.get_noc_addr(0, tile * index_bytes), index_addr, index_bytes);
        noc_async_read<fp_tile_bytes>(mask.get_noc_addr(tile), get_write_ptr(mask_cb), fp_tile_bytes);
        noc_async_read_barrier();
        const auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_addr);
#endif
        const uint32_t rows = get_write_ptr(rows_cb);
        const uint32_t first_id = ids[0];
        bool repeated = first_id == ids[31];
        if (repeated) {
            for (uint32_t row = 1; row < 31; ++row) {
                repeated &= ids[row] == first_id;
            }
        }
        if (repeated) {
            // Global windows can contain many tiles of the same sentinel index.
            // Read the stick once per core and replicate it in local L1.
            noc_async_read<stick_bytes>(table.get_noc_addr(first_id), rows, stick_bytes);
            noc_async_read_barrier();
            const uint64_t local_stick = get_noc_addr(rows);
            for (uint32_t row = 1; row < 32; ++row) {
                noc_async_read<stick_bytes>(local_stick, rows + row * stick_bytes, stick_bytes);
            }
        } else {
            for (uint32_t row = 0; row < 32; ++row) {
                noc_async_read<stick_bytes>(table.get_noc_addr(ids[row]), rows + row * stick_bytes, stick_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(rows_cb, 1);
        cb_push_back(mask_cb, 1);
    }
#endif
}
