// Grouped batch-one BF16 projections: each group splits K over stripes, gathers FP32 row
// partials on its leader and writes the BF16-rounded row into a TILE [1,1,1,N] output.
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
constexpr uint32_t stripes = get_compile_time_arg_val(1);
constexpr uint32_t groups = get_compile_time_arg_val(2);
constexpr uint32_t grid_x = get_compile_time_arg_val(3);
constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
constexpr uint32_t x_cb = 0, w_cb = 1, aggregate_cb = 3, partial_cb = 16, row_cb = 17;
constexpr uint32_t row_bytes = n_tiles * 128;
constexpr uint32_t aggregate_pages = (stripes * row_bytes + 4095) / 4096;
constexpr auto x_args = TensorAccessorArgs<5>();
constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto h_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

uint64_t leader_addr(uint32_t group, uint32_t address) {
    const uint32_t packed = get_common_arg_val<uint32_t>(3 * groups + group);
    return get_noc_addr(packed & 0xffff, packed >> 16, address);
}
}  // namespace

void kernel_main() {
    const uint32_t task = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t group = task / stripes;
    const uint32_t stripe = task % stripes;
#ifndef MATVEC_WRITER
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(3 * group), 2048);
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(3 * group + 1), 2048);
    cb_reserve_back(w_cb, k_tiles * n_tiles);
    cb_reserve_back(x_cb, k_tiles);
    const uint32_t wdst = get_write_ptr(w_cb);
    const uint32_t xdst = get_write_ptr(x_cb);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc_async_read_tile((stripe * k_tiles + k) * n_tiles + n, w, wdst + (k * n_tiles + n) * 2048);
        }
    }
    for (uint32_t k = 0; k < k_tiles; ++k) {
        noc_async_read_tile(stripe * k_tiles + k, x, xdst + k * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(w_cb, k_tiles * n_tiles);
    cb_push_back(x_cb, k_tiles);
    if (stripe == 0) {
        cb_reserve_back(aggregate_cb, aggregate_pages);
        auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait(sem, stripes);
        noc_semaphore_set(sem, 0);
        cb_push_back(aggregate_cb, aggregate_pages);
    }
#else
    // Row 0 of each FP32 output tile: face 0 row 0, then face 1 row 0.
    cb_wait_front(partial_cb, n_tiles);
    const uint32_t partial = get_read_ptr(partial_cb);
    const uint32_t aggregate = get_write_ptr(aggregate_cb) + stripe * row_bytes;
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write<64>(partial + n * 4096 + f * 1024, leader_addr(group, aggregate + n * 128 + f * 64), 64);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(leader_addr(group, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(partial_cb, n_tiles);
    if (stripe != 0) {
        return;
    }
    // The leader's BF16 row is dense: 32 values per output tile, split over two face rows.
    cb_wait_front(row_cb, 1);
    const uint32_t row = get_read_ptr(row_cb);
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(3 * group + 2), 2048);
    for (uint32_t n = 0; n < n_tiles; ++n) {
        noc_async_write<32>(row + n * 64, h.get_noc_addr(n, 0), 32);
        noc_async_write<32>(row + n * 64 + 32, h.get_noc_addr(n, 512), 32);
    }
    noc_async_write_barrier();
    cb_pop_front(row_cb, 1);
#endif
}
