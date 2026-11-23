// Fused decode router: K-split BF16 projection partials, gathered on one core for
// sigmoid + correction bias, top-8 selection (lowest ID on ties) and unbiased gamma.
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
constexpr uint32_t stripes = get_compile_time_arg_val(1);
constexpr uint32_t grid_x = get_compile_time_arg_val(2);
constexpr uint32_t n_tiles = 4;
constexpr uint32_t x_cb = 0, w_cb = 1, bias_cb = 2, aggregate_cb = 3, selection_cb = 8;
constexpr uint32_t partial_cb = 16, gamma_cb = 17, scores_cb = 18;
constexpr uint32_t aggregate_pages = (stripes * 512 + 4095) / 4096;
// Common args: x, weight, bias, ids, gamma addresses, then the leader's packed NoC coordinates.
constexpr auto x_args = TensorAccessorArgs<3>();
constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto bias_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto ids_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
constexpr auto gamma_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();

uint64_t leader_addr(uint32_t address) {
    const uint32_t packed = get_common_arg_val<uint32_t>(5);
    return get_noc_addr(packed & 0xffff, packed >> 16, address);
}
}  // namespace

void kernel_main() {
    const uint32_t task = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
#ifndef ROUTER_WRITER
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    cb_reserve_back(w_cb, k_tiles * n_tiles);
    cb_reserve_back(x_cb, k_tiles);
    const uint32_t wdst = get_write_ptr(w_cb);
    const uint32_t xdst = get_write_ptr(x_cb);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc_async_read_tile((task * k_tiles + k) * n_tiles + n, w, wdst + (k * n_tiles + n) * 2048);
        }
    }
    for (uint32_t k = 0; k < k_tiles; ++k) {
        noc_async_read_tile(task * k_tiles + k, x, xdst + k * 2048);
    }
    if (task == 0) {
        const auto bias = TensorAccessor(bias_args, get_common_arg_val<uint32_t>(2), 512);
        cb_reserve_back(bias_cb, 1);
        noc_async_read(bias.get_noc_addr(0), get_write_ptr(bias_cb), 512);
    }
    noc_async_read_barrier();
    cb_push_back(w_cb, k_tiles * n_tiles);
    cb_push_back(x_cb, k_tiles);
    if (task == 0) {
        cb_push_back(bias_cb, 1);
        cb_reserve_back(aggregate_cb, aggregate_pages);
        auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait(sem, stripes);
        noc_semaphore_set(sem, 0);
        cb_push_back(aggregate_cb, aggregate_pages);
    }
#else
    // Row 0 of each FP32 output tile is face 0 row 0 and face 1 row 0.
    cb_wait_front(partial_cb, n_tiles);
    const uint32_t partial = get_read_ptr(partial_cb);
    const uint32_t aggregate = get_write_ptr(aggregate_cb) + task * 512;
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write<64>(partial + n * 4096 + f * 1024, leader_addr(aggregate + n * 128 + f * 64), 64);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(leader_addr(get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(partial_cb, n_tiles);
    if (task != 0) {
        return;
    }
    constexpr uint32_t experts = 128;
    constexpr uint32_t k = 8;
    cb_wait_front(scores_cb, 2);
    const auto scores = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(scores_cb));
    const auto biased = scores + 1024;
    uint32_t best[k] = {};
    uint16_t ids[k] = {};
    for (uint32_t i = 0; i < experts; ++i) {
        uint32_t bits = biased[i];
        if (bits == 0x80000000u) {
            bits = 0;
        }
        const uint32_t key = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
        if (key <= best[k - 1]) {
            continue;
        }
        uint32_t pos = k - 1;
        while (pos > 0 && key > best[pos - 1]) {
            best[pos] = best[pos - 1];
            ids[pos] = ids[pos - 1];
            --pos;
        }
        best[pos] = key;
        ids[pos] = i;
    }
    cb_reserve_back(selection_cb, 1);
    const uint32_t selected = get_write_ptr(selection_cb);
    auto values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(selected);
    auto id_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(selected + 256);
    for (uint32_t i = 0; i < 64; ++i) {
        values[i] = i < k ? scores[ids[i]] : 0;
    }
    for (uint32_t i = 0; i < k; ++i) {
        id_stick[i] = ids[i];
    }
    const auto index = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(3), 16);
    noc_async_write<16>(selected + 256, index.get_noc_addr(0), 16);
    cb_pop_front(scores_cb, 2);
    cb_push_back(selection_cb, 1);
    cb_wait_front(gamma_cb, 1);
    const auto gamma = TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(4), 4096);
    noc_async_write<64>(get_read_ptr(gamma_cb), gamma.get_noc_addr(0), 64);
    noc_async_write_barrier();
    cb_pop_front(gamma_cb, 1);
#endif
}
