// Message-gate controller input and down projection in one launch. Per gate, 18 stripes own 256
// columns of [u | m]: RMS over each half via its leader, BF16 rounding, a K-stripe matmul, and an
// FP32 row reduction on the gate leader that writes the BF16 [1,1,1,64] result.
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t half_stripes = get_compile_time_arg_val(0);
constexpr uint32_t groups = get_compile_time_arg_val(1);
constexpr uint32_t grid_x = get_compile_time_arg_val(2);
constexpr uint32_t stripes = 2 * half_stripes;
constexpr uint32_t k_tiles = 8, n_tiles = 2;
constexpr uint32_t x_cb = 0, w_cb = 1, agg_h_cb = 3, row_cb = 4, agg_cb = 8, bcast_cb = 9;
constexpr uint32_t p2_cb = 17, partial_cb = 18, coeff_cb = 19, xdense_cb = 20, hrow_cb = 21;
constexpr auto u_args = TensorAccessorArgs<3>();
constexpr auto m_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();

template <class Args>
void read_stripe(const Args& args, uint32_t address, uint32_t chunk, uint32_t dst) {
    const auto row = TensorAccessor(args, address, 4096);
    for (uint32_t f = 0; f < 16; ++f) {
        noc_async_read<64>(row.get_noc_addr(chunk * 8 + f / 2, (f % 2) * 1024), dst + f * 64, 64);
    }
}
constexpr auto h_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

uint64_t peer(uint32_t task, uint32_t address) {
    const uint32_t packed = get_common_arg_val<uint32_t>(4 * groups + task);
    return get_noc_addr(packed & 0xffff, packed >> 16, address);
}
}  // namespace

void kernel_main() {
    const uint32_t task = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t group = task / stripes;
    const uint32_t stripe = task % stripes;
    const uint32_t half = stripe / half_stripes;
    const uint32_t chunk = stripe % half_stripes;
    const uint32_t half_leader = group * stripes + half * half_stripes;
    const uint32_t gate_leader = group * stripes;
#ifndef MSGDOWN_WRITER
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(4 * group + 2), 2048);
    cb_reserve_back(w_cb, k_tiles * n_tiles);
    const uint32_t wdst = get_write_ptr(w_cb);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc_async_read_tile((stripe * k_tiles + k) * n_tiles + n, w, wdst + (k * n_tiles + n) * 2048);
        }
    }
    // The 256 FP32 values of this stripe: face row 0 of faces 0 and 1 in eight tiles.
    cb_reserve_back(row_cb, 1);
    const uint32_t rdst = get_write_ptr(row_cb);
    if (half == 0) {
        read_stripe(u_args, get_common_arg_val<uint32_t>(4 * group), chunk, rdst);
    } else {
        read_stripe(m_args, get_common_arg_val<uint32_t>(4 * group + 1), chunk, rdst);
    }
    noc_async_read_barrier();
    cb_push_back(w_cb, k_tiles * n_tiles);
    cb_push_back(row_cb, 1);
    if (chunk == 0) {
        cb_reserve_back(agg_cb, 1);
        auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait(sem, half_stripes);
        noc_semaphore_set(sem, 0);
        cb_push_back(agg_cb, 1);
    }
    cb_reserve_back(bcast_cb, 1);
    auto received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait(received, 1);
    noc_semaphore_set(received, 0);
    cb_push_back(bcast_cb, 1);
    if (stripe == 0) {
        cb_reserve_back(agg_h_cb, 2);
        auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_wait(sem, stripes);
        noc_semaphore_set(sem, 0);
        cb_push_back(agg_h_cb, 2);
    }
#else
    // Sum of squares (element 0 of the partial) into the half leader's 16-byte slot.
    cb_wait_front(partial_cb, 1);
    noc_async_write<16>(get_read_ptr(partial_cb), peer(half_leader, get_write_ptr(agg_cb) + chunk * 16), 16);
    noc_async_write_barrier();
    noc_semaphore_inc(peer(half_leader, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(partial_cb, 1);
    if (chunk == 0) {
        cb_wait_front(coeff_cb, 1);
        const uint32_t coeff = get_read_ptr(coeff_cb);
        for (uint32_t p = half_leader; p < half_leader + half_stripes; ++p) {
            noc_async_write<16>(coeff, peer(p, get_write_ptr(bcast_cb)), 16);
        }
        noc_async_write_barrier();
        for (uint32_t p = half_leader; p < half_leader + half_stripes; ++p) {
            noc_semaphore_inc(peer(p, get_semaphore(1)), 1);
        }
        noc_async_atomic_barrier();
        cb_pop_front(coeff_cb, 1);
    }
    // Dense BF16 stripe -> eight [8,32] activation tiles (row 0 in face rows 0 of faces 0 and 1).
    cb_wait_front(xdense_cb, 1);
    cb_reserve_back(x_cb, k_tiles);
    auto dense = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(xdense_cb));
    auto tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(x_cb));
    for (uint32_t k = 0; k < k_tiles; ++k) {
#pragma GCC unroll 8
        for (uint32_t i = 0; i < 8; ++i) {
            tiles[k * 128 + i] = dense[k * 16 + i];
            tiles[k * 128 + 64 + i] = dense[k * 16 + 8 + i];
        }
    }
    cb_push_back(x_cb, k_tiles);
    cb_pop_front(xdense_cb, 1);
    // FP32 row partial of the projection to the gate leader.
    cb_wait_front(p2_cb, n_tiles);
    const uint32_t partial = get_read_ptr(p2_cb);
    const uint32_t aggregate = get_write_ptr(agg_h_cb) + stripe * 256;
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write<64>(partial + n * 4096 + f * 1024, peer(gate_leader, aggregate + n * 128 + f * 64), 64);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(peer(gate_leader, get_semaphore(2)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(p2_cb, n_tiles);
    if (stripe != 0) {
        return;
    }
    cb_wait_front(hrow_cb, 1);
    const uint32_t out = get_read_ptr(hrow_cb);
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(4 * group + 3), 2048);
    for (uint32_t n = 0; n < n_tiles; ++n) {
        noc_async_write<32>(out + n * 64, h.get_noc_addr(n, 0), 32);
        noc_async_write<32>(out + n * 64 + 32, h.get_noc_addr(n, 512), 32);
    }
    noc_async_write_barrier();
    cb_pop_front(hrow_cb, 1);
#endif
}
