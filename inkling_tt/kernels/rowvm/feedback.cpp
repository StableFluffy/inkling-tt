// Greedy token feedback: the next step's embedding row, position, tau and an ID history entry
// from the head's argmax. Scalars move as aligned 64-byte blocks (read-modify-write).
#include "../common/read_vc.h"

namespace {
constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
constexpr uint32_t lmax = get_compile_time_arg_val(1);
constexpr auto index_args = TensorAccessorArgs<2>();
constexpr auto table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
constexpr auto x_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr auto pos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto tau_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
constexpr auto tau_args = TensorAccessorArgs<tau_table_args.next_compile_time_args_offset()>();
constexpr auto ids_args = TensorAccessorArgs<tau_args.next_compile_time_args_offset()>();

struct Word {
    uint64_t block;
    uint32_t l1;
};

Word fetch(uint32_t buffer, uint64_t remote) {
    const uint32_t offset = static_cast<uint32_t>(remote) & 63;
    noc_async_read(remote - offset, buffer, 64);
    return {remote - offset, buffer + offset};
}

volatile tt_l1_ptr uint32_t& word(const Word& w) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w.l1); }
}  // namespace

void kernel_main() {
    select_read_vc();
    const auto index = TensorAccessor(index_args, get_common_arg_val<uint32_t>(0), 4);
    const auto table = TensorAccessor(table_args, get_common_arg_val<uint32_t>(1), row_bytes);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(2), row_bytes);
    const auto pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(3), 4);
    const auto tau_table = TensorAccessor(tau_table_args, get_common_arg_val<uint32_t>(4), lmax * 4);
    const auto tau = TensorAccessor(tau_args, get_common_arg_val<uint32_t>(5), 4096);
    const auto ids = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(6), lmax * 4);
    const uint32_t base = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t row = base + 6 * 64;

    const Word id = fetch(base, index.get_noc_addr(0));
    const Word position = fetch(base + 64, pos.get_noc_addr(0));
    const Word tau_out = fetch(base + 128, tau.get_noc_addr(0));
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t token = word(id);
    const uint32_t next = word(position) + 1;
    const uint32_t clamped = next < lmax ? next : lmax - 1;

    noc_async_read(table.get_noc_addr(token), row, row_bytes);
    const Word tau_in = fetch(base + 192, tau_table.get_noc_addr(0, clamped * 4));
    const Word history = fetch(base + 256, ids.get_noc_addr(0, clamped * 4));
    noc_async_read_barrier();
    invalidate_l1_cache();

    noc_async_write(row, x.get_noc_addr(0), row_bytes);
    word(tau_out) = word(tau_in);
    noc_async_write(base + 128, tau_out.block, 64);
    word(position) = next;
    noc_async_write(base + 64, position.block, 64);
    if (next < lmax) {
        word(history) = token;
        noc_async_write(base + 256, history.block, 64);
    }
    noc_async_write_barrier();
}
