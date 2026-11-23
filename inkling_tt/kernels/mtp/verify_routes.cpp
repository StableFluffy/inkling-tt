#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t gather_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t experts = 128;
    constexpr uint32_t slots = 8;
    constexpr uint32_t inverse_bytes = 32 * slots * sizeof(uint32_t);
    constexpr uint32_t ends_bytes = experts * sizeof(uint32_t);
    constexpr auto ids_args = TensorAccessorArgs<2>();
    constexpr auto ends_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto gather_args = TensorAccessorArgs<ends_args.next_compile_time_args_offset()>();
    constexpr auto inverse_args = TensorAccessorArgs<gather_args.next_compile_time_args_offset()>();
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const auto ends = TensorAccessor(ends_args, addr(rows), ends_bytes);
    const auto gather = TensorAccessor(gather_args, addr(rows + 1), gather_bytes);
    const auto inverse = TensorAccessor(inverse_args, addr(rows + 2), inverse_bytes);
    const uint32_t input = get_write_ptr(0);
    const uint32_t end_address = input + 64 * rows;
    const uint32_t gather_address = end_address + ends_bytes;
    const uint32_t inverse_address = gather_address + gather_bytes;
    const auto ids = [input](uint32_t row) {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input + row * 64);
    };
    uint32_t next[experts] = {};
    for (uint32_t row = 0; row < rows; ++row) {
        const auto src = TensorAccessor(ids_args, addr(row), slots * sizeof(uint16_t));
        noc_async_read<16>(src.get_noc_addr(0), input + row * 64, 16);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            ++next[ids(row)[slot]];
        }
    }
    auto limits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_address);
    uint32_t blocks = 0;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const bool active = next[expert] != 0;
        next[expert] = blocks * slots;
        blocks += active;
        limits[expert] = blocks;
    }
    auto forward = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_address);
    auto reverse = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inverse_address);
    for (uint32_t i = 0; i < gather_bytes / 4; ++i) {
        forward[i] = 0;
    }
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            const uint32_t entry = next[ids(row)[slot]]++;
            forward[entry] = row + 1;
            reverse[row * slots + slot] = entry + 1;
        }
    }
    for (uint32_t i = rows * slots; i < 32 * slots; ++i) {
        reverse[i] = 0;
    }
    noc_async_write<ends_bytes>(end_address, ends.get_noc_addr(0), ends_bytes);
    noc_async_write(gather_address, gather.get_noc_addr(0), gather_bytes);
    noc_async_write<inverse_bytes>(inverse_address, inverse.get_noc_addr(0), inverse_bytes);
    noc_async_write_barrier();
}
