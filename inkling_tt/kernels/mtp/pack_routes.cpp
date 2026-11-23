#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t slots = 8;
constexpr uint32_t routes = 32 * slots;
constexpr uint32_t page = 4096;

uint32_t left_face(uint32_t row) {
    return (row / 16 * 2) * 256 + (row % 16) * 16;
}
}

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto ids_args = TensorAccessorArgs<1>();
    constexpr auto gamma_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto out_ids_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto out_gamma_args = TensorAccessorArgs<out_ids_args.next_compile_time_args_offset()>();
    const auto out_ids = TensorAccessor(out_ids_args, get_common_arg_val<uint32_t>(2 * rows), routes * 4);
    const auto out_gamma = TensorAccessor(out_gamma_args, get_common_arg_val<uint32_t>(2 * rows + 1), page);
    const uint32_t buf = get_write_ptr(0);
    const uint32_t ids_buf = buf + 64 * rows;
    const uint32_t gamma_buf = ids_buf + routes * 4;
    for (uint32_t r = 0; r < rows; ++r) {
        const auto ids = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(2 * r), 16);
        const auto gamma = TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(2 * r + 1), page);
        noc_async_read<16>(ids.get_noc_addr(0), buf + 64 * r, 16);
        noc_async_read<32>(gamma.get_noc_addr(0), buf + 64 * r + 32, 32);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_buf);
    auto weights = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gamma_buf);
    for (uint32_t r = 0; r < 32; ++r) {
        auto picked = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + 64 * r);
        auto values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + 64 * r + 32);
        for (uint32_t s = 0; s < slots; ++s) {
            ids[r * slots + s] = r < rows ? picked[s] : 128;
            const uint32_t w = r < rows ? values[s] : 0;
            const uint32_t left = s * (page / 4) + left_face(r);
            for (uint32_t c = 0; c < 16; ++c) {
                weights[left + c] = w;
                weights[left + c + 256] = w;
            }
        }
    }
    noc_async_write<routes * 4>(ids_buf, out_ids.get_noc_addr(0), routes * 4);
    for (uint32_t s = 0; s < slots; ++s) {
        noc_async_write<page>(gamma_buf + s * page, out_gamma.get_noc_addr(s), page);
    }
    noc_async_write_barrier();
}
