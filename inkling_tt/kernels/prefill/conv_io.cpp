#include "api/dataflow/dataflow_api.h"

constexpr uint32_t fp_bytes = 4096;
constexpr uint32_t bf_bytes = 2048;
constexpr uint32_t cols = get_compile_time_arg_val(0);
constexpr uint32_t tiles = get_compile_time_arg_val(2);
constexpr bool bf_input = get_compile_time_arg_val(3);
constexpr uint32_t x_bytes = bf_input ? bf_bytes : fp_bytes;
constexpr uint32_t x_row = bf_input ? 32 : 64;

uint32_t row_offset(uint32_t row, uint32_t half) {
    return ((row / 16 * 2 + half) * 256 + row % 16 * 16) * 4;
}

void local_copy(uint32_t src, uint32_t dst, uint32_t bytes) {
    if (bytes) {
        noc_async_read(get_noc_addr(src), dst, bytes);
    }
}

void kernel_main() {
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr auto a0 = TensorAccessorArgs<4>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(a0, addr(0), 32);
    const uint32_t metadata = get_write_ptr(4)
#ifdef PREFILL_WRITE
        + 32
#endif
        ;
    noc_async_read<32>(meta.get_noc_addr(0), metadata, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto descriptor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata);
    const bool fresh = descriptor[0] == 0;
    const uint32_t valid = descriptor[1];
    const uint32_t valid_tiles = (valid + 31) / 32;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);

#ifdef PREFILL_WRITE
    const auto out = TensorAccessor(a1, addr(1), bf_bytes);
    for (uint32_t col = first; col < end; ++col) {
        for (uint32_t t = 0; t < tiles; ++t) {
            cb_wait_front(16, 1);
            noc_async_write<bf_bytes>(get_read_ptr(16), out.get_noc_addr(t * cols + col), bf_bytes);
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
#else
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
#ifdef PREFILL_DYN
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    const auto dyn = TensorAccessor(a4, addr(4), bf_bytes);
#endif
    const auto x = TensorAccessor(a1, addr(1), x_bytes);
    const auto taps = TensorAccessor(a2, addr(2), fp_bytes);
    const auto history = TensorAccessor(a3, addr(3), fp_bytes);
    const uint32_t scratch = get_write_ptr(3);
    const uint32_t saved = scratch + 2 * fp_bytes;
    const uint32_t coeff = scratch + 3 * fp_bytes;
    const uint32_t dyn_data = scratch + 4 * fp_bytes;
    for (uint32_t col = first; col < end; ++col) {
        uint32_t prev = scratch;
        uint32_t curr = scratch + fp_bytes;
        noc_async_read<fp_bytes>(taps.get_noc_addr(col), coeff, fp_bytes);
        if (fresh) {
            for (uint32_t i = 0; i < fp_bytes; i += MEM_ZEROS_SIZE) {
                local_copy(MEM_ZEROS_BASE, saved + i, MEM_ZEROS_SIZE);
            }
        } else {
            noc_async_read<fp_bytes>(history.get_noc_addr(col), saved, fp_bytes);
        }
        noc_async_read_barrier();
        if constexpr (bf_input) {
            invalidate_l1_cache();
        }

        cb_reserve_back(1, 4);
        const uint32_t static_taps = get_write_ptr(1);
        for (uint32_t lag = 0; lag < 4; ++lag) {
            for (uint32_t half = 0; half < 2; ++half) {
                local_copy(coeff + row_offset(lag, half), static_taps + lag * fp_bytes + row_offset(0, half), 64);
                const uint32_t src = saved + row_offset(lag, half);
                const uint32_t dst = prev + row_offset(31 - lag, half) * x_row / 64;
                if constexpr (bf_input) {
                    const auto from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                    auto to = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
                    for (uint32_t lane = 0; lane < 16; ++lane) {
                        to[lane] = from[lane] >> 16;
                    }
                } else {
                    local_copy(src, dst, x_row);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(1, 4);

        for (uint32_t t = 0; t < tiles; ++t) {
            noc_async_read<x_bytes>(x.get_noc_addr(t * cols + col), curr, x_bytes);
#ifdef PREFILL_DYN
            noc_async_read<bf_bytes>(dyn.get_noc_addr(t), dyn_data, bf_bytes);
#endif
            noc_async_read_barrier();
#ifdef PREFILL_DYN
            invalidate_l1_cache();
            const auto raw_dyn = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dyn_data);
            cb_reserve_back(2, 4);
            auto dyn_taps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
            for (uint32_t lag = 0; lag < 4; ++lag) {
                const uint32_t index = col / 4 * 4 + lag;
                for (uint32_t row = 0; row < 32; ++row) {
                    dyn_taps[lag * 1024 + row_offset(row, 0) / 4] =
                        uint32_t(raw_dyn[row_offset(row, index / 16) / 4 + index % 16]) << 16;
                }
            }
            cb_push_back(2, 4);
#endif
            cb_reserve_back(0, 4);
            const uint32_t shifted = get_write_ptr(0);
            for (uint32_t lag = 0; lag < 4; ++lag) {
                const uint32_t dst = shifted + lag * x_bytes;
                for (uint32_t face = 0; face < 4; ++face) {
                    const uint32_t half = face % 2;
                    const uint32_t row = face / 2 * 16;
                    const uint32_t prefix = face < 2 ? prev : curr;
                    local_copy(prefix + row_offset((row + 32 - lag) % 32, half) * x_row / 64,
                               dst + face * (x_bytes / 4), lag * x_row);
                    local_copy(curr + row_offset(row, half) * x_row / 64,
                               dst + face * (x_bytes / 4) + lag * x_row, (16 - lag) * x_row);
                }
            }
            if (t + 1 == valid_tiles) {
                if constexpr (bf_input) {
                    invalidate_l1_cache();
                }
                const int32_t last = (valid - 1) % 32;
                for (int32_t lag = 0; lag < 4; ++lag) {
                    const int32_t row = last - lag;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t src = (row < 0 ? prev : curr) +
                                             row_offset((row + 32) % 32, half) * x_row / 64;
                        const uint32_t dst = saved + row_offset(lag, half);
                        if constexpr (bf_input) {
                            const auto from = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
                            auto to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                            for (uint32_t lane = 0; lane < 16; ++lane) {
                                to[lane] = uint32_t(from[lane]) << 16;
                            }
                        } else {
                            local_copy(src, dst, x_row);
                        }
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(0, 4);
            const uint32_t next = prev;
            prev = curr;
            curr = next;
        }
        noc_async_write<fp_bytes>(saved, history.get_noc_addr(col), fp_bytes);
        noc_async_write_barrier();
    }
#endif
}
