#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t bf_tile = 2048;
constexpr uint32_t fp_tile = 4096;
constexpr uint32_t address_row = 64;
constexpr uint32_t kv_tiles = 32;
constexpr uint32_t visit_tiles = 272;

uint32_t face_row(uint32_t row, uint32_t half, uint32_t bytes) {
    return ((row / 16 * 2 + half) * 16 + row % 16) * bytes;
}
}

void kernel_main() {
    constexpr uint32_t verify_rows = get_compile_time_arg_val(0);
    constexpr auto meta_args = TensorAccessorArgs<1>();
    constexpr auto table_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto hist_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(0), 32);
    const auto table = TensorAccessor(table_args, get_common_arg_val<uint32_t>(1), address_row);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t buf = scratch + address_row;
    noc_async_read<32>(meta.get_noc_addr(0), buf, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t start = header[0];
    const uint32_t rows = header[1];
    if (rows == 0 || rows > verify_rows) {
        return;
    }
    auto addresses = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t visit = 0xffffffffu;
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        if (task / visit_tiles != visit) {
            visit = task / visit_tiles;
            noc_async_read<address_row>(table.get_noc_addr(visit), scratch, address_row);
            noc_async_read_barrier();
            invalidate_l1_cache();
        }
        const uint32_t tile = task % visit_tiles;
        if (tile < 2 * kv_tiles) {
            const uint32_t kind = tile / kv_tiles;
            const uint32_t head = tile % kv_tiles / 4;
            const uint32_t col = tile % 4;
            const uint32_t capacity = addresses[12];
            const auto live = TensorAccessor(kv_args, addresses[kind], bf_tile);
            const auto staged = TensorAccessor(kv_args, addresses[2 + kind], bf_tile);
            noc_async_read<bf_tile>(staged.get_noc_addr(head * 4 + col), buf, bf_tile);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t p = (start + r) % capacity;
                const uint32_t live_tile = (head * (capacity / 32) + p / 32) * 4 + col;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write<32>(
                        buf + face_row(r, f, 32), live.get_noc_addr(live_tile, face_row(p % 32, f, 32)), 32);
                }
            }
        } else {
            uint32_t column = tile - 2 * kv_tiles;
            uint32_t kind = 0;
            for (uint32_t width = 32; column >= width; width = kind < 2 ? 32 : 72) {
                column -= width;
                ++kind;
            }
            const auto live = TensorAccessor(hist_args, addresses[4 + kind], fp_tile);
            const auto staged = TensorAccessor(hist_args, addresses[8 + kind], fp_tile);
            for (uint32_t lag = 0; lag < 4; ++lag) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint64_t from = lag < rows
                        ? staged.get_noc_addr(column, face_row(rows - lag - 1, f, 64))
                        : live.get_noc_addr(column, face_row(lag - rows, f, 64));
                    noc_async_read<64>(from, buf + (2 * lag + f) * 64, 64);
                }
            }
            noc_async_read_barrier();
            for (uint32_t lag = 0; lag < 4; ++lag) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write<64>(buf + (2 * lag + f) * 64, live.get_noc_addr(column, face_row(lag, f, 64)), 64);
                }
            }
        }
        noc_async_write_barrier();
    }
}
