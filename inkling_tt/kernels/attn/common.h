#pragma once

#include <cstdint>

namespace attn {
constexpr uint32_t splits = 16;
constexpr uint32_t queries = 3;
constexpr uint32_t dim_tiles = 4;
constexpr uint32_t bf_tile = 2048;
constexpr uint32_t fp_tile = 4096;
constexpr uint32_t bf_row = 32;
constexpr uint32_t fp_row = 64;
constexpr uint32_t window = 1024;
#ifdef ATTN_ROWS
constexpr uint32_t rows = ATTN_ROWS;
constexpr bool batched_rows = true;
#else
constexpr uint32_t rows = 1;
constexpr bool batched_rows = false;
#endif

inline bool relative_tile_is_zero(uint32_t pos, uint32_t tile) {
    return pos >= window + tile * 32 + 31;
}
constexpr uint32_t scale_bits = 0x3db504f3; // 1 / sqrt(128)
constexpr uint32_t mask_bits = 0xf149f2ca; // FP32 -1e30

constexpr uint32_t q_cb = 0;
constexpr uint32_t k_cb = 1;
constexpr uint32_t v_cb = 2;
constexpr uint32_t r_cb = 3;
constexpr uint32_t table_cb = 4;
constexpr uint32_t compute_position_cb = 5;
constexpr uint32_t stats_inbox = 6;
constexpr uint32_t local_stats_cb = 7;
constexpr uint32_t group_stats_cb = 8;
constexpr uint32_t factor_out_cb = 9;
constexpr uint32_t factor_in_cb = 10;
constexpr uint32_t reader_scratch = 11;
constexpr uint32_t exps_cb = 12;
constexpr uint32_t partials_in_cb = 13;
constexpr uint32_t gate_cb = 14;
constexpr uint32_t probs_cb = 15;
constexpr uint32_t partials_out_cb = 16;
constexpr uint32_t gated_cb = 17;
constexpr uint32_t writer_scratch = 18;
constexpr uint32_t writer_position_cb = 19;
constexpr uint32_t running_stats_cb = 20;
constexpr uint32_t running_values_cb = 21;
constexpr uint32_t stats_sem = 0;
constexpr uint32_t factor_sem = 1;
constexpr uint32_t partials_sem = 2;

struct Work {
    uint32_t first_tile;
    uint32_t tile_count;
    uint32_t active_splits;
    uint32_t valid_keys;

    Work chunk(uint32_t offset, uint32_t size) const {
        Work chunk = *this;
        chunk.first_tile += offset;
        chunk.tile_count = tile_count - offset < size ? tile_count - offset : size;
        return chunk;
    }

    Work(uint32_t pos, uint32_t cache_tiles, uint32_t split) {
        valid_keys = pos < cache_tiles * 32 ? pos + 1 : cache_tiles * 32;
        const uint32_t live_tiles = (valid_keys + 31) / 32;
        const uint32_t tiles_per_split = (live_tiles + splits - 1) / splits;
        active_splits = (live_tiles + tiles_per_split - 1) / tiles_per_split;
        first_tile = split * tiles_per_split;
        const uint32_t left = first_tile < live_tiles ? live_tiles - first_tile : 0;
        tile_count = left < tiles_per_split ? left : tiles_per_split;
    }
};
}
