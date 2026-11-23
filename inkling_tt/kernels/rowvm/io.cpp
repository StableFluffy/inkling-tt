// Move only the logical face rows. Dense scratch is contiguous in physical face order.
#include "../common/read_vc.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef ROWVM_ZONES
#define RV_ZONE(name) DeviceZoneScopedN(name)
#else
#define RV_ZONE(name)
#endif
#include <utility>

#ifdef ROWVM_TASK_BASE
inline uint32_t rowvm_task() {
    return get_absolute_logical_y() * ROWVM_GRID_X + get_absolute_logical_x() - ROWVM_TASK_BASE;
}
#else
inline uint32_t rowvm_task() { return get_arg_val<uint32_t>(0); }
#endif

namespace {
constexpr uint32_t count = get_compile_time_arg_val(0);
constexpr uint32_t region_words = 15;

#ifdef ROWVM_COLLECTIVE
constexpr uint32_t stripes = ROWVM_STRIPES;
constexpr uint32_t aggregate_cb = 8;
constexpr uint32_t broadcast_cb = 9;
constexpr uint32_t partial_cb = 18;
constexpr uint32_t coeff_cb = 19;
constexpr uint32_t gather_sem = 0;
constexpr uint32_t broadcast_sem = 1;
constexpr uint32_t common_coords = count * 3 + ROWVM_OUTPUTS;

uint64_t peer_addr(uint32_t peer, uint32_t address) {
    uint32_t packed = get_common_arg_val<uint32_t>(common_coords + peer);
    return get_noc_addr(packed & 0xffff, packed >> 16, address);
}

void exchange(uint32_t task, bool rms_pass) {
    RV_ZONE("RV_XCHG");
    const uint32_t stripe = task % stripes;
    const uint32_t leader = task - stripe;
    const uint32_t bytes = rms_pass ? 16 : 256;
    const uint32_t received = get_semaphore(broadcast_sem);
    const uint32_t arrivals = get_semaphore(gather_sem);
#ifdef ROWVM_WRITE
    cb_wait_front(partial_cb, 1);
    const uint32_t partial = get_read_ptr(partial_cb);
    // A leader reuses the same mailbox after its compute thread drains it.
    // The CB always wraps by its full capacity (see the reader below).
    const uint32_t aggregate = get_write_ptr(aggregate_cb) + stripe * bytes;
    noc_async_write(partial, peer_addr(leader, aggregate), bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(peer_addr(leader, arrivals), 1);
    cb_pop_front(partial_cb, 1);
    if (stripe == 0) {
        cb_wait_front(coeff_cb, 1);
        const uint32_t coeff = get_read_ptr(coeff_cb);
        const uint32_t dst = get_write_ptr(broadcast_cb);
        for (uint32_t peer = leader; peer < leader + stripes; ++peer) {
            noc_async_write(coeff, peer_addr(peer, dst), bytes);
        }
        noc_async_write_barrier();
        for (uint32_t peer = leader; peer < leader + stripes; ++peer) {
            noc_semaphore_inc(peer_addr(peer, received), 1);
        }
        cb_pop_front(coeff_cb, 1);
    }
#else
    (void)bytes;
    (void)leader;
    if (stripe == 0) {
        cb_reserve_back(aggregate_cb, ROWVM_AGG_PAGES);
        auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals);
        noc_semaphore_wait(sem, stripes);
        noc_semaphore_set(sem, 0);
        cb_push_back(aggregate_cb, ROWVM_AGG_PAGES);
    }
    cb_reserve_back(broadcast_cb, 1);
    auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(received);
    noc_semaphore_wait(sem, 1);
    noc_semaphore_set(sem, 0);
    cb_push_back(broadcast_cb, 1);
#endif
}
#endif

template <uint32_t I, bool Batch = false, bool Commit = false>
void transfer(uint32_t task) {
#if defined(ROWVM_CONV_MIX) && !defined(ROWVM_WRITE)
    if constexpr (I == 0) {
        return;  // The convolution produces the partial on this core.
    }
#endif
#if defined(ROWVM_GROUP_INPUT) && !defined(ROWVM_WRITE)
    if (task / stripes != I) {
        return;
    }
#endif
    constexpr uint32_t spec = 1 + region_words * I;
    constexpr uint32_t cb = get_compile_time_arg_val(spec);
    constexpr uint32_t bytes = get_compile_time_arg_val(spec + 1);
    constexpr uint32_t tile_cols = get_compile_time_arg_val(spec + 2);
    constexpr bool batched_rows = get_compile_time_arg_val(spec + 3);
    constexpr uint32_t rows = get_compile_time_arg_val(spec + 4);
    constexpr uint32_t cols = get_compile_time_arg_val(spec + 5);
    constexpr uint32_t row_step = get_compile_time_arg_val(spec + 6);
    constexpr uint32_t col_step = get_compile_time_arg_val(spec + 7);
    constexpr bool concat_rows = get_compile_time_arg_val(spec + 8);
    constexpr uint32_t group_col_step = get_compile_time_arg_val(spec + 9);
    constexpr uint32_t extent = get_compile_time_arg_val(spec + 10);
    constexpr bool shared_prefix = get_compile_time_arg_val(spec + 11);
    constexpr bool whole_tiles = get_compile_time_arg_val(spec + 12);
    constexpr uint32_t cache_length = get_compile_time_arg_val(spec + 13);
    constexpr bool row_major = get_compile_time_arg_val(spec + 14);
    constexpr uint32_t dense_tiles = (cols + 1023) / 1024;
    constexpr uint32_t face_bytes = 16 * bytes;
    constexpr uint32_t tile_bytes = 1024 * bytes;

    constexpr auto args = TensorAccessorArgs<get_compile_time_arg_val(1 + region_words * count + I)>();
    uint32_t address = get_common_arg_val<uint32_t>(I);
#if defined(ROWVM_SPLIT_OUTPUT) && defined(ROWVM_WRITE)
    if constexpr (I == 0) {
        address = get_common_arg_val<uint32_t>(count * 3 + task / stripes);
    }
#endif
    const auto tensor = TensorAccessor(args, address, row_major ? tile_cols * 32 * bytes : tile_bytes);
#ifndef ROWVM_WRITE
    constexpr bool defer = Batch && (bytes == 4 || row_major || whole_tiles || !decltype(tensor)::is_dram);
    if constexpr (Commit) {
        if constexpr (defer) {
            cb_push_back(cb, rows * (whole_tiles ? cols / 32 : dense_tiles));
        }
        return;
    }
#endif
    constexpr uint32_t task_cols = get_compile_time_arg_val(1 + region_words * count + count);
    uint32_t row_task = task;
    uint32_t col_task = task;
    if constexpr (task_cols > 0) {
        row_task = task / (task_cols > 0 ? task_cols : 1);
        col_task = task % (task_cols > 0 ? task_cols : 1);
    }
    const uint32_t start_row = get_common_arg_val<uint32_t>(count + 2 * I) + row_task * row_step;
    const uint32_t start_face = get_common_arg_val<uint32_t>(count + 2 * I + 1) +
                                col_task * col_step + row_task * group_col_step;
#ifndef ROWVM_WRITE
    if constexpr (whole_tiles) {
        constexpr uint32_t tile_count = cols / 32;
        cb_reserve_back(cb, rows * tile_count);
        const uint32_t dst = get_write_ptr(cb);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t t = 0; t < tile_count; ++t) {
                const uint32_t tile = (start_row + r) * tile_cols + start_face / 2 + t;
                noc_async_read<tile_bytes>(tensor.get_noc_addr(tile), dst + (r * tile_count + t) * tile_bytes, tile_bytes);
            }
        }
        if constexpr (!defer) {
            noc_async_read_barrier();
            cb_push_back(cb, rows * tile_count);
        }
        return;
    }
#endif
    uint32_t valid_cols = cols;
    if constexpr (extent != 0 && extent % cols != 0) {
        const uint32_t remaining = extent - 16 * col_task * col_step;
        valid_cols = remaining < cols ? remaining : cols;
    }
#ifdef ROWVM_WRITE
    cb_wait_front(cb, rows * dense_tiles);
    const uint32_t dense = get_read_ptr(cb);
#ifdef ROWVM_CACHE_POSITION_ARGS
    if constexpr (cache_length != 0) {
        const auto position = TensorAccessor(
            TensorAccessorArgs<ROWVM_CACHE_POSITION_ARGS>(), get_common_arg_val<uint32_t>(count * 3), 4);
        const uint32_t addr = get_write_ptr(30);
        noc_async_read<4>(position.get_noc_addr(0), addr, 4);
        noc_async_read_barrier();
        uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        if constexpr (ROWVM_CACHE_WRAP) {
            pos &= cache_length - 1;
        }
        if (pos < cache_length) {
            const uint32_t page = (task * (cache_length / 32) + pos / 32) * tile_cols;
            const uint32_t inner = pos % 32;
            for (uint32_t f = 0; f < cols / 16; ++f) {
                const uint32_t offset = (inner / 16 * 32 + (f % 2) * 16 + inner % 16) * face_bytes;
                noc_async_write<face_bytes>(dense + f * face_bytes, tensor.get_noc_addr(page + f / 2, offset), face_bytes);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb, rows * dense_tiles);
        return;
    }
#endif
#else
    cb_reserve_back(cb, rows * dense_tiles);
    const uint32_t dense = get_write_ptr(cb);
    // One even/odd SFPU pair covers 64 consecutive physical values.
    if constexpr (cols % 64 || (extent != 0 && extent % cols != 0)) {
        for (uint32_t r = 0; r < rows; ++r) {
            auto end = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                dense + r * dense_tiles * tile_bytes + valid_cols * bytes);
            for (uint32_t w = 0; w < (((cols + 63) / 64) * 64 - valid_cols) * bytes / 4; ++w) {
                end[w] = 0;
            }
        }
    }
    // DRAM reads require source and destination to agree modulo 64. A BF16 face
    // row occupies 32 bytes, so stage mismatched faces in a separate aligned CB.
    uint32_t bounce = 0;
    if constexpr (bytes == 2 && decltype(tensor)::is_dram) {
        bounce = get_write_ptr(31);
    }
#endif
    if constexpr (row_major) {
        // One contiguous transfer per logical row: a page holds the whole row.
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t row = shared_prefix ? (r == 0 ? 0 : start_row + r - 1)
                                               : start_row + (concat_rows ? 0 : r);
            const uint32_t local = dense + r * dense_tiles * tile_bytes;
            const uint32_t offset = (start_face + (concat_rows ? r * (valid_cols / 16) : 0)) * face_bytes;
#ifdef ROWVM_WRITE
            noc_async_write(local, tensor.get_noc_addr(row, offset), valid_cols * bytes);
#else
            noc_async_read(tensor.get_noc_addr(row, offset), local, valid_cols * bytes);
#endif
        }
#ifdef ROWVM_WRITE
        noc_async_write_barrier();
        cb_pop_front(cb, rows * dense_tiles);
#else
        if constexpr (!defer) {
            noc_async_read_barrier();
            cb_push_back(cb, rows * dense_tiles);
        }
#endif
        return;
    }
    const uint32_t faces = valid_cols / 16;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t row = shared_prefix ? (r == 0 ? 0 : start_row + r - 1)
                                           : start_row + (concat_rows ? 0 : r);
        const uint32_t tile_row = batched_rows ? row : row / 32;
        const uint32_t inner_row = batched_rows ? 0 : row % 32;
        const uint32_t local = dense + r * dense_tiles * tile_bytes;
        for (uint32_t f = 0; f < faces; ++f) {
            const uint32_t face = start_face + f + (concat_rows ? r * faces : 0);
            const uint32_t tile = tile_row * tile_cols + face / 2;
            const uint32_t offset = (inner_row / 16 * 32 + (face % 2) * 16 + inner_row % 16) * face_bytes;
            const uint64_t remote = tensor.get_noc_addr(tile, offset);
#ifdef ROWVM_WRITE
            uint32_t src = local + f * face_bytes;
#ifdef ROWVM_PAD_HEAD
            if (f == 1) {
                src = MEM_ZEROS_BASE;
            }
#endif
            noc_async_write<face_bytes>(src, remote, face_bytes);
#else
            uint32_t dst = local + f * face_bytes;
            if constexpr (bytes == 2 && decltype(tensor)::is_dram) {
                if ((dst ^ offset) & 32) {
                    dst = bounce + 64 * (f / 2) + (offset & 32);
                }
            }
            noc_async_read<face_bytes>(remote, dst, face_bytes);
#endif
        }
#ifndef ROWVM_WRITE
        if constexpr (bytes == 2 && decltype(tensor)::is_dram) {
            noc_async_read_barrier();
            for (uint32_t f = 0; f < faces; ++f) {
                if (((f ^ inner_row) & 1) != 0) {
                    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        bounce + 64 * (f / 2) + (inner_row % 2) * 32);
                    auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local + f * 32);
#pragma GCC unroll 8
                    for (uint32_t i = 0; i < 8; ++i) {
                        dst[i] = src[i];
                    }
                }
            }
        }
#endif
    }
#ifdef ROWVM_WRITE
    noc_async_write_barrier();
    cb_pop_front(cb, rows * dense_tiles);
#else
    if constexpr (!defer) {
        noc_async_read_barrier();
        cb_push_back(cb, rows * dense_tiles);
    }
#endif
}

#if defined(ROWVM_MESSAGE_UP) && defined(ROWVM_WRITE)
void gather_logit_rows() {
    RV_ZONE("RV_GATHER");
    constexpr uint32_t projection_cb = 15;
    cb_wait_front(projection_cb, 4);
    const auto tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(projection_cb));
    for (uint32_t gate = 0; gate < 2; ++gate) {
        const uint32_t target_cb = 2 + gate;
        cb_reserve_back(target_cb, 1);
        auto row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(target_cb));
        for (uint32_t tile = 0; tile < 2; ++tile) {
            const auto src = tiles + (gate * 2 + tile) * 1024;
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16; ++i) {
                row[32 * tile + i] = src[i];
                row[32 * tile + 16 + i] = src[256 + i];
            }
        }
        cb_push_back(target_cb, 1);
    }
    cb_pop_front(projection_cb, 4);
}
#endif

#if defined(ROWVM_ROUTER) && defined(ROWVM_WRITE)
void select_experts() {
    constexpr uint32_t scores_cb = 18;
    constexpr uint32_t selection_cb = 8;
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
        // Monotonic unsigned keys for signed IEEE floats. Iterating in expert
        // order and using strict comparison gives low-index-first ties.
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
    // Keep IDs on a 16-byte boundary outside the FP32 values and reduction lanes.
    auto id_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(selected + 256);
    for (uint32_t i = 0; i < 64; ++i) {
        values[i] = i < k ? scores[ids[i]] : 0;
    }
    for (uint32_t i = 0; i < k; ++i) {
        id_stick[i] = ids[i];
    }
    constexpr auto args = TensorAccessorArgs<ROWVM_INDEX_ARGS>();
    const auto index = TensorAccessor(args, get_common_arg_val<uint32_t>(count * 3), 16);
    noc_async_write<16>(selected + 256, index.get_noc_addr(0), 16);
    noc_async_write_barrier();
    cb_pop_front(scores_cb, 2);
    cb_push_back(selection_cb, 1);
}
#endif

template <uint32_t... I>
void transfer_all(uint32_t task, std::integer_sequence<uint32_t, I...>) {
    (transfer<I>(task), ...);
}

#if defined(ROWVM_CONV_MIX) && !defined(ROWVM_WRITE)
template <uint32_t Offset, uint32_t... I>
void read_batch(uint32_t task, std::integer_sequence<uint32_t, I...>) {
    (transfer<I + Offset, true, false>(task), ...);
    noc_async_read_barrier();
    (transfer<I + Offset, true, true>(task), ...);
}
#endif

// A convolution mix finishes its history and residual rows before the collectives;
// the writer drains them while the stripes exchange mix statistics.
#if defined(ROWVM_CONV_MIX) && defined(ROWVM_WRITE)
constexpr bool early_output(uint32_t i) { return i == 1 || i == 2; }
#else
constexpr bool early_output(uint32_t) { return false; }
#endif

template <uint32_t... I>
void transfer_late(uint32_t task, std::integer_sequence<uint32_t, I...>) {
    ((early_output(I) ? void() : transfer<I>(task)), ...);
}

}  // namespace

void kernel_main() {
    select_read_vc();
    const uint32_t task = rowvm_task();
#ifndef ROWVM_WRITE
    {
    RV_ZONE("RV_READ");
#ifdef ROWVM_CONV_MIX
    read_batch<4>(task, std::make_integer_sequence<uint32_t, count - 4>{});
    read_batch<0>(task, std::make_integer_sequence<uint32_t, 4>{});
#else
    transfer_all(task, std::make_integer_sequence<uint32_t, count>{});
#endif
    }
#endif
#if defined(ROWVM_CONV_MIX) && defined(ROWVM_WRITE)
    transfer<1>(task);
    transfer<2>(task);
#endif
#if defined(ROWVM_MESSAGE_UP) && defined(ROWVM_WRITE)
    gather_logit_rows();
#endif
#ifdef ROWVM_COLLECTIVE
#if ROWVM_APPEND_BANK
    exchange(task, true);
#endif
    exchange(task, ROWVM_RMS_ONLY);
#if ROWVM_SECOND_PASS
    exchange(task, true);
#endif
#endif
#ifdef ROWVM_WRITE
#ifdef ROWVM_ROUTER
    select_experts();
#endif
    RV_ZONE("RV_WRITE");
    transfer_late(task, std::make_integer_sequence<uint32_t, count>{});
#endif
}
