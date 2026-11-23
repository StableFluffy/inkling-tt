// Head tail on the native width-sharded BF16 logits: write the logical row as a row-major
// page and reduce a vocabulary-masked argmax (lowest index on ties) to one core.
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t shard_tiles = get_compile_time_arg_val(0);
constexpr uint32_t total_tiles = get_compile_time_arg_val(1);
constexpr uint32_t vocab = get_compile_time_arg_val(2);
constexpr uint32_t workers = get_compile_time_arg_val(3);
constexpr uint32_t grid_x = get_compile_time_arg_val(4);
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t row_bytes = shard_tiles * 64;
constexpr uint32_t staging_cb = 0;
constexpr uint32_t mailbox_cb = 1;
}  // namespace

void kernel_main() {
    const uint32_t shard = get_common_arg_val<uint32_t>(0);
    const auto logits = TensorAccessor(TensorAccessorArgs<5>(), get_common_arg_val<uint32_t>(1), total_tiles * 64);
    constexpr auto index_args = TensorAccessorArgs<TensorAccessorArgs<5>::next_compile_time_args_offset()>();
    const auto index = TensorAccessor(index_args, get_common_arg_val<uint32_t>(2), 4);
    const uint32_t leader = get_common_arg_val<uint32_t>(3);
    const uint32_t worker = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    // Both data-movement cores of a worker scan half of its shard each.
#ifdef HEAD_TAIL_SECOND
    constexpr uint32_t half = 1;
#else
    constexpr uint32_t half = 0;
#endif
    constexpr uint32_t half_tiles = (shard_tiles + 1) / 2;
    const uint32_t shard_first = worker * shard_tiles;
    const uint32_t shard_count = shard_first + shard_tiles <= total_tiles ? shard_tiles : total_tiles - shard_first;
    const uint32_t local_first = half ? (half_tiles < shard_count ? half_tiles : shard_count) : 0;
    const uint32_t local_end = half ? shard_count : (half_tiles < shard_count ? half_tiles : shard_count);
    const uint32_t first_tile = shard_first + local_first;
    const uint32_t tiles = local_end - local_first;
    const uint32_t slot = 2 * worker + half;

    // Row 0 of each tile lives in the first face row of faces 0 and 1.
    const uint32_t staging = get_write_ptr(staging_cb) + local_first * 64;
    auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
    for (uint32_t t = 0; t < tiles; ++t) {
        auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shard + (local_first + t) * tile_bytes);
#pragma GCC unroll 8
        for (uint32_t w = 0; w < 8; ++w) {
            dst[16 * t + w] = src[w];
            dst[16 * t + 8 + w] = src[128 + w];
        }
    }
    if (tiles) {
        noc_async_write(staging, logits.get_noc_addr(0, first_tile * 64), tiles * 64);
    }

    auto values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging);
    const uint32_t first = first_tile * 32;
    uint32_t count = tiles * 32;
    if (first + count > vocab) {
        count = first >= vocab ? 0 : vocab - first;
    }
    uint32_t best_key = 0;
    uint32_t best = 0xffffffffu;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t bits = values[i];
        if (bits == 0x8000u) {
            bits = 0;
        }
        const uint32_t key = (bits & 0x8000u) ? (~bits & 0xffffu) : (bits | 0x8000u);
        if (key > best_key || best == 0xffffffffu) {
            best_key = key;
            best = first + i;
        }
    }
    const uint32_t mailbox = get_write_ptr(mailbox_cb);
    const uint32_t local = get_write_ptr(staging_cb) + row_bytes + half * 16;
    auto pair = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local);
    pair[0] = count ? best_key : 0;
    pair[1] = count ? best : 0xffffffffu;
    const uint64_t leader_mailbox = get_noc_addr(leader & 0xffff, leader >> 16, mailbox + slot * 16);
    noc_async_write(local, leader_mailbox, 8);
    noc_async_write_barrier();
    const uint32_t arrivals = get_semaphore(0);
    noc_semaphore_inc(get_noc_addr(leader & 0xffff, leader >> 16, arrivals), 1);
    noc_async_atomic_barrier();
    if (worker != 0 || half) {
        return;
    }
    auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals);
    noc_semaphore_wait(sem, 2 * workers);
    noc_semaphore_set(sem, 0);
    auto box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox);
    uint32_t key = 0;
    uint32_t id = 0xffffffffu;
    for (uint32_t w = 0; w < 2 * workers; ++w) {
        const uint32_t k = box[4 * w];
        const uint32_t i = box[4 * w + 1];
        if (i != 0xffffffffu && (id == 0xffffffffu || k > key)) {
            key = k;
            id = i;
        }
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local) = id;
    noc_async_write(local, index.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
