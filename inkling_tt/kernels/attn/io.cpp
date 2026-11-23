#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.h"

using namespace attn;

namespace {
constexpr uint32_t cache_tiles = get_compile_time_arg_val(0);
constexpr bool global_cache = get_compile_time_arg_val(1);
constexpr uint32_t split_tiles = (cache_tiles + splits - 1) / splits;
constexpr uint32_t chunk_tiles = split_tiles < 4 ? split_tiles : 4;
constexpr bool streaming = split_tiles > chunk_tiles;
constexpr uint32_t gate_tile = get_compile_time_arg_val(2);
constexpr uint32_t grid_width = get_compile_time_arg_val(3);
constexpr uint32_t first_head = get_compile_time_arg_val(4);
constexpr uint32_t operand_count = 9;

uint64_t peer_addr(uint32_t head, uint32_t split, uint32_t addr) {
    const uint32_t xy = get_common_arg_val<uint32_t>(operand_count + (head - first_head) * splits + split);
    return get_noc_addr(xy & 0xffff, xy >> 16, addr);
}

void await_arrivals(uint32_t id, uint32_t count) {
    auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
    noc_semaphore_wait(sem, count);
    noc_semaphore_set(sem, 0);
}

template <uint32_t I>
auto operand(uint32_t bytes) {
    constexpr auto args = TensorAccessorArgs<get_compile_time_arg_val(5 + I)>();
    return TensorAccessor(args, get_common_arg_val<uint32_t>(I), bytes);
}

template <class Accessor>
void read_cache(const Accessor& cache, uint32_t head, const Work& work, uint32_t cb) {
    const uint32_t count = work.tile_count * dim_tiles;
    const uint32_t first = (head * cache_tiles + work.first_tile) * dim_tiles;
    const uint32_t dst = get_write_ptr(cb);
    // Interleaved pages start at the same bank for many workers. Rotate the
    // issue order to use more banks while preserving the CB's tile order.
    uint32_t tile = (head + work.first_tile / 2) & 7;
    if (tile >= count) {
        tile -= count;
    }
    for (uint32_t i = 0; i < count; ++i) {
        noc_async_read<bf_tile>(cache.get_noc_addr(first + tile), dst + tile * bf_tile, bf_tile);
        tile = tile + 1 == count ? 0 : tile + 1;
    }
}

void read_values(uint32_t head, const Work& work) {
    const auto v = operand<2>(bf_tile);
    const uint32_t kv_tiles = work.tile_count * dim_tiles;
    cb_reserve_back(v_cb, kv_tiles);
    read_cache(v, head, work, v_cb);
    noc_async_read_barrier();
    cb_push_back(v_cb, kv_tiles);
}

#ifndef ATTN_WRITER
uint32_t read_position() {
    const auto position = operand<6>(4);
    cb_reserve_back(compute_position_cb, 1);
    cb_reserve_back(writer_position_cb, 1);
    const uint32_t dst = get_write_ptr(compute_position_cb);
    noc_async_read<4>(position.get_noc_addr(0), dst, 4);
    noc_async_read_barrier();
    const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(writer_position_cb)) = pos;
    cb_push_back(compute_position_cb, 1);
    cb_push_back(writer_position_cb, 1);
    return pos;
}

void read_inputs(uint32_t head, uint32_t split, uint32_t pos, const Work& work, bool first_chunk = true) {
    DeviceZoneScopedN("INK_ATTN_READ");
    const auto q = operand<0>(bf_tile);
    const auto k = operand<1>(bf_tile);
    const auto r = operand<3>(bf_tile);
    const auto sink = operand<4>(bf_tile);
    const auto table = operand<5>(bf_tile);
    const auto gate = operand<7>(bf_tile);

    const bool active = work.tile_count != 0;
    const bool relative = active && (!global_cache || !relative_tile_is_zero(pos, work.first_tile + work.tile_count - 1));
    const uint32_t kv_tiles = work.tile_count * dim_tiles;
    if (active) {
        cb_reserve_back(q_cb, dim_tiles);
        cb_reserve_back(k_cb, kv_tiles);
        cb_reserve_back(table_cb, work.tile_count);
    }
    if (relative) {
        cb_reserve_back(r_cb, 1);
    }
    const uint32_t scratch = get_write_ptr(reader_scratch);
    const uint32_t row = head & 1; // Preserve BF16 row parity for DRAM reads.
    const uint32_t q_addr = get_write_ptr(q_cb);
    const uint32_t r_addr = get_write_ptr(r_cb);
    const bool output_worker = first_chunk && split < dim_tiles;
    if (output_worker) {
        cb_reserve_back(gate_cb, 1);
    }
    if (active) {
        // All three Q and R rows normally sit in the same tile face.
        // The group beginning at query 15 crosses into the lower faces.
        for (uint32_t first = 0; first < queries;) {
            const uint32_t q_head = head * queries + first;
            const uint32_t face_left = 16 - q_head % 16;
            const uint32_t count = queries - first < face_left ? queries - first : face_left;
            const uint32_t bytes = count * bf_row;
            const uint32_t offset = q_head / 16 * 1024 + q_head % 16 * bf_row;
            const uint32_t dest_row = (row + first) * bf_row;
            if (relative) {
                noc_async_read(r.get_noc_addr(0, offset), r_addr + dest_row, bytes);
            }
            for (uint32_t d = 0; d < dim_tiles; ++d) {
                const uint32_t dst = q_addr + d * bf_tile + dest_row;
                noc_async_read(q.get_noc_addr(d, offset), dst, bytes);
                noc_async_read(q.get_noc_addr(d, offset + 512), dst + 512, bytes);
            }
            first += count;
        }
        if (relative) {
            noc_async_read<queries * bf_row>(get_noc_addr(MEM_ZEROS_BASE), r_addr + 512 + row * bf_row, queries * bf_row);
        }
    }
    for (uint32_t h = 0; h < queries; ++h) {
        const uint32_t q_head = head * queries + h;
        const uint32_t offset = q_head / 16 * 1024 + q_head % 16 * bf_row;
        const uint32_t dest_row = (row + h) * bf_row;
        if (first_chunk && split == 0) {
            noc_async_read<bf_row>(sink.get_noc_addr(0, offset),
                                  scratch + (h + 1) * 64 + (q_head & 1) * bf_row, bf_row);
        }
        if (output_worker) {
            const uint32_t g_tile = gate_tile + q_head * dim_tiles + split;
            const uint32_t dst = get_write_ptr(gate_cb) + dest_row;
            noc_async_read<bf_row>(gate.get_noc_addr(g_tile, 0), dst, bf_row);
            noc_async_read<bf_row>(gate.get_noc_addr(g_tile, 512), dst + 512, bf_row);
        }
    }
    if (active) {
        read_cache(k, head, work, k_cb);
    }
    noc_async_read_barrier();
    if (output_worker) {
        cb_push_back(gate_cb, 1);
    }
    if (!active) {
        return;
    }
    cb_push_back(q_cb, dim_tiles);
    if (relative) {
        cb_push_back(r_cb, 1);
    }
    cb_push_back(k_cb, kv_tiles);

    constexpr uint32_t blocks = window / 32 + int(global_cache);
    const uint32_t phase = pos % 32;
    const uint32_t table_addr = get_write_ptr(table_cb);
    if (relative) {
        for (uint32_t i = 0; i < work.tile_count; ++i) {
            const uint32_t key_tile = work.first_tile + i;
            const uint32_t distance = pos / 32 - key_tile;
            if (global_cache && relative_tile_is_zero(pos, key_tile)) {
                continue;
            }
            const uint32_t block = global_cache ? distance : distance & (blocks - 1);
            const uint32_t dst = table_addr + i * bf_tile;
            noc_async_read<bf_tile>(table.get_noc_addr(phase * blocks + block), dst, bf_tile);
        }
        noc_async_read_barrier();
    }
    cb_push_back(table_cb, work.tile_count);

}

void gather_stats(uint32_t head, uint32_t active) {
    DeviceZoneScopedN("INK_ATTN_GATHER_STATS");
    await_arrivals(stats_sem, active);
    cb_reserve_back(group_stats_cb, 2);
    auto maxes = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(group_stats_cb));
    auto sums = maxes + fp_tile / 4;
    const auto inbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(stats_inbox));
    const auto sinks = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(reader_scratch));
    const uint32_t row = head & 1;
    for (uint32_t h = 0; h < queries; ++h) {
        for (uint32_t split = 0; split < splits; ++split) {
            maxes[(row + h) * 16 + split] = split < active ? inbox[split * 8 + h * 2] : mask_bits;
            sums[(row + h) * 16 + split] = split < active ? inbox[split * 8 + h * 2 + 1] : 0;
        }
        for (uint32_t j = 1; j < 16; ++j) {
            maxes[256 + (row + h) * 16 + j] = mask_bits;
            sums[256 + (row + h) * 16 + j] = 0;
        }
        maxes[256 + (row + h) * 16] = uint32_t(sinks[(h + 1) * 32 + ((head * queries + h) & 1) * 16]) << 16;
        sums[256 + (row + h) * 16] = 0x3f800000;
    }
    cb_push_back(group_stats_cb, 2);
}
#else
void send_stats(uint32_t head, uint32_t split) {
    cb_wait_front(local_stats_cb, 1);
    const auto local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(local_stats_cb));
    const uint32_t scratch = get_write_ptr(writer_scratch);
    auto msg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t row = head & 1;
    for (uint32_t h = 0; h < queries; ++h) {
        msg[h * 2] = local[(row + h) * 16];
        msg[h * 2 + 1] = local[(row + h) * 16 + 1];
    }
    noc_async_write(scratch, peer_addr(head, 0, get_write_ptr(stats_inbox) + split * 32), 32);
    noc_async_write_barrier();
    noc_semaphore_inc(peer_addr(head, 0, get_semaphore(stats_sem)), 1);
    cb_pop_front(local_stats_cb, 1);
}

void broadcast_factors(uint32_t head, uint32_t active) {
    cb_wait_front(factor_out_cb, 1);
    const uint32_t src = get_read_ptr(factor_out_cb);
    for (uint32_t split = 0; split < active; ++split) {
        noc_async_write(src, peer_addr(head, split, get_write_ptr(factor_in_cb)), (queries + (head & 1)) * fp_row);
    }
    noc_async_write_barrier();
    for (uint32_t split = 0; split < active; ++split) {
        noc_semaphore_inc(peer_addr(head, split, get_semaphore(factor_sem)), 1);
    }
    cb_pop_front(factor_out_cb, 1);
}

void send_partials(uint32_t head, uint32_t split) {
    DeviceZoneScopedN("INK_ATTN_SEND_VALUES");
    cb_wait_front(partials_out_cb, dim_tiles);
    const uint32_t src = get_read_ptr(partials_out_cb);
    const uint32_t dst = get_write_ptr(partials_in_cb) + split * fp_tile;
    for (uint32_t d = 0; d < dim_tiles; ++d) {
        const uint32_t tile = src + d * fp_tile;
        noc_async_write(tile, peer_addr(head, d, dst), (queries + (head & 1)) * fp_row);
        noc_async_write(tile + 1024, peer_addr(head, d, dst + 1024), (queries + (head & 1)) * fp_row);
    }
    noc_async_write_barrier();
    for (uint32_t d = 0; d < dim_tiles; ++d) {
        noc_semaphore_inc(peer_addr(head, d, get_semaphore(partials_sem)), 1);
    }
    cb_pop_front(partials_out_cb, dim_tiles);
}

void write_output(uint32_t head, uint32_t d) {
    const uint32_t row = head & 1;
    const auto out = operand<8>(bf_tile);
    cb_wait_front(gated_cb, 1);
    const uint32_t src = get_read_ptr(gated_cb);
    for (uint32_t h = 0; h < queries; ++h) {
        const uint32_t tile = (head * queries + h) * dim_tiles + d;
        for (uint32_t face = 0; face < 2; ++face) {
            noc_async_write(src + face * 512 + (row + h) * bf_row, out.get_noc_addr(tile, face * 512), bf_row);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(gated_cb, 1);
}
#endif
}  // namespace

void kernel_main() {
    const uint32_t worker = get_absolute_logical_y() * grid_width + get_absolute_logical_x();
    const uint32_t head = first_head + worker / splits;
    const uint32_t split = worker % splits;
#ifndef ATTN_WRITER
    const uint32_t pos = read_position();
    const Work work(pos, cache_tiles, split);
    if (work.tile_count || split < dim_tiles) {
        if constexpr (streaming) {
            if (!work.tile_count) {
                read_inputs(head, split, pos, work);
            }
            for (uint32_t offset = 0; offset < work.tile_count; offset += chunk_tiles) {
                read_inputs(head, split, pos, work.chunk(offset, chunk_tiles), offset == 0);
            }
        } else {
            read_inputs(head, split, pos, work);
        }
    }
    if (split == 0) {
        gather_stats(head, work.active_splits);
    }
    if (work.tile_count) {
        cb_reserve_back(factor_in_cb, 1);
        await_arrivals(factor_sem, 1);
        cb_push_back(factor_in_cb, 1);
    }
    if (split < dim_tiles) {
        cb_reserve_back(partials_in_cb, work.active_splits);
        await_arrivals(partials_sem, work.active_splits);
        cb_push_back(partials_in_cb, work.active_splits);
    }
#else
    cb_wait_front(writer_position_cb, 1);
    const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(writer_position_cb));
    cb_pop_front(writer_position_cb, 1);
    const Work work(pos, cache_tiles, split);
    if (work.tile_count) {
        if constexpr (!streaming) {
            read_values(head, work);
        }
        send_stats(head, split);
        if (split == 0) {
            broadcast_factors(head, work.active_splits);
        }
        if constexpr (streaming) {
            for (uint32_t offset = 0; offset < work.tile_count; offset += chunk_tiles) {
                read_values(head, work.chunk(offset, chunk_tiles));
            }
        }
        send_partials(head, split);
    }
    if (split < dim_tiles) {
        write_output(head, split);
    }
#endif
}
