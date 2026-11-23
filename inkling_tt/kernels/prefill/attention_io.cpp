#include "api/dataflow/dataflow_api.h"
#include "query_group.h"

constexpr uint32_t bf_bytes = 2048;
constexpr uint32_t fp_bytes = 4096;
constexpr uint32_t width_tiles = 4;
constexpr uint32_t window_tiles = 32;
constexpr uint32_t value_group_keys = 8;
constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cache_tiles = get_compile_time_arg_val(1);
constexpr bool global = get_compile_time_arg_val(2);
constexpr uint32_t heads = get_compile_time_arg_val(3);
constexpr uint32_t saved_tiles = get_compile_time_arg_val(4);

// Alternate early and late query tiles so contiguous task ranges have
// similar causal spans. The per-task arithmetic is unchanged.
uint32_t query_tile(uint32_t task) {
    const uint32_t i = task % q_tiles;
    return i % 2 == 0 ? i / 2 : q_tiles - 1 - i / 2;
}

uint32_t face_index(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}

auto words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

template <typename Acc>
void issue_head_tile(const Acc& tensor, uint32_t tile, uint32_t dest) {
    for (uint32_t d = 0; d < width_tiles; ++d) {
        noc_async_read<bf_bytes>(tensor.get_noc_addr(tile * width_tiles + d), dest + d * bf_bytes, bf_bytes);
    }
}

template <typename Acc>
void read_head_tile(const Acc& tensor, uint32_t tile, uint32_t cb) {
    cb_reserve_back(cb, width_tiles);
    issue_head_tile(tensor, tile, get_write_ptr(cb));
}

// Row r of a bias tile at lag L reads the reversed distance logits of tiles
// 31-L and 32-L at positions 31-r+c. Both window edges use the same shear:
// positions past tile 31 (lag 0) or before tile 0 (lag 32) are not visible.
void make_bias(uint32_t dst, uint32_t src, uint32_t lag) {
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
    const uint32_t* distance = reinterpret_cast<const uint32_t*>(src);
    constexpr uint32_t masked = 0xff800000u;
    const uint32_t hidden = lag == 0 || !global ? masked : 0u;
    const int32_t base = static_cast<int32_t>((window_tiles - lag) * 1024) - 1024;
    for (uint32_t row = 0; row < 32; ++row) {
        const uint32_t x = 31 - row;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t offset = x + half * 16;
            const uint32_t col = offset % 16;
            const int32_t tile = base + static_cast<int32_t>(offset / 32 * 1024);
            const int32_t face = static_cast<int32_t>(face_index(row, offset % 32) - col);
            const int32_t following = offset % 32 < 16 ? 256 : 1024 - 256;
            uint32_t* dest = out + face_index(row, half * 16);
            const uint32_t n = 16 - col;
            // Piece a covers positions [offset, offset + n), piece b the rest of the 16.
            const uint32_t b_pos = offset + n;
            const bool a_visible = lag == 0 ? offset < 32 : lag == window_tiles ? offset >= 32 : true;
            const bool b_visible = lag == 0 ? b_pos < 32 : lag == window_tiles ? b_pos >= 32 : true;
            if (a_visible) {
                const uint32_t* a = distance + tile + face + col;
#pragma GCC unroll 16
                for (uint32_t i = 0; i < n; ++i) {
                    dest[i] = a[i];
                }
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    dest[i] = hidden;
                }
            }
            if (b_visible) {
                const uint32_t* b = distance + tile + face + following;
#pragma GCC unroll 16
                for (uint32_t i = 0; i < col; ++i) {
                    dest[n + i] = b[i];
                }
            } else {
                for (uint32_t i = 0; i < col; ++i) {
                    dest[n + i] = hidden;
                }
            }
        }
    }
    asm volatile("" ::: "memory");
}

void kernel_main() {
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr auto meta_args = TensorAccessorArgs<5>();
    constexpr auto a0 = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(meta_args, addr(0), 32);
#ifdef PREFILL_WRITE
    const uint32_t metadata = get_write_ptr(6);
#else
    cb_reserve_back(7, 1);
    const uint32_t metadata = get_write_ptr(7);
#endif
    noc_async_read<32>(meta.get_noc_addr(0), metadata, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t start = words(metadata)[0] / 32;
    const uint32_t valid_tiles = (words(metadata)[1] + 31) / 32;
#ifndef PREFILL_WRITE
    cb_push_back(7, 1);
#endif

#ifdef PREFILL_WRITE
    // The writer builds the odd-lag bias tiles while the reader builds the even ones.
    // Token CBs order the shared distance scratch: 15 (reader -> writer) marks the
    // task's distances loaded and 17 (writer -> reader) releases them.
    const auto out = TensorAccessor(a0, addr(1), bf_bytes);
    const uint32_t distances = get_write_ptr(5);
#if PREFILL_QUERY_GROUP > 1
    const uint32_t groups = q_tiles / query_tiles_per_task;
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        const uint32_t qt = first_query(task, q_tiles);
        if (qt >= valid_tiles) {
            continue;
        }
        const uint32_t n = live_queries(qt, valid_tiles);
        const uint32_t first = start + qt;
        const uint32_t shared = shared_keys(first, saved_tiles);
        for (uint32_t i = 0; i < n; ++i) {
            cb_wait_front(15, 1);
            invalidate_l1_cache();
            const uint32_t last = first + i * query_step;
            for (uint32_t key = shared; key <= last; ++key) {
                const uint32_t lag = last - key;
                if (lag <= window_tiles && lag % 2 == 1) {
                    cb_reserve_back(14, 1);
                    make_bias(get_write_ptr(14), distances, lag);
                    cb_push_back(14, 1);
                }
            }
            cb_pop_front(15, 1);
            cb_reserve_back(17, 1);
            cb_push_back(17, 1);
        }
        for (uint32_t i = 0; i < n; ++i) {
            cb_wait_front(16, width_tiles);
            const uint32_t data = get_read_ptr(16);
            const uint32_t tile = (qt + i * query_step) * heads + (task / groups) * heads_per_task + i * head_step;
            for (uint32_t d = 0; d < width_tiles; ++d) {
                noc_async_write<bf_bytes>(data + d * bf_bytes, out.get_noc_addr(tile * width_tiles + d), bf_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(16, width_tiles);
        }
    }
#else
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        const uint32_t qt = query_tile(task);
        if (qt >= valid_tiles) {
            continue;
        }
        const uint32_t q_abs = start + qt;
        const uint32_t first = global || q_abs <= window_tiles ? 0 : q_abs - window_tiles;
        cb_wait_front(15, 1);
        invalidate_l1_cache();
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t key = first; key <= q_abs; ++key) {
                const uint32_t lag = q_abs - key;
                const bool need_score = pass == 0 || lag >= saved_tiles;
                if (lag <= window_tiles && need_score && lag % 2 == 1) {
                    cb_reserve_back(14, 1);
                    make_bias(get_write_ptr(14), distances, lag);
                    cb_push_back(14, 1);
                }
            }
        }
        cb_pop_front(15, 1);
        cb_reserve_back(17, 1);
        cb_push_back(17, 1);
        cb_wait_front(16, width_tiles);
        const uint32_t data = get_read_ptr(16);
        const uint32_t tile = qt * heads + task / q_tiles;
        for (uint32_t d = 0; d < width_tiles; ++d) {
            noc_async_write<bf_bytes>(data + d * bf_bytes, out.get_noc_addr(tile * width_tiles + d), bf_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(16, width_tiles);
    }
#endif
#else
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(a0, addr(1), bf_bytes);
    const auto k = TensorAccessor(a1, addr(2), bf_bytes);
    const auto v = TensorAccessor(a2, addr(3), bf_bytes);
    const auto old_k = TensorAccessor(a3, addr(4), bf_bytes);
    const auto old_v = TensorAccessor(a4, addr(5), bf_bytes);
    const auto sinks = TensorAccessor(a5, addr(6), bf_bytes);
#if PREFILL_DISTANCE
    // The compute kernel projects each task's relative query onto the table.
    const auto relative = TensorAccessor(a6, addr(7), bf_bytes);
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(a7, addr(8), bf_bytes);
    cb_reserve_back(19, window_tiles);
    for (uint32_t i = 0; i < window_tiles; ++i) {
        noc_async_read<bf_bytes>(table.get_noc_addr(i), get_write_ptr(19) + i * bf_bytes, bf_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(19, window_tiles);
#else
    const auto distance = TensorAccessor(a6, addr(7), fp_bytes);
#endif
    const uint32_t scratch = get_write_ptr(4);
    const uint32_t distances = get_write_ptr(5);
#if PREFILL_QUERY_GROUP > 1
    static_assert(global && saved_tiles > window_tiles);
    const uint32_t groups = q_tiles / query_tiles_per_task;
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        const uint32_t qt = first_query(task, q_tiles);
        if (qt >= valid_tiles) {
            continue;
        }
        const uint32_t n = live_queries(qt, valid_tiles);
        const uint32_t head = (task / groups) * heads_per_task;
        const uint32_t first = start + qt;
        const uint32_t shared = shared_keys(first, saved_tiles);
        const uint32_t kv_head = head / 3;
        auto issue = [&](bool value, uint32_t key, uint32_t dest) {
            if (key < start) {
                const uint32_t tile = kv_head * cache_tiles + key;
                if (value) {
                    issue_head_tile(old_v, tile, dest);
                } else {
                    issue_head_tile(old_k, tile, dest);
                }
            } else {
                const uint32_t tile = kv_head * q_tiles + key - start;
                if (value) {
                    issue_head_tile(v, tile, dest);
                } else {
                    issue_head_tile(k, tile, dest);
                }
            }
        };
        cb_reserve_back(0, group_size * width_tiles);
        cb_reserve_back(9, group_size);
        const uint32_t q_dest = get_write_ptr(0);
        const uint32_t sink_dest = get_write_ptr(9);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t slot = (head + i * head_step) * q_tiles + qt + i * query_step;
            issue_head_tile(q, slot, q_dest + i * width_tiles * bf_bytes);
            noc_async_read<bf_bytes>(sinks.get_noc_addr(qt + i * query_step), scratch, bf_bytes);
            noc_async_read_barrier();
            invalidate_l1_cache();
            const auto raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
            auto out = words(sink_dest + i * fp_bytes);
            for (uint32_t row = 0; row < 32; ++row) {
                out[face_index(row, 0)] = uint32_t(raw[face_index(row, head + i * head_step)]) << 16;
            }
        }
        cb_push_back(9, group_size);
        cb_push_back(0, group_size * width_tiles);
        for (uint32_t key = 0; key < shared; ++key) {
            cb_reserve_back(1, width_tiles);
            issue(false, key, get_write_ptr(1));
            noc_async_read_barrier();
            cb_push_back(1, width_tiles);
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t last = first + i * query_step;
            const uint32_t slot = (head + i * head_step) * q_tiles + qt + i * query_step;
#if PREFILL_DISTANCE
            cb_reserve_back(18, 1);
            noc_async_read<bf_bytes>(relative.get_noc_addr(slot), get_write_ptr(18), bf_bytes);
            noc_async_read_barrier();
            cb_push_back(18, 1);
            cb_wait_front(5, window_tiles);
#else
            const uint32_t distance_tiles = last < window_tiles ? last + 1 : window_tiles;
            for (uint32_t j = window_tiles - distance_tiles; j < window_tiles; ++j) {
                noc_async_read<fp_bytes>(distance.get_noc_addr(slot * window_tiles + j), distances + j * fp_bytes, fp_bytes);
            }
            noc_async_read_barrier();
#endif
            invalidate_l1_cache();
            cb_reserve_back(15, 1);
            cb_push_back(15, 1);
            for (uint32_t key = shared; key <= last; ++key) {
                cb_reserve_back(1, width_tiles);
                issue(false, key, get_write_ptr(1));
                const uint32_t lag = last - key;
                if (lag <= window_tiles && lag % 2 == 0) {
                    cb_reserve_back(3, 1);
                    make_bias(get_write_ptr(3), distances, lag);
                    cb_push_back(3, 1);
                }
                noc_async_read_barrier();
                cb_push_back(1, width_tiles);
            }
            cb_wait_front(17, 1);
            cb_pop_front(17, 1);
#if PREFILL_DISTANCE
            cb_pop_front(5, window_tiles);
#endif
        }
        auto second_pass = [&](uint32_t begin, uint32_t end, uint32_t keep) {
            for (uint32_t group = begin; group < end; group += value_group_keys) {
                const uint32_t count = end - group < value_group_keys ? end - group : value_group_keys;
                cb_reserve_back(2, value_group_keys * width_tiles);
                const uint32_t v_dest = get_write_ptr(2);
                for (uint32_t j = 0; j < count; ++j) {
                    const uint32_t key = group + j;
                    if (key < keep) {
                        cb_reserve_back(1, width_tiles);
                        issue(false, key, get_write_ptr(1));
                    }
                    issue(true, key, v_dest + j * width_tiles * bf_bytes);
                    noc_async_read_barrier();
                    if (key < keep) {
                        cb_push_back(1, width_tiles);
                    }
                }
                cb_push_back(2, value_group_keys * width_tiles);
            }
        };
        second_pass(0, shared, shared);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t count = first + i * query_step + 1;
            second_pass(shared, count, count > saved_tiles ? count - saved_tiles : 0);
        }
    }
#else
    bool shared_distances = false;
    bool key_cb_odd = false, value_cb_odd = false;
    for (uint32_t task = get_arg_val<uint32_t>(0); task < get_arg_val<uint32_t>(1); ++task) {
        const uint32_t qt = query_tile(task);
        const uint32_t head = task / q_tiles;
        const uint32_t slot = head * q_tiles + qt;
        const uint32_t q_abs = start + qt;
        const uint32_t first = global || q_abs <= window_tiles ? 0 : q_abs - window_tiles;
        const uint32_t count = qt < valid_tiles ? q_abs + 1 - first : 0;

        if (!count) {
            continue;
        }
        if (shared_distances) {
            cb_wait_front(17, 1);
            cb_pop_front(17, 1);
#if PREFILL_DISTANCE
            cb_pop_front(5, window_tiles);
#endif
        }
        shared_distances = true;
        read_head_tile(q, slot, 0);
        noc_async_read<bf_bytes>(sinks.get_noc_addr(qt), scratch, bf_bytes);
#if PREFILL_DISTANCE
        cb_reserve_back(18, 1);
        noc_async_read<bf_bytes>(relative.get_noc_addr(slot), get_write_ptr(18), bf_bytes);
        noc_async_read_barrier();
        cb_push_back(18, 1);
        cb_wait_front(5, window_tiles);
#else
        const uint32_t distance_tiles = q_abs < window_tiles ? q_abs + 1 : window_tiles;
        for (uint32_t i = window_tiles - distance_tiles; i < window_tiles; ++i) {
            noc_async_read<fp_bytes>(
                distance.get_noc_addr(slot * window_tiles + i), distances + i * fp_bytes, fp_bytes);
        }
#endif
        noc_async_read_barrier();
        invalidate_l1_cache();
        cb_reserve_back(15, 1);
        cb_push_back(15, 1);
        cb_push_back(0, width_tiles);
        cb_reserve_back(9, 1);
        auto sink = words(get_write_ptr(9));
        const auto raw_sink = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        for (uint32_t row = 0; row < 32; ++row) {
            sink[face_index(row, 0)] = uint32_t(raw_sink[face_index(row, head)]) << 16;
        }
        cb_push_back(9, 1);

        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t key = first; key <= q_abs; ++key) {
                const uint32_t lag = q_abs - key;
                const bool need_score = pass == 0 || lag >= saved_tiles;
                if constexpr (global && q_tiles == 64) {
                    // The value product waits for probabilities for a whole group of eight keys.
                    if (key + 1 < start && lag > window_tiles + 1 && (!need_score || !key_cb_odd) &&
                        (!pass || (!value_cb_odd && (key - first) % value_group_keys != value_group_keys - 1)) &&
                        (pass == 0 || lag > saved_tiles || !need_score)) {
                        constexpr uint32_t pair = 2 * width_tiles;
                        if (need_score) {
                            cb_reserve_back(1, pair);
                        }
                        if (pass) {
                            cb_reserve_back(2, pair);
                        }
                        const uint32_t tile = head / 3 * cache_tiles + key;
                        for (uint32_t i = 0; i < 2; ++i) {
                            const uint32_t offset = i * width_tiles * bf_bytes;
                            if (need_score) {
                                issue_head_tile(old_k, tile + i, get_write_ptr(1) + offset);
                            }
                            if (pass) {
                                issue_head_tile(old_v, tile + i, get_write_ptr(2) + offset);
                            }
                        }
                        noc_async_read_barrier();
                        if (need_score) {
                            cb_push_back(1, pair);
                        }
                        if (pass) {
                            cb_push_back(2, pair);
                        }
                        ++key;
                        continue;
                    }
                }
                if (key >= start) {
                    const uint32_t tile = head / 3 * q_tiles + key - start;
                    if (need_score) {
                        read_head_tile(k, tile, 1);
                    }
                    if (pass) {
                        read_head_tile(v, tile, 2);
                    }
                } else {
                    const uint32_t tile = head / 3 * cache_tiles + (global ? key : key % window_tiles);
                    if (need_score) {
                        read_head_tile(old_k, tile, 1);
                    }
                    if (pass) {
                        read_head_tile(old_v, tile, 2);
                    }
                }
                if (lag <= window_tiles && need_score && lag % 2 == 0) {
                    cb_reserve_back(3, 1);
                    make_bias(get_write_ptr(3), distances, lag);
                    cb_push_back(3, 1);
                }
                noc_async_read_barrier();
                if (need_score) {
                    cb_push_back(1, width_tiles);
                    key_cb_odd = !key_cb_odd;
                }
                if (pass) {
                    cb_push_back(2, width_tiles);
                    value_cb_odd = !value_cb_odd;
                }
            }
        }
    }
#endif
#endif
}
