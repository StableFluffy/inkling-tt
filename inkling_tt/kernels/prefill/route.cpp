// Select routes and build the packed expert workspace's maps.
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t experts = 128;
constexpr uint32_t topk = 8;
constexpr uint32_t routes_per_tile = 32 * topk;
constexpr uint32_t tile_bytes = 4096;
constexpr uint32_t ids_bytes = routes_per_tile * sizeof(uint32_t);

uint32_t face_index(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}

void kernel_main() {
    const auto addr = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr auto args0 = TensorAccessorArgs<3>();
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t input = get_write_ptr(1);
#if PREFILL_SELECT
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    constexpr auto args4 = TensorAccessorArgs<args3.next_compile_time_args_offset()>();
    const auto scores = TensorAccessor(args0, addr(0), tile_bytes);
    const auto ranked = TensorAccessor(args1, addr(1), tile_bytes);
    const auto meta = TensorAccessor(args2, addr(2), 32);
    const auto ids = TensorAccessor(args3, addr(3), ids_bytes);
    const auto gamma = TensorAccessor(args4, addr(4), tile_bytes);
    noc_async_read<32>(meta.get_noc_addr(0), scratch, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[1];
    auto selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto weights = reinterpret_cast<volatile tt_l1_ptr float*>(input + 8 * tile_bytes);

    for (uint32_t block = get_arg_val<uint32_t>(0); block < get_arg_val<uint32_t>(1); ++block) {
        if (32 * block < valid) {
            for (uint32_t c = 0; c < 4; ++c) {
                noc_async_read<tile_bytes>(scores.get_noc_addr(4 * block + c), input + c * tile_bytes, tile_bytes);
                noc_async_read<tile_bytes>(ranked.get_noc_addr(4 * block + c), input + (4 + c) * tile_bytes, tile_bytes);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
        }
        const auto raw = reinterpret_cast<volatile tt_l1_ptr float*>(input);
        const auto order = reinterpret_cast<volatile tt_l1_ptr int32_t*>(input + 4 * tile_bytes);
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t winners[topk] = {};
            float values[topk] = {};
            if (block * 32 + r < valid) {
                int32_t ranks[topk];
                for (auto& rank : ranks) {
                    rank = static_cast<int32_t>(0x80000000u);
                }
                for (uint32_t e = 0; e < experts; ++e) {
                    const uint32_t i = e / 32 * 1024 + face_index(r, e % 32);
                    const int32_t bits = order[i];
                    // Monotonic signed integer ordering of finite IEEE floats.
                    const int32_t rank = bits < 0 ? bits ^ 0x7fffffff : bits;
                    if (rank <= ranks[topk - 1]) {
                        continue;
                    }
                    uint32_t s = topk - 1;
                    while (s && rank > ranks[s - 1]) {
                        ranks[s] = ranks[s - 1];
                        winners[s] = winners[s - 1];
                        values[s] = values[s - 1];
                        --s;
                    }
                    ranks[s] = rank;
                    winners[s] = e;
                    values[s] = raw[i];
                }
                float sum = 0.0f;
                for (float value : values) {
                    sum += value;
                }
                const float denominator = sum + 1e-20f;
                for (uint32_t s = 1; s < topk; ++s) {
                    uint32_t e = winners[s];
                    float v = values[s];
                    uint32_t k = s;
                    while (k && e < winners[k - 1]) {
                        winners[k] = winners[k - 1];
                        values[k] = values[k - 1];
                        --k;
                    }
                    winners[k] = e;
                    values[k] = v;
                }
                for (uint32_t s = 0; s < topk; ++s) {
                    values[s] = (values[s] / denominator) * 8.0f;
                }
            } else {
                for (auto& winner : winners) {
                    winner = experts;
                }
            }
            for (uint32_t s = 0; s < topk; ++s) {
                selected[r * topk + s] = winners[s];
                const auto left = s * 1024 + face_index(r, 0);
                for (uint32_t col = 0; col < 16; ++col) {
                    weights[left + col] = values[s];
                    weights[left + 256 + col] = values[s];
                }
            }
        }
        noc_async_write<ids_bytes>(scratch, ids.get_noc_addr(block), ids_bytes);
        for (uint32_t s = 0; s < topk; ++s) {
            noc_async_write<tile_bytes>(input + (8 + s) * tile_bytes, gamma.get_noc_addr(block * topk + s), tile_bytes);
        }
        noc_async_write_barrier();
    }
#else
    constexpr uint32_t token_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t gather_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(args0, addr(0), ids_bytes);
    const auto ends = TensorAccessor(args1, addr(1), experts * 4);
    const auto gather = TensorAccessor(args2, addr(2), gather_bytes);
    const auto inverse = TensorAccessor(args3, addr(3), ids_bytes);
    auto routes = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input);
    auto forward = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t ends_l1 = scratch + gather_bytes;
    auto limits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ends_l1);

    for (uint32_t t = 0; t < token_tiles; ++t) {
        noc_async_read<ids_bytes>(ids.get_noc_addr(t), input + t * ids_bytes, ids_bytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

    uint32_t count[experts + 1] = {};
    for (uint32_t i = 0; i < token_tiles * routes_per_tile; ++i) {
        ++count[routes[i]];
    }
    uint32_t next[experts];
    uint32_t end = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        next[e] = end;
        const uint32_t live_end = end + count[e];
        end = (live_end + rows - 1) & ~(rows - 1);
        for (uint32_t pad = live_end; pad < end; ++pad) {
            forward[pad] = 0;
        }
        limits[e] = end / rows;
    }
    for (uint32_t pad = end; pad < gather_bytes / 4; ++pad) {
        forward[pad] = 0;
    }
    for (uint32_t i = 0; i < token_tiles * routes_per_tile; ++i) {
        const uint32_t e = routes[i];
        if (e == experts) {
            routes[i] = 0;
        } else {
            const uint32_t row = next[e]++;
            forward[row] = i / topk + 1;
            routes[i] = row + 1;
        }
    }
    noc_async_write(scratch, gather.get_noc_addr(0), gather_bytes);
    noc_async_write<512>(ends_l1, ends.get_noc_addr(0), 512);
    for (uint32_t t = 0; t < token_tiles; ++t) {
        noc_async_write<ids_bytes>(input + t * ids_bytes, inverse.get_noc_addr(t), ids_bytes);
    }
    noc_async_write_barrier();
#endif
}
