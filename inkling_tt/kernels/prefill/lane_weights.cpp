// Stream one four-column expert slice per lane expert into (k, column) order.
// A short transaction-ID window bounds the reads in flight per worker; with
// every worker streaming at once, deep queues lower total DRAM throughput.
#include "api/dataflow/dataflow_api.h"
#include "lane.h"

namespace {
constexpr uint32_t tile_bytes = 1088;
// One 13 KB chunk in flight per worker measured fastest with all workers streaming.
constexpr uint32_t window = 1;
constexpr uint32_t ring = window + 1;
static_assert(window >= 1 && ring <= 15);

struct Stream {
    uint32_t vc;
    uint32_t chunk = 0;

    // Opens a chunk of reads from the bank selected by the last set_state.
    uint32_t open() {
        if (chunk >= window) {
            noc_async_read_barrier_with_trid((chunk - window) % ring + 1);
        }
        const uint32_t trid = chunk % ring + 1;
        noc_async_read_set_trid(trid);
        ++chunk;
        return trid;
    }

    void drain() {
        const uint32_t first = chunk > window ? chunk - window : 0;
        for (uint32_t c = first; c < chunk; ++c) {
            noc_async_read_barrier_with_trid(c % ring + 1);
        }
        chunk = 0;
    }
};
}  // namespace

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr uint32_t weight_tiles = k * cols;
    constexpr uint32_t packet_tiles = 12;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;
    static_assert(cols == 4);
    const auto ends_acc = TensorAccessor(TensorAccessorArgs<3>(), get_common_arg_val<uint32_t>(0), 512);
    const uint32_t weights = get_common_arg_val<uint32_t>(1);
    cb_reserve_back(2, 1);
    const uint32_t ends_l1 = get_write_ptr(2);
    noc_async_read<512>(ends_acc.get_noc_addr(0), ends_l1, 512);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ends_l1);
    cb_push_back(2, 1);
    const auto ends = [&](uint32_t e) { return offsets[e]; };
    const uint32_t group = get_arg_val<uint32_t>(lane_args::group);
    const LaneRange range = lane_range(ends, get_arg_val<uint32_t>(lane_args::lane),
                                       get_arg_val<uint32_t>(lane_args::lanes));
    Stream stream{get_arg_val<uint32_t>(lane_args::vc)};
    uint64_t banks[4];
    for (uint32_t i = 0; i < 4; ++i) {
        banks[i] = get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(lane_args::banks + i), weights);
    }
    auto bank_state = [&](uint64_t bank, uint32_t bytes) {
        noc_async_read_one_packet_set_state<true>(bank, bytes, stream.vc);
    };
#if EXPERTS_GATE
    const bool shared = group >= 16;
#endif
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t expert = 0; expert < 128; ++expert) {
        uint32_t begin, end;
        if (!lane_expert(ends, range, expert, begin, end)) {
            continue;
        }
        cb_reserve_back(1, weight_tiles);
        const uint32_t dest = get_write_ptr(1);
#if EXPERTS_GATE
        const uint32_t base = expert * k * 9 * tile_bytes;
        if (!shared) {
            // Bank-local 64-channel group: [gate0 gate1 up0 up1] per K row.
            const uint32_t src = base + (group % 2) * weight_tiles * tile_bytes;
            bank_state(banks[0], packet_bytes);
            for (uint32_t p = 0; p < weight_tiles * tile_bytes; p += packet_bytes) {
                const uint32_t trid = stream.open();
                noc_async_read_one_packet_with_state_with_trid(
                    static_cast<uint32_t>(banks[0]), src + p, dest + p, trid);
            }
        } else {
            // Two 32-channel groups, each split into alternating K blocks over a bank pair.
            // Column order [gate_a gate_b up_a up_b] matches the local groups.
            for (uint32_t q = 0; q < 2; ++q) {
                for (uint32_t kb = 0; kb < k / 6; ++kb) {
                    const uint64_t bank = banks[2 * q + kb % 2];
                    const uint32_t src = base + 8 * k * tile_bytes + (kb / 2) * 12 * tile_bytes;
                    bank_state(bank, tile_bytes);
                    const uint32_t trid = stream.open();
                    for (uint32_t s = 0; s < 6; ++s) {
                        for (uint32_t c = 0; c < 2; ++c) {
                            noc_async_read_one_packet_with_state_with_trid(
                                static_cast<uint32_t>(bank), src + (s * 2 + c) * tile_bytes,
                                dest + ((kb * 6 + s) * 4 + c * 2 + q) * tile_bytes, trid);
                        }
                    }
                }
            }
        }
#else
        // Output columns 4g..4g+3 live in banks n % 8 at column slot n / 8.
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t n = group * 4 + i;
            const uint32_t src = (expert * 9 + n / 8) * k * tile_bytes;
            bank_state(banks[i], tile_bytes);
            uint32_t trid = 0;
            for (uint32_t kk = 0; kk < k; ++kk) {
                if (kk % packet_tiles == 0) {
                    trid = stream.open();
                }
                noc_async_read_one_packet_with_state_with_trid(
                    static_cast<uint32_t>(banks[i]), src + kk * tile_bytes, dest + (kk * 4 + i) * tile_bytes, trid);
            }
        }
#endif
        stream.drain();
        cb_push_back(1, weight_tiles);
    }
    // Dedicated NoC command state survives launches in the same mode.
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);
    noc_async_read_set_trid(0);
}
