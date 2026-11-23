// Read one complete bank-local expert slice per selected expert. Keep it
// in the CB until all activation tiles for that expert have consumed it.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t block_k = get_compile_time_arg_val(1);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 1088;
    constexpr uint32_t block_bytes = block_k * cols * tile_bytes;
    constexpr uint32_t packet_bytes = 12 * tile_bytes;
    constexpr uint32_t weight_tiles = k * cols;
    static_assert(block_bytes % packet_bytes == 0);
    const auto ends = TensorAccessor(TensorAccessorArgs<3>(), get_common_arg_val<uint32_t>(0), 512);
    const uint32_t weights = get_common_arg_val<uint32_t>(1);
    cb_reserve_back(2, 1);
    const uint32_t ends_l1 = get_write_ptr(2);
    noc_async_read<512>(ends.get_noc_addr(0), ends_l1, 512);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const auto offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ends_l1);
    cb_push_back(2, 1);

#if EXPERTS_GATE
    const uint64_t banks[2] = {
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(1), weights),
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(2), weights),
    };
    const uint32_t lane = get_arg_val<uint32_t>(3);
    constexpr bool shared = cols == 2;
    const uint32_t group = (shared ? k * 8 : get_arg_val<uint32_t>(5) * weight_tiles) * tile_bytes;
    constexpr uint32_t lanes = EXPERTS_GU_LANES;
#else
    const uint64_t bank = get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(0), weights);
    const uint32_t col = get_arg_val<uint32_t>(1);
    constexpr uint32_t lane = 0;
    constexpr uint32_t lanes = 1;
#endif
    const auto read_packet = [](uint64_t source, uint32_t target) {
        noc_async_read<packet_bytes>(source, target, packet_bytes);
        noc_async_read_barrier();
    };
    for (uint32_t expert = lane; expert < 128; expert += lanes) {
        if (offsets[expert] == (expert == 0 ? 0 : offsets[expert - 1])) {
            continue;
        }
        cb_reserve_back(1, weight_tiles);
        const uint32_t dest = get_write_ptr(1);
#if EXPERTS_GATE
        for (uint32_t block = 0; block < k / block_k; ++block) {
            const uint64_t bank = banks[block % 2];
            const uint32_t src = expert * k * 9 * tile_bytes + group +
                                 (shared ? block / 2 : block) * block_bytes;
            for (uint32_t p = 0; p < block_bytes; p += packet_bytes) {
                read_packet(bank + src + p, dest + block * block_bytes + p);
            }
        }
#else
        const uint32_t src = (expert * 9 + col) * weight_tiles * tile_bytes;
        for (uint32_t p = 0; p < weight_tiles * tile_bytes; p += packet_bytes) {
            read_packet(bank + src + p, dest + p);
        }
#endif
        cb_push_back(1, weight_tiles);
    }
}
