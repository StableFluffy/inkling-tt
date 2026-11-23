#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t block_k = get_compile_time_arg_val(1);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 1088;
    constexpr uint32_t block_tiles = block_k * cols;
    constexpr uint32_t block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t packet_bytes = 12 * tile_bytes;
    constexpr uint32_t blocks_per_expert = k / block_k;
    constexpr uint32_t ring = EXPERTS_VERIFY_RING;
    constexpr uint32_t ahead = ring - 2;
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
    const uint32_t vc = get_arg_val<uint32_t>(4);
    constexpr bool shared = cols == 2;
    const uint32_t group = (shared ? k * 8 : get_arg_val<uint32_t>(5) * k * cols) * tile_bytes;
    constexpr uint32_t lanes = EXPERTS_GU_LANES;
#else
    const uint64_t bank = get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(0), weights);
    const uint32_t col = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    constexpr uint32_t lane = 0;
    constexpr uint32_t lanes = 1;
#endif
    uint8_t active[128];
    uint32_t count = 0;
    for (uint32_t expert = lane; expert < 128; expert += lanes) {
        if (offsets[expert] != (expert == 0 ? 0 : offsets[expert - 1])) {
            active[count++] = expert;
        }
    }
    if (!count) {
        return;
    }
#if !EXPERTS_GATE
    noc_async_read_one_packet_set_state<true>(bank, packet_bytes, vc);
#endif

    const uint32_t blocks = count * blocks_per_expert;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(1, (ahead + 1) * block_tiles);
    const uint32_t base = get_write_ptr(1);
    auto ticket = [](uint32_t block) { return block % ring + 1; };
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t expert = active[block / blocks_per_expert];
        const uint32_t kb = block % blocks_per_expert;
#if EXPERTS_GATE
        const uint64_t bank = banks[kb % 2];
        noc_async_read_one_packet_set_state<true>(bank, packet_bytes, vc);
        const uint32_t src = expert * k * 9 * tile_bytes + group + (shared ? kb / 2 : kb) * block_bytes;
#else
        const uint32_t src = ((expert * 9 + col) * blocks_per_expert + kb) * block_bytes;
#endif
        const uint32_t dest = base + (block % ring) * block_bytes;
        const uint32_t trid = ticket(block);
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < block_bytes; p += packet_bytes) {
            noc_async_read_one_packet_with_state_with_trid(bank, src + p, dest + p, trid);
        }
        if (block >= ahead) {
            noc_async_read_barrier_with_trid(ticket(block - ahead));
            cb_push_back(1, block_tiles);
            if (block + 1 < blocks) {
                cb_reserve_back(1, (ahead + 1) * block_tiles);
            }
        }
    }
    for (uint32_t block = blocks - ahead; block < blocks; ++block) {
        noc_async_read_barrier_with_trid(ticket(block));
        cb_push_back(1, block_tiles);
    }
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);
    noc_async_read_set_trid(0);
}
