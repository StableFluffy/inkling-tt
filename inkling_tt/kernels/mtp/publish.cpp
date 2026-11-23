#include "api/dataflow/dataflow_api.h"

#ifdef DRAFT_POLICY
uint32_t next_width(volatile tt_l1_ptr uint32_t* policy, uint32_t position, uint32_t count, uint32_t width) {
    constexpr uint32_t scale = 4096;
    const uint32_t accepted = (9 * policy[11] + 5) / 10 + (count - 1) * scale;
    const uint32_t rejected = (9 * policy[12] + 5) / 10 + (count < width ? scale : 0);
    policy[11] = accepted;
    policy[12] = rejected;
    const uint32_t acceptance = accepted * scale / (accepted + rejected);
    const uint32_t costs = position <= policy[2] ? 7 : 3;
    uint32_t best = 1;
    uint32_t best_tokens = scale;
    uint32_t tokens = scale;
    uint32_t power = scale;
    for (uint32_t rows = 2; rows <= policy[1] && position + rows <= policy[0]; ++rows) {
        power = power * acceptance / scale;
        tokens += power;
        if (uint64_t(tokens) * policy[costs + best - 1] > uint64_t(best_tokens) * policy[costs + rows - 1]) {
            best = rows;
            best_tokens = tokens;
        }
    }
    return best;
}
#endif

void kernel_main() {
    constexpr auto result_args = TensorAccessorArgs<0>();
    constexpr auto pos_args = TensorAccessorArgs<result_args.next_compile_time_args_offset()>();
    constexpr auto id_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto result = TensorAccessor(result_args, get_common_arg_val<uint32_t>(0), 64);
    const auto pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(1), 4);
    const auto id = TensorAccessor(id_args, get_common_arg_val<uint32_t>(2), 4);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read<64>(result.get_noc_addr(0), scratch, 64);
#ifdef DRAFT_POLICY
    constexpr auto policy_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(policy_args, get_common_arg_val<uint32_t>(3), 64);
    const uint32_t state_addr = scratch + 192;
    noc_async_read<64>(state.get_noc_addr(0), state_addr, 64);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t count = values[0];
    const uint32_t end = values[1];
    values[16] = values[count + 1];
    values[32] = end;
#ifdef DRAFT_POLICY
    auto policy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_addr);
    if (policy[0]) {
        values[15] = next_width(policy, end, count, values[14]);
        noc_async_write<64>(scratch, result.get_noc_addr(0), 64);
        noc_async_write<64>(state_addr, state.get_noc_addr(0), 64);
    }
#endif
    noc_async_write<4>(scratch + 64, id.get_noc_addr(0), 4);
    noc_async_write<4>(scratch + 128, pos.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
