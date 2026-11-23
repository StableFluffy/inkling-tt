#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto result_args = TensorAccessorArgs<0>();
    constexpr auto pos_args = TensorAccessorArgs<result_args.next_compile_time_args_offset()>();
    constexpr auto id_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto result = TensorAccessor(result_args, get_common_arg_val<uint32_t>(0), 64);
    const auto pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(1), 4);
    const auto id = TensorAccessor(id_args, get_common_arg_val<uint32_t>(2), 4);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read<64>(result.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    auto values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t count = values[0];
    values[16] = values[1];
    values[0] = values[count + 1];
    noc_async_write<4>(scratch, id.get_noc_addr(0), 4);
    noc_async_write<4>(scratch + 64, pos.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
