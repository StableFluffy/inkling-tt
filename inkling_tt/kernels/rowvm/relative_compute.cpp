#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace {
void round_bf16(uint32_t, uint32_t, uint32_t) {
    using namespace sfpi;
    for (uint32_t i = 0; i < 32; ++i) {
        dst_reg[i] = ::float32_to_bf16_rne(vFloat(dst_reg[i]));
    }
}
}  // namespace
#endif

void kernel_main() {
    constexpr auto q = tt::CBIndex::c_0;
    constexpr auto rows = tt::CBIndex::c_1;
    constexpr auto mask = tt::CBIndex::c_2;
    constexpr auto table = tt::CBIndex::c_3;
    constexpr auto out = tt::CBIndex::c_16;
    constexpr bool bf16_output = get_compile_time_arg_val(0) == 2;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    CircularBuffer q_buf(q), rows_buf(rows), mask_buf(mask), table_buf(table), out_buf(out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(q, table, out);
    q_buf.wait_front(1);
    for (uint32_t tile = first; tile < end; ++tile) {
        rows_buf.wait_front(1);
        table_buf.reserve_back(1);
        reconfig_data_format_srca(rows);
        pack_reconfig_data_format(table);
        tilize_init(rows, 1, table);
        tilize_block(rows, 1, table);
        tilize_uninit(rows, table);
        table_buf.push_back(1);
        rows_buf.pop_front(1);

        table_buf.wait_front(1);
        mask_buf.wait_front(1);
        out_buf.reserve_back(1);
        tile_regs_acquire();
        matmul_init(q, table, /*transpose=*/1);
        matmul_tiles(q, table, 0, 0, 0);
        reconfig_data_format_srca(mask);
        copy_init(mask);
        copy_tile(mask, 0, 1);
        sfpu_add_bcast_row_init();
        sfpu_add_bcast_row(0, 1);
        if constexpr (bf16_output) {
            // Match the nearest-even BF16 result of TTNN's FP32 add.
            MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, 0, 0, 0, VectorMode::None)));
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(out);
        pack_tile(0, out);
        tile_regs_release();
        out_buf.push_back(1);
        mask_buf.pop_front(1);
        table_buf.pop_front(1);
    }
    q_buf.pop_front(1);
}
