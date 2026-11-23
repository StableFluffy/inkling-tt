#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace {
using namespace sfpi;

void weighted(uint32_t, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < 32; ++v) {
        dst_reg[32 + v] = vFloat(dst_reg[32 + v]) * vFloat(dst_reg[64 + v]);
    }
}

void accumulate(uint32_t slot, uint32_t, uint32_t) {
    for (uint32_t v = 0; v < 32; ++v) {
        vFloat sum = dst_reg[32 + v];
        if (slot != 0) {
            sum += dst_reg[v];
        }
        dst_reg[v] = slot == 7 ? ::float32_to_bf16_rne(sum) : sum;
    }
}
}  // namespace
#endif

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0, 16);
    for (uint32_t i = start; i < end; ++i) {
        if (i == start || i % 72 == 0) {
            cb_wait_front(1, 8);
        }
        tile_regs_acquire();
        for (uint32_t slot = 0; slot < 8; ++slot) {
            cb_wait_front(0, 1);
            reconfig_data_format_srca(0);
            copy_init(0);
            copy_tile(0, 0, 1);
            cb_pop_front(0, 1);
            reconfig_data_format_srca(1);
            copy_init(1);
            copy_tile(1, slot, 2);
            MATH((_llk_math_eltwise_binary_sfpu_params_(weighted, 0, 0, 0, VectorMode::None)));
            MATH((_llk_math_eltwise_binary_sfpu_params_(accumulate, slot, 0, 0, VectorMode::None)));
        }
        if (i + 1 == end || (i + 1) % 72 == 0) {
            cb_pop_front(1, 8);
        }
        tile_regs_commit();
        cb_reserve_back(16, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
