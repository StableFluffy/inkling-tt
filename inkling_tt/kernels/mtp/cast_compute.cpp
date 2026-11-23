#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

using namespace ckernel;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

void round_dest(uint32_t, uint32_t, uint32_t) {
    for (uint32_t i = 0; i < 32; ++i) {
        dst_reg[i] = ::float32_to_bf16_rne(sfpi::vFloat(dst_reg[i]));
    }
}
#endif

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 0, 16);
    copy_init(0);
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
        cb_wait_front(0, 1);
        cb_reserve_back(16, 1);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        MATH((_llk_math_eltwise_binary_sfpu_params_(round_dest, 0, 0, 0, VectorMode::None)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_pop_front(0, 1);
        cb_push_back(16, 1);
    }
}
