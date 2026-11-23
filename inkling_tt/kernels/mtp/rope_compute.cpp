#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

void round_bf16(uint32_t tile, uint32_t, uint32_t) {
    for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t i = tile * 32 + lane;
        dst_reg[i] = ::float32_to_bf16_rne(sfpi::vFloat(dst_reg[i]));
    }
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    for (uint32_t pair = get_arg_val<uint32_t>(0); pair < get_arg_val<uint32_t>(1); ++pair) {
        cb_wait_front(0, 2);
        cb_wait_front(1, 2);
        cb_reserve_back(16, 2);
        tile_regs_acquire();
        reconfig_data_format_srca(0);
        copy_init(0);
        copy_tile(0, 0, 0);
        copy_tile(0, 1, 1);
        reconfig_data_format_srca(1);
        copy_init(1);
        copy_tile(1, 0, 2);
        copy_tile(1, 1, 3);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 4);
        mul_binary_tile(1, 3, 5);
        mul_binary_tile(1, 2, 1);
        mul_binary_tile(0, 3, 0);
        sub_binary_tile_init();
        sub_binary_tile(4, 5, 4);
        add_binary_tile_init();
        add_binary_tile(1, 0, 1);
        MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, 4, 0, 0, VectorMode::None)));
        MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, 1, 0, 0, VectorMode::None)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, 16);
        pack_tile(1, 16);
        tile_regs_release();
        cb_pop_front(0, 2);
        cb_pop_front(1, 2);
        cb_push_back(16, 2);
    }
}
