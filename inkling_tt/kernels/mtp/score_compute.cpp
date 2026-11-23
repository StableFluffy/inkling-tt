#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_softcap.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

void score(uint32_t, uint32_t, uint32_t) {
    using namespace sfpi;
    for (uint32_t vector = 0; vector < 32; ++vector) {
        vFloat correction = dst_reg[vector];
        vFloat bias = dst_reg[32 + vector];
        vFloat base = dst_reg[64 + vector];
        vFloat biased = ::float32_to_bf16_rne(correction + bias);
        vFloat summed = ::float32_to_bf16_rne(biased + base);
        vFloat capped = ckernel::sfpu::_sfpu_softcap_(summed, vFloat(30.0f), vFloat(1.0f / 30.0f));
        dst_reg[vector] = ::float32_to_bf16_rne(capped);
    }
}
#endif

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 0, 16);
    MATH((ckernel::sfpu::softcap_init()));
    for (uint32_t tile = get_arg_val<uint32_t>(0); tile < get_arg_val<uint32_t>(1); ++tile) {
        cb_reserve_back(16, 1);
        tile_regs_acquire();
        for (uint32_t cb = 0; cb < 3; ++cb) {
            cb_wait_front(cb, 1);
            reconfig_data_format_srca(cb);
            copy_init(cb);
            copy_tile(cb, 0, cb);
            cb_pop_front(cb, 1);
        }
        MATH((_llk_math_eltwise_binary_sfpu_params_(score, 0, 0, 0, VectorMode::None)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
