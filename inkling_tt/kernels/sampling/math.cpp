#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void load(uint32_t cb, uint32_t tile, uint32_t slot) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, slot);
}

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 16);
    cb_wait_front(2, 1);
    const uint32_t inverse_temperature = read_tile_value(2, 0, 0);
    cb_pop_front(2, 1);
    cb_wait_front(0, tiles);
    cb_wait_front(1, tiles);
    cb_reserve_back(16, tiles);
    for (uint32_t i = 0; i < tiles; ++i) {
        tile_regs_acquire();
        load(0, i, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x3d088889);
        tanh_tile_init<false>();
        tanh_tile<false>(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x41f00000);
        load(1, i, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inverse_temperature);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(16);
        pack_tile<true>(0, 16, i);
        tile_regs_release();
    }
    cb_pop_front(0, tiles);
    cb_pop_front(1, tiles);
    cb_push_back(16, tiles);
    cb_wait_front(3, 1);
    const uint32_t maximum = read_tile_value(3, 0, 0);
    cb_pop_front(3, 1);
    cb_wait_front(16, tiles);
    cb_reserve_back(17, tiles);
    for (uint32_t i = 0; i < tiles; ++i) {
        tile_regs_acquire();
        load(16, i, 0);
        binop_with_scalar_tile_init();
        sub_unary_tile(0, maximum);
        exp_tile_init<false>();
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(17);
        pack_tile<true>(0, 17, i);
        tile_regs_release();
    }
    cb_push_back(17, tiles);
}
