// Per-row, per-head L2 clip or RMS scale in FP32.
// Global queries are then rounded and multiplied by the row's position scale.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

namespace {
constexpr uint32_t one = 0x3f800000;

#ifdef TRISC_MATH
void round_bf16(uint32_t tile, uint32_t, uint32_t) {
    for (uint32_t i = 0; i < 32; ++i) {
        const auto lane = tile * 32 + i;
        sfpi::dst_reg[lane] = ::float32_to_bf16_rne(sfpi::vFloat(sfpi::dst_reg[lane]));
    }
}
#endif

inline void bf16_rne(uint32_t tile) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(round_bf16, tile, 0, 0, VectorMode::None)));
}

// The SFPU row reduction folds each 4-row group of a face pair in its own
// order. The TTNN head layout puts head h at tile row h, so the reference
// order for a head is that of row group h % 16 / 4. Reduce copies with the
// groups rotated and take each group's sum from the pass that put it there.
#ifdef TRISC_MATH
void rotate_groups(uint32_t from, uint32_t to, uint32_t shift) {
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t e = 0; e < 2; ++e) {
                sfpi::dst_reg[to * 32 + f * 8 + (k + shift) % 4 * 2 + e] =
                    sfpi::vFloat(sfpi::dst_reg[from * 32 + f * 8 + k * 2 + e]);
            }
        }
    }
}

void gather_groups(uint32_t to, uint32_t target, uint32_t) {
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t k = 0; k < 4; ++k) {
            if (k == target) {
                continue;
            }
            const uint32_t pass = (target + 4 - k) % 4;
            for (uint32_t e = 0; e < 2; ++e) {
                sfpi::dst_reg[to * 32 + f * 8 + k * 2 + e] =
                    sfpi::vFloat(sfpi::dst_reg[(to + pass) * 32 + f * 8 + target * 2 + e]);
            }
        }
    }
}
#endif
}  // namespace

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(2);
    constexpr bool scaled = get_compile_time_arg_val(3);
    constexpr uint32_t head_tiles = (get_compile_time_arg_val(4) + 31) / 32;
    constexpr bool rms = get_compile_time_arg_val(5);
    constexpr uint32_t threshold = get_compile_time_arg_val(6);
    constexpr uint32_t epsilon = get_compile_time_arg_val(7);
    constexpr uint32_t inv_width = get_compile_time_arg_val(8);
    constexpr uint32_t sums = head_tiles;
    constexpr uint32_t scratch = sums + 1;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t end = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0, 16);
    for (uint32_t task = start; task < end; ++task) {
        cb_wait_front(0, head_tiles);
        tile_regs_acquire();
        reconfig_data_format_srca(0);
        copy_init(0);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            copy_tile(0, d, d);
        }
        cb_pop_front(0, head_tiles);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, sums);
        add_binary_tile_init();
        for (uint32_t d = 1; d < head_tiles; ++d) {
            mul_binary_tile_init();
            mul_binary_tile(d, d, scratch);
            add_binary_tile_init();
            add_binary_tile(sums, scratch, sums);
        }
        const uint32_t target = task / row_tiles % 16 / 4;
        for (uint32_t shift = 1; shift < 4; ++shift) {
            MATH((_llk_math_eltwise_binary_sfpu_params_(rotate_groups, sums, sums + shift, shift, VectorMode::None)));
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        for (uint32_t pass = 0; pass < 4; ++pass) {
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(sums + pass);
        }
        MATH((_llk_math_eltwise_binary_sfpu_params_(gather_groups, sums, target, 0, VectorMode::None)));
        binop_with_scalar_tile_init();
        if constexpr (rms) {
            mul_unary_tile(sums, inv_width, VectorMode::C);
            add_unary_tile(sums, epsilon, VectorMode::C);
            rsqrt_tile_init();
            rsqrt_tile(sums, VectorMode::C);
        } else {
            sqrt_tile_init();
            sqrt_tile(sums);
            binop_with_scalar_tile_init();
            add_unary_tile(sums, epsilon, VectorMode::C);
            recip_tile_init();
            recip_tile(sums, VectorMode::C);
            binop_with_scalar_tile_init();
            mul_unary_tile(sums, threshold, VectorMode::C);
            fill_tile_init();
            fill_tile_bitcast(scratch, one);
            binary_min_tile_init();
            binary_min_tile(sums, scratch, sums, VectorMode::C);
        }
        sfpu_mul_bcast_col_init();
        for (uint32_t d = 0; d < head_tiles; ++d) {
            sfpu_mul_bcast_col(d, sums);
            bf16_rne(d);
        }
        if constexpr (scaled) {
            cb_wait_front(1, 1);
            reconfig_data_format_srca(1);
            copy_init(1);
            copy_tile(1, 0, scratch);
            cb_pop_front(1, 1);
            sfpu_mul_bcast_col_init();
            for (uint32_t d = 0; d < head_tiles; ++d) {
                sfpu_mul_bcast_col(d, scratch);
                bf16_rne(d);
            }
        }
        tile_regs_commit();
        cb_reserve_back(16, head_tiles);
        tile_regs_wait();
        for (uint32_t d = 0; d < head_tiles; ++d) {
            pack_tile(d, 16);
        }
        tile_regs_release();
        cb_push_back(16, head_tiles);
    }
}
