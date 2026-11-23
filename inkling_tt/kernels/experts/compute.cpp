#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#ifdef EXPERTS_VERIFY_COMBINE
#include "api/compute/tile_move_copy.h"
#endif

using namespace ckernel;

constexpr uint32_t columns = get_compile_time_arg_val(2);
constexpr uint32_t pairs = columns / 2;

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_tanh.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_sigmoid.h"

namespace {
using namespace sfpi;

// In FP32 DST, each face's first four rows use two vectors (even/odd).
// The logical B1 row lives in the first two faces, eight vectors apart.
constexpr uint32_t vectors_in_tile = 32;
constexpr uint32_t second_face = 8;

template <typename F>
void face_rows(F fn) {
#pragma GCC unroll 2
    for (uint32_t face = 0; face <= second_face; face += second_face) {
#pragma GCC unroll 2
        for (uint32_t parity = 0; parity < 2; ++parity) {
            fn(face + parity);
        }
    }
}

// Match the Blackhole sparse matmul's BF16 projection writeback.
sfpi_inline vFloat round_projection(vFloat value) {
    return sfpi::convert<sfpi::vFloat16b>(value, sfpi::RoundMode::Nearest);
}

void gate_sigmoid(uint32_t tile, uint32_t, uint32_t) {
    const uint32_t base = tile * vectors_in_tile;
    face_rows([=](uint32_t row) {
        vFloat gate = round_projection(dst_reg[base + row]);
        dst_reg[base + row] = gate;
        dst_reg[6 * vectors_in_tile + row] = ckernel::sfpu::_sfpu_sigmoid_(gate);
    });
}

template <bool Gate>
void cap(uint32_t tile, uint32_t, uint32_t) {
    const uint32_t base = tile * vectors_in_tile;
    face_rows([=](uint32_t row) {
        vFloat value = dst_reg[base + row];
        if constexpr (!Gate) {
            value = round_projection(value);
        }
        value *= Gate ? 0.25f : 0.04f;
        value = ckernel::sfpu::_sfpu_tanh_fp32_accurate_(value);
        dst_reg[base + row] = value * (Gate ? 4.0f : 25.0f);
    });
}

void hidden(uint32_t tile, uint32_t, uint32_t) {
    const uint32_t base = tile * vectors_in_tile;
    face_rows([=](uint32_t row) {
        vFloat value = dst_reg[base + row];
        value *= dst_reg[6 * vectors_in_tile + row];
        value *= dst_reg[base + pairs * vectors_in_tile + row];
        dst_reg[base + row] = ::float32_to_bf16_rne(value);
    });
}

void combine(uint32_t gamma_bits, uint32_t slot, uint32_t tile) {
    vFloat gamma = __builtin_bit_cast(float, gamma_bits);
    const uint32_t base = tile * vectors_in_tile;
    face_rows([=](uint32_t row) {
        const uint32_t d = base + row;
        const uint32_t sum = d + columns * vectors_in_tile;
        vFloat value = round_projection(dst_reg[d]);
        vFloat term = value * gamma;
        vFloat acc = slot == 0 ? term : vFloat(dst_reg[sum]) + term;
        dst_reg[sum] = slot == 7 ? ::float32_to_bf16_rne(acc) : acc;
        dst_reg[d] = 0.0f;
    });
}
}  // namespace
#endif

void kernel_main() {
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t block_k = get_compile_time_arg_val(1);
    constexpr uint32_t cols = get_compile_time_arg_val(2);

#ifdef EXPERTS_VERIFY_COMBINE
    compute_kernel_hw_startup(cb_x, cb_out);
    copy_init(cb_x);
    for (uint32_t i = get_arg_val<uint32_t>(0); i < get_arg_val<uint32_t>(1); ++i) {
        uint32_t gamma[8];
        for (uint32_t slot = 0; slot < 8; ++slot) {
            MATH(gamma[slot] = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        }
        tile_regs_acquire();
        for (uint32_t slot = 0; slot < 8; ++slot) {
            cb_wait_front(cb_x, 1);
            copy_tile(cb_x, 0, 0);
            cb_pop_front(cb_x, 1);
            MATH((_llk_math_eltwise_binary_sfpu_params_(combine, gamma[slot], slot, 0, VectorMode::None)));
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(1, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#else
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_block_init(cb_x, cb_w, 0, cols, 1, k);
#if EXPERTS_GATE
    cb_wait_front(cb_x, k);
    for (uint32_t slot = 0; slot < 8 / EXPERTS_GU_LANES; ++slot) {
        matmul_block_init(cb_x, cb_w, 0, cols, 1, k);
        tile_regs_acquire();
        for (uint32_t step = 0; step < k; step += block_k) {
            cb_wait_front(cb_w, block_k * cols);
            matmul_block_k(cb_x, cb_w, step, 0, 0, 0, cols, 1, k, block_k, cols);
            cb_pop_front(cb_w, block_k * cols);
        }
        for (uint32_t tile = 0; tile < pairs; ++tile) {
            MATH((ckernel::sfpu::sigmoid_init<false>()));
            MATH((_llk_math_eltwise_binary_sfpu_params_(gate_sigmoid, tile, 0, 0, VectorMode::None)));
            MATH((ckernel::sfpu::tanh_init<false, true>()));
            MATH((_llk_math_eltwise_binary_sfpu_params_(cap<true>, tile, 0, 0, VectorMode::None)));
            MATH((_llk_math_eltwise_binary_sfpu_params_(cap<false>, tile + pairs, 0, 0, VectorMode::None)));
            MATH((_llk_math_eltwise_binary_sfpu_params_(hidden, tile, 0, 0, VectorMode::None)));
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, cols / 2);
        tile_regs_wait();
        for (uint32_t i = 0; i < cols / 2; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols / 2);
    }
    cb_pop_front(cb_x, k);
#else
    uint32_t gamma[8];
    for (uint32_t slot = 0; slot < 8; ++slot) {
        MATH(gamma[slot] = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    }
    tile_regs_acquire();
    for (uint32_t slot = 0; slot < 8; ++slot) {
        cb_wait_front(cb_x, k);
        for (uint32_t step = 0; step < k; step += block_k) {
            cb_wait_front(cb_w, block_k * cols);
            matmul_block_k(cb_x, cb_w, step, 0, 0, 0, cols, 1, k, block_k, cols);
            cb_pop_front(cb_w, block_k * cols);
        }
        cb_pop_front(cb_x, k);
        for (uint32_t tile = 0; tile < cols; ++tile) {
            MATH((_llk_math_eltwise_binary_sfpu_params_(combine, gamma[slot], slot, tile, VectorMode::None)));
        }
    }
    constexpr uint32_t first = cols;
    constexpr uint32_t count = cols;
    tile_regs_commit();
    cb_reserve_back(cb_out, count);
    tile_regs_wait();
    for (uint32_t i = 0; i < count; ++i) {
        pack_tile(first + i, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, count);
#endif
#endif
}
