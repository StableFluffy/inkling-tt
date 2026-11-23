#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

using namespace ckernel;
constexpr uint32_t cols = get_compile_time_arg_val(2);

#if defined(TRISC_MATH) && EXPERTS_GATE
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_sigmoid.h"
#ifdef EXPERTS_VERIFY
#include "ckernel_sfpu_tanh.h"
#else
#include "ckernel_sfpu_softcap.h"
#endif
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace {
using namespace sfpi;
constexpr uint32_t vectors = 32;
constexpr uint32_t sigmoid_tile = 6;

template <typename F>
void live_rows(F fn) {
#pragma GCC unroll 2
    for (uint32_t face = 0; face < 16; face += 8) {
#pragma GCC unroll 4
        for (uint32_t v = 0; v < 4; ++v) {
            fn(face + v);
        }
    }
}

void gate_sigmoid(uint32_t tile, uint32_t, uint32_t) {
    live_rows([=](uint32_t v) {
        const uint32_t i = tile * vectors + v;
        vFloat g = convert<vFloat16b>(vFloat(dst_reg[i]), RoundMode::Nearest);
        dst_reg[i] = g;
        dst_reg[sigmoid_tile * vectors + v] = ckernel::sfpu::_sfpu_sigmoid_(g);
    });
}

template <bool Gate>
void cap(uint32_t tile, uint32_t, uint32_t) {
    live_rows([=](uint32_t v) {
        const uint32_t i = tile * vectors + v;
        vFloat x = dst_reg[i];
        if constexpr (!Gate) {
            x = convert<vFloat16b>(x, RoundMode::Nearest);
        }
#ifdef EXPERTS_VERIFY
        x *= Gate ? 0.25f : 0.04f;
        dst_reg[i] = ckernel::sfpu::_sfpu_tanh_fp32_accurate_(x) * (Gate ? 4.0f : 25.0f);
#else
        dst_reg[i] = ckernel::sfpu::_sfpu_softcap_(
            x, vFloat(Gate ? 4.0f : 25.0f), vFloat(Gate ? 0.25f : 0.04f));
#endif
    });
}

void hidden(uint32_t tile, uint32_t, uint32_t) {
    live_rows([=](uint32_t v) {
        const uint32_t i = tile * vectors + v;
        vFloat h = vFloat(dst_reg[i]) * vFloat(dst_reg[sigmoid_tile * vectors + v]);
        h *= dst_reg[i + cols / 2 * vectors];
        dst_reg[i] = ::float32_to_bf16_rne(h);
    });
}
}  // namespace
#endif

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t block_k = get_compile_time_arg_val(1);
    constexpr uint32_t out_cols = EXPERTS_GATE ? cols / 2 : cols;
#if EXPERTS_GATE
    const uint32_t lane = get_arg_val<uint32_t>(3);
    constexpr uint32_t lanes = EXPERTS_GU_LANES;
#else
    constexpr uint32_t lane = 0;
    constexpr uint32_t lanes = 1;
#endif
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(0, 1, 16);
    matmul_block_init(0, 1, 0, cols, 1, k);
    cb_wait_front(2, 1);
    for (uint32_t expert = lane; expert < 128; expert += lanes) {
        const uint32_t begin = expert == 0 ? 0 : read_tile_value(2, 0, expert - 1);
        const uint32_t end = read_tile_value(2, 0, expert);
        if (begin == end) {
            continue;
        }
#ifndef EXPERTS_VERIFY_STREAM
        cb_wait_front(1, k * cols);
#endif
        for (uint32_t block = begin; block < end; ++block) {
            cb_wait_front(0, k);
            tile_regs_acquire();
            for (uint32_t step = 0; step < k; step += block_k) {
#ifdef EXPERTS_VERIFY_STREAM
                cb_wait_front(1, block_k * cols);
                matmul_block_k(0, 1, step, 0, 0, 0, cols, 1, k, block_k, cols);
                cb_pop_front(1, block_k * cols);
#else
                matmul_block_k(0, 1, step, step * cols, 0, 0, cols, 1, k, block_k, cols);
#endif
            }
            cb_pop_front(0, k);
#if EXPERTS_GATE
            for (uint32_t t = 0; t < out_cols; ++t) {
                MATH((ckernel::sfpu::sigmoid_init<false>()));
                MATH((_llk_math_eltwise_binary_sfpu_params_(gate_sigmoid, t, 0, 0, VectorMode::None)));
#ifdef EXPERTS_VERIFY
                MATH((ckernel::sfpu::tanh_init<false, true>()));
#else
                MATH((ckernel::sfpu::softcap_init()));
#endif
                MATH((_llk_math_eltwise_binary_sfpu_params_(cap<true>, t, 0, 0, VectorMode::None)));
                MATH((_llk_math_eltwise_binary_sfpu_params_(cap<false>, t + out_cols, 0, 0, VectorMode::None)));
                MATH((_llk_math_eltwise_binary_sfpu_params_(hidden, t, 0, 0, VectorMode::None)));
            }
#endif
            tile_regs_commit();
            cb_reserve_back(16, out_cols);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_cols; ++i) {
                pack_tile(i, 16);
            }
            tile_regs_release();
            cb_push_back(16, out_cols);
        }
#ifndef EXPERTS_VERIFY_STREAM
        cb_pop_front(1, k * cols);
#endif
    }
    cb_pop_front(2, 1);
}
