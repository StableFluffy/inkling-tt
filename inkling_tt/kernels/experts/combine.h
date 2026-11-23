#pragma once

#include "ckernel_sfpu_conversions.h"

namespace expert_math {
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

template <uint32_t Columns>
void combine(uint32_t gamma_bits, uint32_t slot, uint32_t tile) {
    vFloat gamma = __builtin_bit_cast(float, gamma_bits);
    const uint32_t base = tile * vectors_in_tile;
    face_rows([=](uint32_t row) {
        const uint32_t d = base + row;
        const uint32_t sum = d + Columns * vectors_in_tile;
        vFloat value = round_projection(dst_reg[d]);
        vFloat term = value * gamma;
        vFloat acc = slot == 0 ? term : vFloat(dst_reg[sum]) + term;
        dst_reg[sum] = slot == 7 ? ::float32_to_bf16_rne(acc) : acc;
        dst_reg[d] = 0.0f;
    });
}
}  // namespace expert_math
