// CPU fast path using the pinned tt-metal row packer from bfp_simd.hpp.
#include "bfp_simd.hpp"
#include <bit>
#include <cstddef>
#include <cstdint>

using tt::tt_metal::bfp_simd::select_row_packer;

template <int bits>
void quantize(const float* input, float* output, size_t rows) {
    constexpr int mask = (1 << bits) - 1;
    const auto pack = select_row_packer<bits, float>();
    for (size_t row = 0; row < rows; ++row) {
        uint8_t exponent;
        uint8_t packed[2 * (bits + 1)];
        pack(input + 16 * row, &exponent, packed);
        float* dest = output + 16 * row;
        if (exponent == 0) {
            std::fill(dest, dest + 16, 0.0f);
            continue;
        }
        if (exponent >= bits && exponent < 255) {
            const float step = std::bit_cast<float>(uint32_t{exponent - unsigned(bits - 1)} << 23);
            for (int i = 0; i < 16; ++i) {
                const unsigned code = bits == 3 ? (packed[i / 2] >> (4 * (i % 2))) & 15 : packed[i];
                const int magnitude = code & mask;
                dest[i] = ((code >> bits) ? -magnitude : magnitude) * step;
            }
            continue;
        }
        for (int i = 0; i < 16; ++i) {
            const unsigned code = bits == 3 ? (packed[i / 2] >> (4 * (i % 2))) & 15 : packed[i];
            const unsigned magnitude = code & mask;
            if (magnitude == 0) {
                dest[i] = 0.0f;
                continue;
            }
            const int shift = bits - std::bit_width(magnitude);
            const unsigned fraction = ((magnitude << (shift + 1)) & mask) << (23 - bits);
            const unsigned word = ((code >> bits) << 31) | ((exponent - shift) << 23) | fraction;
            dest[i] = std::bit_cast<float>(word);
        }
    }
}

extern "C" void quantize_bfp(const float* input, float* output, size_t rows, int bits) {
    if (bits == 3) quantize<3>(input, output, rows);
    else quantize<7>(input, output, rows);
}
