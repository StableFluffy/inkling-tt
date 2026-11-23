"""Compare emulator bytes and host-unpacked bits against pinned tt-metal C++."""

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from bfp import NativeRowQuantizer, pack_tiles, unpack_tiles
from dequant import CHECKPOINT, SHARD, decode, limit_torch, read_raw
from safetensors import safe_open

REVISION = "79951bd44b7"
DATA_FORMAT = "tt_metal/impl/data_format"
CPP = r"""
#include "bfp_simd.hpp"
#include <cassert>
#include <cstdio>
namespace tt { enum class DataFormat { Bfp2, Bfp2_b, Bfp4, Bfp4_b, Bfp8, Bfp8_b }; }
#define TT_ASSERT(cond, ...) assert(cond)
@TT_UNPACK@
using tt::DataFormat;
using namespace tt::tt_metal::bfp_simd;

template <int bits>
void run() {
    const auto scalar = select_row_packer<bits, float>(Isa::Scalar);
    const auto cpu = select_row_packer<bits, float>(Isa::Auto);
    constexpr auto format = bits == 3 ? DataFormat::Bfp4_b : DataFormat::Bfp8_b;
    constexpr int row_bytes = 2 * (bits + 1);
    float tile[32][32];
    while (fread(tile, sizeof(tile), 1, stdin) == 1) {
        uint8_t packed[64 + 64 * row_bytes];
        uint32_t decoded[32][32];
        for (int face = 0; face < 4; ++face) {
            int tile_col = 16 * (face % 2);
            for (int row = 0; row < 16; ++row) {
                int tile_row = 16 * (face / 2) + row;
                int i = 16 * face + row;
                auto* exp = packed + i;
                auto* data = packed + 64 + i * row_bytes;
                cpu(&tile[tile_row][tile_col], exp, data);
                uint8_t scalar_exp, scalar_data[row_bytes];
                scalar(&tile[tile_row][tile_col], &scalar_exp, scalar_data);
                assert(*exp == scalar_exp && std::memcmp(data, scalar_data, row_bytes) == 0);
                for (int j = 0; j < 16; ++j) {
                    uint8_t code = bits == 3 ? (data[j / 2] >> (4 * (j % 2))) & 15 : data[j];
                    decoded[tile_row][tile_col + j] = convert_bfp_to_u32(format, code, *exp, false);
                }
            }
        }
        fwrite(packed, sizeof(packed), 1, stdout);
        fwrite(decoded, sizeof(decoded), 1, stdout);
    }
}

int main(int argc, char** argv) {
    assert(argc == 2);
    if (argv[1][0] == '3') run<3>();
    else run<7>();
}
"""


def pinned_sources(root, scratch):
    def source(name):
        return subprocess.check_output(["git", "-C", str(root), "show", f"{REVISION}:{DATA_FORMAT}/{name}"], text=True)

    (scratch / "bfp_simd.hpp").write_text(source("bfp_simd.hpp"))
    # The harness only instantiates the float templates.
    (scratch / "tt-metalium").mkdir()
    (scratch / "tt-metalium/bfloat16.hpp").write_text("struct bfloat16 { unsigned short value; };\n")
    return source


def compile_reference(root, scratch):
    source = pinned_sources(root, scratch)
    unpack = source("blockfloat_common.cpp")
    start = unpack.index("uint32_t convert_bfp_to_u32")
    end = unpack.index("template <tt::DataFormat BfpFormat, bool truncate_bfp_mantissa>", start)
    (scratch / "verify.cpp").write_text(CPP.replace("@TT_UNPACK@", unpack[start:end]))
    executable = scratch / "verify"
    subprocess.run(
        ["g++", "-std=c++20", "-O2", "-I", str(scratch), str(scratch / "verify.cpp"), "-o", str(executable)],
        check=True,
        env={**os.environ, "TMPDIR": str(scratch)},
    )
    return executable


def test_tiles(checkpoint):
    generator = torch.Generator().manual_seed(79091)
    bits = torch.randint(-(1 << 31), 1 << 31, (256, 32, 32), dtype=torch.int32, generator=generator)
    normal = torch.randn(256, 32, 32, generator=generator)
    random_scale = 2.0 ** torch.randint(-124, 120, (256, 32, 2, 1), generator=generator)
    normals = (normal.reshape(256, 32, 2, 16) * random_scale).reshape(256, 32, 32)
    edges = torch.zeros((2, 32, 32), dtype=torch.float32)
    for row, mbits in enumerate((3, 7)):
        tie = torch.tensor(1.0 + 2.0**-mbits)
        near = [tie, torch.nextafter(tie, torch.tensor(0.0)), torch.nextafter(tie, torch.tensor(2.0))]
        edges[0, row, :8] = torch.tensor([1.0, -1.9999999, *near, *(-x for x in near)])
    edges.view(torch.int32)[0, 3, :6] = torch.tensor([0, -(1 << 31), 1, -2147483647, 0x007FFFFF, 0x00800000])
    edges.view(torch.int32)[0, 4, :4] = torch.tensor([0x7F800000, 0x7FC00123, 0x7FFFFFFF, -8388608])
    edges.view(torch.int32)[0, 5, :3] = torch.tensor([1, -2147483647, -(1 << 31)])
    with safe_open(checkpoint / SHARD, framework="pt", device="cpu") as shard:
        samples = []
        for layer in (0, 20):
            w = decode(read_raw(shard, layer, "gate_up_proj", slice(0, 1)), "gate_up_proj").fp32[0]
            for matrix in (w, w.T):
                view = matrix[:256, :256].reshape(8, 32, 8, 32).transpose(1, 2).reshape(64, 32, 32)
                samples.append(view)
    return torch.cat((bits.view(torch.float32), normals, edges, *samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-metal", type=Path, default=Path(os.environ.get("TT_METAL_HOME", "tt-metal")))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--build-fast", type=Path, help="Build and test the optional shared library")
    args = parser.parse_args()
    limit_torch(1)
    tiles = test_tiles(args.checkpoint)
    with TemporaryDirectory(dir=Path.cwd()) as temp:
        executable = compile_reference(args.tt_metal, Path(temp))
        fast = None
        if args.build_fast:
            cpp = Path(__file__).with_name("fast_bfp.cpp")
            subprocess.run(
                [
                    "g++",
                    "-std=c++20",
                    "-O3",
                    "-march=native",
                    "-fPIC",
                    "-shared",
                    "-I",
                    temp,
                    str(cpp),
                    "-o",
                    str(args.build_fast),
                ],
                check=True,
                env={**os.environ, "TMPDIR": temp},
            )
            fast = NativeRowQuantizer(args.build_fast)
        for bits in (3, 7):
            packed = pack_tiles(tiles, bits)
            unpacked = unpack_tiles(packed, bits).view(torch.int32)
            actual = subprocess.run(
                [str(executable), str(bits)], input=tiles.numpy().tobytes(), capture_output=True, check=True
            ).stdout
            nbytes = packed.shape[-1]
            rows = torch.frombuffer(bytearray(actual), dtype=torch.uint8).reshape(len(tiles), nbytes + 4096)
            assert torch.equal(packed, rows[:, :nbytes])
            expected = rows[:, nbytes:].contiguous().view(torch.int32).reshape_as(unpacked)
            assert torch.equal(unpacked, expected)
            if fast:
                assert torch.equal(fast(tiles, bits, "K").view(torch.int32), expected)
            print(f"BFP{bits + 1}_B: {len(tiles)} tiles, {nbytes} bytes per tile, matched packing and unpacking bits")


if __name__ == "__main__":
    main()
