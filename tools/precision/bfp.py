"""CPU implementation of tt-metal 79951bd44b7 FP32 host BFP_B tiles."""

import torch


def encode_rows(rows, magnitude_bits):
    """Return one exponent and 16 sign-magnitude codes per face row."""
    assert rows.dtype == torch.float32 and rows.shape[-1] == 16
    assert magnitude_bits in (3, 7)
    words = rows.view(torch.int32)
    exps = (words >> 23) & 255
    shared = exps.amax(dim=-1, keepdim=True)
    aligned = ((words & 0x7FFFFF) | 0x800000) >> (shared - exps).clamp_max_(31)
    shift = 24 - magnitude_bits
    odd = (aligned >> shift) & 1
    magnitude = (aligned + ((1 << (shift - 1)) - 1) + odd) >> shift
    magnitude.clamp_max_((1 << magnitude_bits) - 1)
    magnitude.masked_fill_(exps == 0, 0)
    sign = (words < 0).to(torch.int32) << magnitude_bits
    return shared, magnitude | (sign * (magnitude != 0))


def decode_rows(shared, codes, magnitude_bits):
    """Reproduce host unpacking, including low exponent and special bit patterns."""
    codes = codes.int()
    shared = shared.int()
    magnitude = codes & ((1 << magnitude_bits) - 1)
    sign = (codes >> magnitude_bits) << 31
    # For normal checkpoint exponents, the power-of-two grid gives identical bits.
    if not (((shared > 0) & (shared < magnitude_bits)) | (shared == 255)).any():
        step_exponent = (shared - (magnitude_bits - 1)).clamp_min_(1)
        value = magnitude.float() * (step_exponent << 23).view(torch.float32)
        return (value.view(torch.int32) | sign).view(torch.float32)

    shifts = torch.tensor([0] + [magnitude_bits - m.bit_length() for m in range(1, 1 << magnitude_bits)])
    normalization = shifts[magnitude.long()].int()
    exponent = shared - normalization
    assert not ((exponent < 0) & (magnitude != 0)).any()
    exponent = torch.where(magnitude == 0, 0, exponent)
    fraction = ((magnitude << (normalization + 1)) & ((1 << magnitude_bits) - 1)) << (23 - magnitude_bits)
    return (sign | (exponent << 23) | fraction).view(torch.float32)


def quantize(weight, magnitude_bits, axis):
    """Return unpacked weights in PyTorch [expert, out, in] index order."""
    assert axis in ("K", "N")
    storage = weight if axis == "K" else weight.transpose(-1, -2).contiguous()
    rows = storage.reshape(*storage.shape[:-1], -1, 16)
    shared, codes = encode_rows(rows, magnitude_bits)
    values = decode_rows(shared, codes, magnitude_bits).reshape(storage.shape)
    return values if axis == "K" else values.transpose(-1, -2)


def pack_tiles(tiles, magnitude_bits):
    """Serialize (..., 32, 32) row-major tiles in TT face order."""
    assert tiles.shape[-2:] == (32, 32)
    faces = tiles.reshape(*tiles.shape[:-2], 2, 16, 2, 16).transpose(-2, -3)
    rows = faces.reshape(*tiles.shape[:-2], 64, 16)
    exps, codes = encode_rows(rows, magnitude_bits)
    codes = codes.to(torch.uint8)
    if magnitude_bits == 3:
        codes = codes[..., ::2] | (codes[..., 1::2] << 4)
    return torch.cat((exps.squeeze(-1).to(torch.uint8), codes.flatten(-2)), dim=-1)


def unpack_tiles(packed, magnitude_bits):
    """Deserialize standard tiles to (..., 32, 32) FP32 in row-major order."""
    prefix = packed.shape[:-1]
    assert packed.dtype == torch.uint8 and packed.shape[-1] == 64 + 128 * (magnitude_bits + 1)
    exps = packed[..., :64].unsqueeze(-1)
    codes = packed[..., 64:]
    if magnitude_bits == 3:
        codes = torch.stack((codes & 15, codes >> 4), dim=-1)
    faces = decode_rows(exps, codes.reshape(*prefix, 64, 16), magnitude_bits)
    return faces.reshape(*prefix, 2, 2, 16, 16).transpose(-2, -3).reshape(*prefix, 32, 32)


class NativeRowQuantizer:
    """Use the pinned CPU SIMD encoder with the same grid as quantize()."""

    def __init__(self, library):
        import ctypes

        self.convert = ctypes.CDLL(str(library.resolve())).quantize_bfp
        self.convert.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        self.convert.restype = None

    def __call__(self, weight, magnitude_bits, axis):
        assert magnitude_bits in (3, 7) and axis in ("K", "N") and weight.dtype == torch.float32
        storage = weight if axis == "K" else weight.transpose(-1, -2)
        storage = storage.contiguous()
        assert storage.shape[-1] % 16 == 0
        output = torch.empty_like(storage)
        self.convert(storage.data_ptr(), output.data_ptr(), storage.numel() // 16, magnitude_bits)
        return output if axis == "K" else output.transpose(-1, -2)
