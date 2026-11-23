#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_row_bytes = 32;
    const uint32_t face = get_common_arg_val<uint32_t>(3);
    const auto src = TensorAccessor(TensorAccessorArgs<1>(), get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto dst = TensorAccessor(
        TensorAccessorArgs<get_compile_time_arg_val(0)>(), get_common_arg_val<uint32_t>(1), tile_bytes);
    const uint32_t row = get_write_ptr(0);
    const uint32_t tile = get_write_ptr(16);

    // Each BF16 face begins on a 64-byte boundary. Keep both reads aligned
    // even when the two faces belong to different tiles.
    noc_async_read<face_row_bytes>(src.get_noc_addr(face / 2, (face % 2) * 512), row, face_row_bytes);
    noc_async_read<face_row_bytes>(
        src.get_noc_addr((face + 1) / 2, ((face + 1) % 2) * 512), row + 64, face_row_bytes);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t offset = 0; offset < tile_bytes; offset += MEM_ZEROS_SIZE) {
        noc_async_read<MEM_ZEROS_SIZE>(zeros, tile + offset, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();

    const auto logits = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row);
    auto sink = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile);
    for (uint32_t head = 0; head < 16; ++head) {
        sink[head * 16] = logits[head];
    }
    for (uint32_t head = 0; head < 8; ++head) {
        sink[512 + head * 16] = logits[32 + head];
    }
    noc_async_write<tile_bytes>(tile, dst.get_noc_addr(0), tile_bytes);
    noc_async_write_barrier();
}
