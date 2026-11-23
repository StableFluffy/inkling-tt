#pragma once

#include "api/dataflow/dataflow_api.h"

inline void select_read_vc() {
    const uint32_t vc = get_absolute_logical_y() % 4;
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, vc);
}
