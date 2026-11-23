#pragma once

#ifndef PREFILL_QUERY_GROUP
#define PREFILL_QUERY_GROUP 1
#endif

constexpr uint32_t group_size = PREFILL_QUERY_GROUP;

inline uint32_t first_query(uint32_t task, uint32_t query_tiles) {
    const uint32_t groups = query_tiles / group_size;
    const uint32_t g = task % groups;
    return group_size * (g % 2 == 0 ? g / 2 : groups - 1 - g / 2);
}

inline uint32_t shared_keys(uint32_t first, uint32_t saved) {
    const uint32_t count = first + 1;
    return count > saved ? (count - saved) / 8 * 8 : 0;
}
