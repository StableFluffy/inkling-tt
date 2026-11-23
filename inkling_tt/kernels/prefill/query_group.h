#pragma once

#ifndef PREFILL_QUERY_GROUP
#define PREFILL_QUERY_GROUP 1
#endif

constexpr uint32_t group_size = PREFILL_QUERY_GROUP;
constexpr bool group_heads = group_size == 3;
constexpr uint32_t query_step = group_heads ? 0 : 1;
constexpr uint32_t head_step = group_heads ? 1 : 0;
constexpr uint32_t query_tiles_per_task = group_heads ? 1 : group_size;
constexpr uint32_t heads_per_task = group_heads ? group_size : 1;

inline uint32_t first_query(uint32_t task, uint32_t query_tiles) {
    const uint32_t groups = query_tiles / query_tiles_per_task;
    const uint32_t g = task % groups;
    return query_tiles_per_task * (g % 2 == 0 ? g / 2 : groups - 1 - g / 2);
}

inline uint32_t shared_keys(uint32_t first, uint32_t saved) {
    const uint32_t count = first + 1;
    return count > saved ? (count - saved) / 8 * 8 : 0;
}

inline uint32_t live_queries(uint32_t qt, uint32_t valid_tiles) {
    return group_heads || valid_tiles - qt >= group_size ? group_size : valid_tiles - qt;
}
