#pragma once

#include <cstdint>

// Lanes split the expert-ordered compact blocks into equal contiguous ranges.
// Every worker of a lane visits the same blocks in the same order.
struct LaneRange {
    uint32_t lo;
    uint32_t hi;
};

template <typename Ends>
inline LaneRange lane_range(const Ends& ends, uint32_t lane, uint32_t lanes) {
    const uint32_t total = ends(127);
    return {lane * total / lanes, (lane + 1) * total / lanes};
}

template <typename Ends>
inline bool lane_expert(const Ends& ends, LaneRange range, uint32_t expert, uint32_t& begin, uint32_t& end) {
    begin = expert == 0 ? 0 : ends(expert - 1);
    end = ends(expert);
    begin = begin > range.lo ? begin : range.lo;
    end = end < range.hi ? end : range.hi;
    return begin < end;
}

// Runtime argument layout shared by the three lane kernels. Rectangles are
// (x0, y0, x1, y1, cores) in NoC 0 coordinates; member coordinates follow them.
namespace lane_args {
constexpr uint32_t group = 0;
constexpr uint32_t lane = 1;
constexpr uint32_t lanes = 2;
constexpr uint32_t banks = 3;
constexpr uint32_t vc = 7;
constexpr uint32_t me = 8;
constexpr uint32_t members = 9;
constexpr uint32_t rects = 10;
}  // namespace lane_args
