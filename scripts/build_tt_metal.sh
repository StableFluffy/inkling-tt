#!/bin/bash
# Build the pinned tt-metal checkout at $TT_METAL_HOME inside the dev image (host-only, no device, ~6 min on 64 cores).
# The tree is mounted and built at /opt/tt-metal, the path the serving image uses, so compiled-in source paths
# do not carry the host directory layout.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${TT_METAL_HOME:?set TT_METAL_HOME to a tt-metal clone}"
PIN=79951bd44b7484a136b3619e0d81f26baa8a1936
[ "$(git -C "$TT_METAL_HOME" rev-parse HEAD)" = "$PIN" ] || { echo "tt-metal is not at $PIN" >&2; exit 1; }
git -C "$TT_METAL_HOME" describe --tags >/dev/null 2>&1 || { echo "tt-metal clone has no tags (CMake needs them): git -C \$TT_METAL_HOME fetch --tags" >&2; exit 1; }
INKLING_DEVICE=none INKLING_NO_ACTIVATE=1 "$HERE/dev_run.sh" "cd \${TT_METAL_HOME} && ./build_metal.sh --release --use-system-sfpi"
