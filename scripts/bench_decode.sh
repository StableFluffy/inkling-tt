#!/bin/bash
# Batch-1 traced decode latency on the Haiku prompt (tests/bench_decode.py): greedy (device argmax) and logits
# readback, 64 replays with changing token and position, first 4 discarded, median and p95 of per-step wall time.
#   INK_LMAX         decode cache length (default 8192, the length of the published figure)
#   INKLING_EXPERTS  stream|stock (default stream); INK_ATTN_DECODE=0 selects the stock SDPA path for comparison
# Needs $INKLING_GOLDEN/capture_haiku.pt (tools/golden/capture.py).
set -euo pipefail
HERE=$(dirname "${BASH_SOURCE[0]}")
source "$HERE/env.sh"
export INK_LMAX=${INK_LMAX:-8192} INK_ATTN_DECODE=${INK_ATTN_DECODE:-1}
mkdir -p "$INKLING_WORK/logs"
LOG=$INKLING_WORK/logs/bench_decode-$INKLING_EXPERTS-attn$INK_ATTN_DECODE-lmax$INK_LMAX-$(date +%Y%m%d-%H%M%S).log
"$HERE/dev_run.sh" "OMP_NUM_THREADS=4 python -u tests/bench_decode.py" 2>&1 | tee "$LOG"
echo "log: $LOG"
