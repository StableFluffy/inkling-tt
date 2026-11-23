#!/bin/bash
# Test tiers. Every device tier needs one free p150a selected by INKLING_DEVICE and stops at the first failure.
#   host     no device: byte-compile the sources and import ttnn, the model and the server
#   kernels  rowvm, expert-stream and fused attention-decode kernel parity (needs INKLING_HF_SD)
#   parity   full model vs HF captures: eager/trace decode, chunked prefill, 300-token prompt
#            (capture_haiku.pt, long300.pt; asserts the thresholds in docs/benchmarks.md)
#   eval     teacher-forced top-1/top-5/relerr on 3 prompts x 128 tokens (eval3x128.pt; asserts thresholds)
#   buckets  128 + 2048-row prefill buckets at lmax 32768: golden check, bucket equality, TTFT ladder, decode handoff
#   serve    engine smoke test through serve.openai_server.Engine
#   all      kernels, parity, eval, buckets and serve, in order
set -euo pipefail
HERE=$(dirname "${BASH_SOURCE[0]}")
source "$HERE/env.sh"
TIER=${1:-parity}
LOG=$INKLING_WORK/logs/tests-$TIER-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
run() {
  local name=$1; shift
  echo "== $name"
  "$HERE/dev_run.sh" "$*" 2>&1 | tee "$LOG/$name.log"
}
need() {
  for f in "$@"; do
    [ -f "$INKLING_GOLDEN/$f" ] || { echo "missing $INKLING_GOLDEN/$f: generate it with tools/golden (README step 3)" >&2; exit 1; }
  done
}
tier_host() {
  INKLING_DEVICE=none run host "python -m compileall -q inkling_tt serve tests tools >/dev/null && python -c 'import ttnn, inkling_tt.model2, inkling_tt.prefill, serve.openai_server, serve.runtime; print(\"imports ok\", ttnn.__file__)'"
}
tier_kernels() {
  run rowvm "python -u tests/test_rowvm.py"
  run rowvm_attention "python -u tests/rowvm_attention.py"
  run experts_stream "python -u inkling_tt/kernels/experts/verify.py --placement mixed"
  run experts_prefill "python -u tests/test_experts_prefill.py"
  run attn_decode "OMP_NUM_THREADS=4 python -u tests/bench_attn_decode.py"
}
tier_parity() {
  need capture_haiku.pt long300.pt
  run full_eager "python -u tests/test_full.py eager"
  run full_trace "python -u tests/test_full.py trace"
  run prefill "python -u tests/test_prefill.py 128"
  run long300 "OMP_NUM_THREADS=4 python -u tests/test_long.py"
}
tier_eval() {
  need eval3x128.pt
  run eval_tf "OMP_NUM_THREADS=4 python -u tests/eval_tf.py bfloat8_b bfloat16 bfloat16 HiFi4"
}
tier_buckets() {
  need long300.pt
  run prefill_buckets "OMP_NUM_THREADS=4 PREFILL_LMAX=32768 PREFILL_LENGTHS=${PREFILL_LENGTHS:-128,300,1024,2048,8192,32768} python -u tests/test_prefill_buckets.py"
}
tier_serve() {
  need capture_haiku.pt
  run test_engine "python -u -m serve.test_engine"
}
case "$TIER" in
  host|kernels|parity|eval|buckets|serve) "tier_$TIER" ;;
  all) tier_kernels; tier_parity; tier_eval; tier_buckets; tier_serve ;;
  *) echo "usage: $0 {host|kernels|parity|eval|buckets|serve|all}" >&2; exit 2 ;;
esac
echo "logs in $LOG"
