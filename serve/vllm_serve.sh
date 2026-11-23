#!/bin/bash
# Experimental: serve Inkling-2-Mini through vLLM 0.31.0 + vllm-tt-plugin (venv from install_vllm_venv.sh).
# Binds to ${HOST:-127.0.0.1}:${PORT:-8401}; the API has no authentication. Context is limited to 2,048 tokens.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${INKLING_SNAPSHOT:?set INKLING_SNAPSHOT to the thinkingmachines/Inkling-2-Mini snapshot}"
: "${INKLING_HF_SD:?set INKLING_HF_SD to the converted shards (tools/convert)}"
VENV=${INKLING_VLLM_VENV:-$REPO/.venv-vllm}
export USER=${USER:-$(id -un 2>/dev/null || echo inkling)} LOGNAME=${LOGNAME:-${USER}}
export EXTRA_MODELS_DIR=$REPO/serve
export TT_VLLM_BUILTIN_MODELS=0
export INKLING_TT_WEIGHTS=${INKLING_TT_WEIGHTS:-$INKLING_HF_SD}
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
if [[ -v INK_GRID ]]; then
  # The vLLM worker can open its device before loading the model adapter, so apply the grid limit up front.
  grid_end=$("$VENV/bin/python" -c 'import os; import inkling_tt.grid; print(os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"])') || exit 1
  export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="$grid_end"
fi
exec "$VENV/bin/vllm" serve "$INKLING_SNAPSHOT" \
  --served-model-name Inkling-2-Mini-TT --max-model-len 2048 --max-num-seqs 1 \
  --no-enable-prefix-caching --reasoning-parser inkling --host "${HOST:-127.0.0.1}" --port "${PORT:-8401}" \
  --additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 400000000, "l1_small_size": 16384}}' "$@"
