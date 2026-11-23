#!/bin/bash
# Create the TTNN Python environment at $TT_METAL_VENV inside the pinned dev image (host-only, no device).
# It layers on the image's /opt/venv (torch 2.11.0+cpu, transformers 5.12.1), installs the built tt-metal tree
# as an editable ttnn package and adds the packages the model and the server import.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${TT_METAL_HOME:?set TT_METAL_HOME to the built tt-metal tree}"
: "${TT_METAL_VENV:?set TT_METAL_VENV to the venv directory to create}"
[ ! -e "$TT_METAL_VENV" ] || { echo "$TT_METAL_VENV exists; remove it or choose another path" >&2; exit 1; }
mkdir -p "$TT_METAL_VENV"
V=$TT_METAL_VENV
INKLING_DEVICE=none INKLING_NO_ACTIVATE=1 "$HERE/dev_run.sh" "set -e
  /opt/venv/bin/python3 -m venv '$V'
  echo /opt/venv/lib/python3.10/site-packages > '$V/lib/python3.10/site-packages/zz_optvenv.pth'
  cd \$TT_METAL_HOME && /opt/venv/bin/uv pip install --python '$V/bin/python' --no-deps -e .
  /opt/venv/bin/uv pip install --python '$V/bin/python' --no-deps graphviz==0.21 ml_dtypes==0.4.1
  /opt/venv/bin/uv pip install --python '$V/bin/python' fastapi==0.118.0 uvicorn==0.37.0
  '$V/bin/python' -c 'import fastapi, uvicorn, torch, transformers; print(\"venv ok: torch\", torch.__version__, \"transformers\", transformers.__version__)'"
