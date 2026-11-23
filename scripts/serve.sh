#!/bin/bash
# OpenAI-compatible server on one p150a from a source checkout (no image). Extra arguments go to serve.openai_server.
#   PORT  listen port (default 8000)
#   HOST  bind address (default 127.0.0.1). The API has no authentication; HOST=0.0.0.0 exposes it on every
#         host interface, because the container shares the host network.
# Needs fastapi/uvicorn in TT_METAL_VENV (scripts/setup_venv.sh installs them).
set -euo pipefail
HERE=$(dirname "${BASH_SOURCE[0]}")
source "$HERE/env.sh"
"$HERE/dev_run.sh" "exec python -u -m serve.openai_server --weights '$INKLING_HF_SD' --tokenizer '$INKLING_SNAPSHOT' --host ${HOST:-127.0.0.1} --port ${PORT:-8000} $*"
