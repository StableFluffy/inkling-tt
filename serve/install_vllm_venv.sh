#!/bin/bash
# Build the vLLM 0.31.0 (VLLM_TARGET_DEVICE=empty) + vllm-tt-plugin compat/vllm-0.31.0 venv on top of a tt-metal Python env.
set -euxo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TT_METAL_VENV:?set TT_METAL_VENV to the tt-metal Python 3.10 venv that provides ttnn}"
V=${INKLING_VLLM_VENV:-$REPO/.venv-vllm}
SRC=${INKLING_SRC_CACHE:-$REPO/.src}
PLUGIN_COMMIT=0e96e9eec1387bbc8e9a75457f2a31d67acdcf0a
VLLM_SDIST_SHA256=dde46f1efbe846e5bd0c5c4ec6ec71842c42254c7ff3d9bebb2a57028715d380
export USER=${USER:-$(id -un 2>/dev/null || echo inkling)} LOGNAME=${LOGNAME:-${USER}}
mkdir -p "$SRC" && cd "$SRC"
test -d vllm-tt-plugin || git clone https://github.com/tenstorrent/vllm-tt-plugin.git
test -d vllm-tt-plugin-031 || git -C vllm-tt-plugin worktree add ../vllm-tt-plugin-031 $PLUGIN_COMMIT
[ "$(git -C vllm-tt-plugin-031 rev-parse HEAD)" = "$PLUGIN_COMMIT" ] || { echo "vllm-tt-plugin-031 is not at $PLUGIN_COMMIT" >&2; exit 1; }
[ -z "$(git -C vllm-tt-plugin-031 status --porcelain --untracked-files=no)" ] || { echo "vllm-tt-plugin-031 has local changes" >&2; exit 1; }
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/requirements/common.txt -o vllm-0.31.0-common.txt
sed -e "s/^transformers >= 5.10.4, < 5.18.0/transformers >= 5.10.4, < 5.15.0/" \
    -e "s/^opencv-python-headless >= 4.13.0 .*/opencv-python-headless == 4.11.0.86/" \
    -e "s/^numpy$/numpy >= 1.24.4, < 2/" vllm-0.31.0-common.txt > vllm-0.31.0-common-tt.txt
test -f vllm-0.31.0.tar.gz || curl -fsSL -o vllm-0.31.0.tar.gz \
  https://files.pythonhosted.org/packages/96/89/11460eaeb96e636f716657d460a6daebc7ddbcd631dd6799f232cae25771/vllm-0.31.0.tar.gz
echo "$VLLM_SDIST_SHA256  vllm-0.31.0.tar.gz" | sha256sum -c
python -m venv "$V"
SP=$(ls -d "$TT_METAL_VENV"/lib/python3.10/site-packages)
echo "import site; site.addsitedir(\"$SP\")" > "$V/lib/python3.10/site-packages/zz_tt_metal_venv.pth"
P=$V/bin/python
$P -m pip install --upgrade "pip==25.3"
$P -m pip install -r vllm-0.31.0-common-tt.txt
$P -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchvision==0.26.0
$P -m pip install numba==0.61.2
VLLM_TARGET_DEVICE=empty $P -m pip install --no-deps ./vllm-0.31.0.tar.gz
$P -m pip install -e "$SRC/vllm-tt-plugin-031"
$P -m pip install -e "$REPO/serve/vllm_inkling"
$P - <<'PY'
import sys, numpy, torch, transformers, ttnn, vllm, vllm_tt_plugin
got = dict(python=sys.version.split()[0][:4], vllm=vllm.__version__, torch=torch.__version__, transformers=transformers.__version__)
print(got, "numpy", numpy.__version__, "ttnn", ttnn.__file__)
want = dict(python="3.10", vllm="0.31.0+empty", torch="2.11.0+cpu", transformers="5.12.1")
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
sys.exit(f"unexpected versions: {bad}" if bad else 0)
PY
