#!/bin/bash
# Run a command inside the pinned tt-metal dev image with a built tt-metal tree and this repo mounted.
#   TT_METAL_HOME     tt-metal checkout at 79951bd44b7484a136b3619e0d81f26baa8a1936 on the host (required). It is
#                     mounted at /opt/tt-metal, where scripts/build_tt_metal.sh builds it and where the image keeps it
#   INKLING_TT_METAL_MOUNT  override that mount point (e.g. the host path, for a git worktree checkout)
#   TT_METAL_VENV     venv from scripts/setup_venv.sh, activated inside the container
#   INKLING_DEVICE    card index for TT_VISIBLE_DEVICES, or "none" for host-only commands
#   INKLING_LOCK_DIR  optional directory for per-card flock files (chip<N>.lock) when several users share a host
#   INKLING_VLLM_VENV, INKLING_SRC_CACHE  optional vLLM venv / source cache locations (serve/install_vllm_venv.sh)
#   INK_* / PREFILL_* feature flags in the environment are passed through to the container
# The kernel cache $INKLING_WORK/kcache is mounted at /cache, as in the serving image.
# The container uses host networking: a server started here binds to 127.0.0.1 unless told otherwise.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal tree}"
[ $# -gt 0 ] || { echo "usage: $0 <command...>" >&2; exit 2; }
mkdir -p "$INKLING_WORK/home" "$INKLING_WORK/kcache" "$INKLING_WORK/logs"
TTM=${INKLING_TT_METAL_MOUNT:-/opt/tt-metal}
DEV=()
if [ "$INKLING_DEVICE" != none ]; then
  if [ -n "${INKLING_LOCK_DIR:-}" ]; then
    for c in ${INKLING_DEVICE//,/ }; do
      exec {fd}>"$INKLING_LOCK_DIR/chip$c.lock"
      flock -w 7200 "$fd" || { echo "card $c busy" >&2; exit 9; }
    done
  fi
  DEV=(--privileged -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent:/dev/tenstorrent
       -e TT_VISIBLE_DEVICES="$INKLING_DEVICE" -e TT_METAL_VISIBLE_DEVICES="$INKLING_DEVICE")
fi
MOUNTS=(-v "$REPO:$REPO" -v "$TT_METAL_HOME:$TTM" -v "$INKLING_WORK:$INKLING_WORK" -v "$INKLING_WORK/kcache:/cache")
for d in "$INKLING_HF_SD" "$INKLING_SNAPSHOT" "$INKLING_GOLDEN" ${TT_METAL_VENV:+"$TT_METAL_VENV"} \
         ${INKLING_VLLM_VENV:+"$INKLING_VLLM_VENV"} ${INKLING_SRC_CACHE:+"$INKLING_SRC_CACHE"}; do
  case "$d" in "$INKLING_WORK"/*|"$REPO"/*) ;; *) [ -e "$d" ] && MOUNTS+=(-v "$d:$d") ;; esac
done
ACTIVATE=
if [ -n "${TT_METAL_VENV:-}" ] && [ -z "${INKLING_NO_ACTIVATE:-}" ]; then
  [ -f "$TT_METAL_VENV/bin/activate" ] || { echo "TT_METAL_VENV=$TT_METAL_VENV has no bin/activate; run scripts/setup_venv.sh" >&2; exit 1; }
  ACTIVATE="source $TT_METAL_VENV/bin/activate; "
fi
FLAGS=(-e INKLING_WORK -e INKLING_HF_SD -e INKLING_SNAPSHOT -e INKLING_GOLDEN -e INKLING_EXPERTS)
for v in INKLING_VLLM_VENV INKLING_SRC_CACHE TT_METAL_VENV; do [ -n "${!v:-}" ] && FLAGS+=(-e "$v"); done
for v in $(compgen -e | grep -E '^(INK|PREFILL)_' || true); do FLAGS+=(-e "$v"); done
exec docker run --rm -i --network host --ipc host --user "$(id -u):$(id -g)" "${DEV[@]}" "${MOUNTS[@]}" \
  -e HOME="$INKLING_WORK/home" -e USER="${USER:-inkling}" -e LOGNAME="${USER:-inkling}" \
  -e TT_METAL_CACHE=/cache -e TT_METAL_HOME="$TTM" -e TT_METAL_RUNTIME_ROOT="$TTM" \
  -e ARCH_NAME=blackhole -e PYTHONPATH="$TTM:$TTM/ttnn:$TTM/tools:$TTM/tt_metal/tt-llk/tools/python:$REPO" \
  "${FLAGS[@]}" \
  -w "$REPO" --entrypoint /bin/bash "$INKLING_DEV_IMAGE" -lc "${ACTIVATE}$*"
