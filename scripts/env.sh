# Shared configuration for scripts/*.sh. Every variable can be overridden from the calling shell.
# Export the path variables in your own shell too (see README, step 0): the Python tools that run outside
# these wrappers (hf download, tools/convert, tools/golden) read the same variables.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# tt-metal dev image for 79951bd4, pinned by digest (tag c10fd7391de8d4873c6d8b84bb430adfe330069f).
: "${INKLING_DEV_IMAGE:=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:47772e38d127584ac8d64b4632ebdd82e5cef574f787cdb319e8f687501535dc}"
: "${INKLING_DEVICE:=0}"
: "${INKLING_WORK:=$REPO/.work}"
: "${INKLING_HF_SD:=$INKLING_WORK/hf_sd}"
: "${INKLING_SNAPSHOT:=$INKLING_WORK/Inkling-2-Mini}"
: "${INKLING_GOLDEN:=$INKLING_WORK/golden}"
: "${INKLING_EXPERTS:=stream}"
case "$INKLING_DEV_IMAGE" in
  *@sha256:*) ;;
  *) [ -n "${ALLOW_UNPINNED:-}" ] || { echo "INKLING_DEV_IMAGE must be pinned by digest (got $INKLING_DEV_IMAGE); set ALLOW_UNPINNED=1 to override" >&2; exit 1; } ;;
esac
export INKLING_DEV_IMAGE INKLING_DEVICE INKLING_WORK INKLING_HF_SD INKLING_SNAPSHOT INKLING_GOLDEN INKLING_EXPERTS
