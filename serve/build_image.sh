#!/bin/bash
# Build the serving image from a committed inkling-tt revision and a built tt-metal tree. Does not push.
#   TT_METAL_HOME  clean tt-metal checkout at the pinned commit, built with scripts/build_tt_metal.sh (required)
#   REF            inkling-tt commit or tag to package (default HEAD). Only `git archive REF` reaches the image;
#                  uncommitted or untracked files in this checkout never do
#   TAG            local image tag (default inkling-2-mini-tt:dev)
#   VERSION        org.opencontainers.image.version label (default 0.1.1-dev)
#   SOURCE_URL     org.opencontainers.image.source label (default https://github.com/StableFluffy/inkling-tt)
#   INKLING_DEV_IMAGE  base image, pinned by digest (default from scripts/env.sh)
#   CTX            staging directory (default .image-ctx in this checkout)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"
: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal tree at the pinned commit}"
TT_METAL_COMMIT_PIN=79951bd44b7484a136b3619e0d81f26baa8a1936
CTX=${CTX:-$REPO/.image-ctx}
TAG=${TAG:-inkling-2-mini-tt:dev}
VERSION=${VERSION:-0.1.1-dev}
SOURCE_URL=${SOURCE_URL:-https://github.com/StableFluffy/inkling-tt}
REF=${REF:-HEAD}
fail() { echo "build_image: $*" >&2; exit 1; }

INKLING_COMMIT=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || fail "cannot resolve $REF"
EPOCH=$(git -C "$REPO" log -1 --format=%at "$INKLING_COMMIT")
[ -z "$(git -C "$REPO" status --porcelain)" ] || echo "build_image: note: the working tree is not clean; the image gets $INKLING_COMMIT only" >&2

COMMIT=$(git -C "$TT_METAL_HOME" rev-parse HEAD)
if [ "$COMMIT" != "$TT_METAL_COMMIT_PIN" ] && [ -z "${ALLOW_UNPINNED:-}" ]; then
  fail "tt-metal is at $COMMIT, expected $TT_METAL_COMMIT_PIN (set ALLOW_UNPINNED=1 to override)"
fi
[ -z "$(git -C "$TT_METAL_HOME" status --porcelain --untracked-files=no)" ] || fail "tt-metal tree has modified tracked files"
[ -z "$(git -C "$TT_METAL_HOME" submodule status --recursive | grep -v '^ ' || true)" ] || fail "tt-metal submodules are not at their recorded commits"
UNTRACKED=$(git -C "$TT_METAL_HOME" ls-files --others --exclude-standard)
[ -z "$UNTRACKED" ] || fail "tt-metal tree has untracked files (use a dedicated clean checkout):
$UNTRACKED"
[ -f "$TT_METAL_HOME/LICENSE" ] && [ -f "$TT_METAL_HOME/NOTICE" ] || fail "tt-metal LICENSE/NOTICE missing"

mkdir -p "$CTX"
rsync -a --delete --link-dest="$TT_METAL_HOME/" \
  --include '/runtime/***' --exclude /.git --exclude /.cpmcache --exclude /tech_reports --exclude /docs --exclude /generated \
  --exclude CMakeFiles --exclude '*.o' --exclude __pycache__ \
  "$TT_METAL_HOME/" "$CTX/tt-metal/"
# Compiled-in paths must not carry the host layout (build with scripts/build_tt_metal.sh, which builds at /opt/tt-metal).
for p in "$TT_METAL_HOME" "$HOME"; do
  rc=0; HITS=$(grep -rlaF -- "$p/" "$CTX/tt-metal") || rc=$?
  [ "$rc" -le 1 ] || fail "host-path scan of the staged tt-metal tree failed (grep exit $rc)"
  [ -z "$HITS" ] || fail "staged tt-metal files contain the host path $p (rebuild with scripts/build_tt_metal.sh):
$(echo "$HITS" | head -5)"
done

STAGE=$(mktemp -d)
git -C "$REPO" archive --format=tar "$INKLING_COMMIT" inkling_tt serve LICENSE NOTICE | tar -x -C "$STAGE"
find "$STAGE" -exec touch -h -d "@$EPOCH" {} +
rsync -a --delete --checksum "$STAGE/" "$CTX/code/"
CODE_SHA=$(cd "$CTX/code" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
printf 'inkling-tt %s\ntt-metal %s\nbase %s\ncode-sha256 %s\n' "$INKLING_COMMIT" "$COMMIT" "$INKLING_DEV_IMAGE" "$CODE_SHA" > "$CTX/build-record.txt"
cat "$CTX/build-record.txt"

docker build --provenance=false --sbom=false -f "$REPO/serve/Dockerfile" \
  --build-arg BASE_IMAGE="$INKLING_DEV_IMAGE" \
  --build-arg TT_METAL_COMMIT="$COMMIT" --build-arg INKLING_VERSION="$VERSION" \
  --build-arg INKLING_COMMIT="$INKLING_COMMIT" --build-arg SOURCE_URL="$SOURCE_URL" \
  -t "$TAG" "$CTX"
docker image inspect "$TAG" --format 'built {{.Id}} revision {{ index .Config.Labels "org.opencontainers.image.revision" }}'
