#!/bin/bash
# Check a locally built serving image against the release commit, then (only with CONFIRM=yes) push it to GHCR.
#   GH_OWNER        GitHub owner in the source label (default StableFluffy)
#   GHCR_NAMESPACE  registry namespace, lowercase (default stablefluffy)
#   VERSION         OCI tag and version label (default 0.1.0)
#   RELEASE_REF     inkling-tt ref the image must come from (default v$VERSION; this checkout must contain it)
#   SRC_IMAGE       local image (default inkling-2-mini-tt:$VERSION)
#   EXPECTED_ID     optional: the image ID that was smoke-tested; the push is refused for any other image
#   CONFIRM         "yes" to tag and push; anything else runs the checks only
# Requires `docker login ghcr.io` with a token that has write:packages for the push.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"
GH_OWNER=${GH_OWNER:-StableFluffy}
GHCR_NAMESPACE=${GHCR_NAMESPACE:-stablefluffy}
VERSION=${VERSION:-0.1.0}
RELEASE_REF=${RELEASE_REF:-v$VERSION}
SRC_IMAGE=${SRC_IMAGE:-inkling-2-mini-tt:$VERSION}
DST=ghcr.io/$GHCR_NAMESPACE/inkling-tt:$VERSION
PIN=79951bd44b7484a136b3619e0d81f26baa8a1936
fail() { echo "FAIL: $*" >&2; exit 1; }
label() { docker image inspect "$SRC_IMAGE" --format "{{ index .Config.Labels \"$1\" }}"; }
in_image() { docker run --rm --network none --entrypoint /bin/sh "$SRC_IMAGE" -c "$1"; }

echo "== pre-push checks for $SRC_IMAGE -> $DST"
[ "$GHCR_NAMESPACE" = "$(printf %s "$GHCR_NAMESPACE" | tr '[:upper:]' '[:lower:]')" ] || fail "GHCR_NAMESPACE must be lowercase"
docker image inspect "$SRC_IMAGE" >/dev/null 2>&1 || fail "$SRC_IMAGE not found locally"
ID=$(docker image inspect "$SRC_IMAGE" --format '{{.Id}}')
[ -z "${EXPECTED_ID:-}" ] || [ "$ID" = "$EXPECTED_ID" ] || fail "image ID $ID is not the smoke-tested $EXPECTED_ID"
EXPECTED_REV=$(git -C "$REPO" rev-parse --verify "$RELEASE_REF^{commit}") || fail "$RELEASE_REF not found in $REPO"
check() { [ "$(label "$1")" = "$2" ] || fail "label $1 is '$(label "$1")', expected '$2'"; }
check org.opencontainers.image.revision "$EXPECTED_REV"
check org.opencontainers.image.source "https://github.com/$GH_OWNER/inkling-tt"
check org.opencontainers.image.version "$VERSION"
check org.opencontainers.image.licenses Apache-2.0
check org.opencontainers.image.base.name "$INKLING_DEV_IMAGE"
check tt-metal.commit "$PIN"
echo "  labels ok: revision $EXPECTED_REV ($RELEASE_REF)"

in_image 'test -f /opt/inkling/LICENSE && test -f /opt/inkling/NOTICE && test -f /opt/tt-metal/LICENSE && test -f /opt/tt-metal/NOTICE' \
  || fail "LICENSE/NOTICE missing under /opt/inkling or /opt/tt-metal"
LIST='find code/inkling_tt code/serve LICENSE NOTICE -type f -not -path "*/__pycache__/*" | LC_ALL=C sort | xargs sha256sum'
T=$(mktemp -d)
mkdir -p "$T/code"
git -C "$REPO" archive --format=tar "$EXPECTED_REV" inkling_tt serve | tar -x -C "$T/code"
git -C "$REPO" archive --format=tar "$EXPECTED_REV" LICENSE NOTICE | tar -x -C "$T"
WANT=$(cd "$T" && sh -c "$LIST")
GOT=$(in_image "cd /opt/inkling && $LIST")
[ "$WANT" = "$GOT" ] || fail "code in the image differs from git archive $EXPECTED_REV:
$(diff <(echo "$WANT") <(echo "$GOT") | head -20)"
echo "  code ok: /opt/inkling matches git archive $EXPECTED_REV ($(echo "$GOT" | wc -l) files)"

SCAN='grep -rIoE "/home/[a-z][a-z0-9_-]*|/Users/[A-Za-z]+|BEGIN (RSA|OPENSSH|EC) PRIVATE|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}" /opt/inkling'
rc=0; HITS=$(in_image "$SCAN") || rc=$?
[ "$rc" -le 1 ] || fail "source scan failed (exit $rc)"
HITS=$(echo "$HITS" | grep -vE '^$|:/home/inkling$' || true)
[ -z "$HITS" ] || fail "host paths or secrets under /opt/inkling:
$HITS"
# tt-metal's own sources contain example /home/... paths; look for this build host's home and user instead.
SCAN="grep -rlaF -e '$HOME/' -e '/home/$(id -un)/' -e '/Users/$(id -un)/' /opt/tt-metal"
rc=0; HITS=$(in_image "$SCAN") || rc=$?
[ "$rc" -le 1 ] || fail "tt-metal scan failed (exit $rc)"
HITS=$(echo "$HITS" | grep -v '^$' || true)
[ -z "$HITS" ] || fail "build-host paths under /opt/tt-metal ($(echo "$HITS" | wc -l) files), e.g.:
$(echo "$HITS" | head -5)"
echo "  scans ok: no host paths or credentials under /opt/inkling and /opt/tt-metal"
docker image inspect "$SRC_IMAGE" --format '  id {{.Id}}  size {{.Size}} B  created {{.Created}}'

if [ "${CONFIRM:-}" != yes ]; then
  echo "checks passed; not pushing (set CONFIRM=yes to tag and push $DST)"
  exit 0
fi
if docker manifest inspect "$DST" >/dev/null 2>&1; then
  fail "$DST already exists in the registry; refusing to overwrite a published tag"
fi
docker tag "$SRC_IMAGE" "$DST"
docker push "$DST"
docker image inspect "$DST" --format 'pushed {{json .RepoDigests}}'
echo "The GHCR package is private after its first push: link it to the repository and set its visibility (PUBLISH_CHECKLIST)."
