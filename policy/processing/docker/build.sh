#!/usr/bin/env bash
# Build the processing image umi-excavator-processing (Dockerfile, repository root as context).
# BUILD_JOBS sets compiler jobs (default 8); PLATFORM (e.g. linux/amd64) is passed to --platform.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
engine=(docker)
if ! docker info >/dev/null 2>&1; then
  engine=(sudo -n docker)
fi
case "${1:-processing}" in
  processing|slam|all) ;;
  *) echo 'Usage: build.sh [processing]' >&2; exit 2 ;;
esac
flags=(--progress=plain --build-arg "BUILD_JOBS=${BUILD_JOBS:-8}")
[[ -n "${PLATFORM:-}" ]] && flags+=(--platform "$PLATFORM")
"${engine[@]}" build "${flags[@]}" -f "$here/Dockerfile" -t umi-excavator-processing "$root"
