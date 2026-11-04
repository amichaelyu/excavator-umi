#!/usr/bin/env bash
# Run a command in the processing image with $UMI_DATA_DIR (default: the repository's dataset/) at /data and policy/processing/
# read-only at /workspace, as the calling user, without network access.
#   run.sh processing COMMAND...   umi-excavator-processing (alias: slam)
set -euo pipefail
case "${1:-}" in
  processing|slam) image=${PROCESSING_IMAGE:-umi-excavator-processing} ;;
  *) image= ;;
esac
if [[ $# -lt 2 || -z $image ]]; then
  echo 'Usage: UMI_DATA_DIR=/path/to/data run.sh processing COMMAND [ARGS...]' >&2
  exit 2
fi
shift
data_dir=${UMI_DATA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)/dataset}
mkdir -p "$data_dir"
data_dir=$(cd "$data_dir" && pwd)
code_dir=$(cd "$(dirname "$0")/.." && pwd)
engine=(docker)
if ! docker info >/dev/null 2>&1; then
  engine=(sudo -n docker)
fi
exec "${engine[@]}" run --rm --init --network=none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e MPLCONFIGDIR=/tmp/matplotlib -e ROS_HOME=/tmp/ros \
  --mount "type=bind,src=$data_dir,dst=/data" \
  --mount "type=bind,src=$code_dir,dst=/workspace,readonly" \
  "$image" "$@"
