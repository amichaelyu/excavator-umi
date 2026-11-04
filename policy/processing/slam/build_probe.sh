#!/usr/bin/env bash
# Compile orbslam_probe against the ORB-SLAM3 build in /opt/ORB_SLAM3. policy/processing/docker/Dockerfile runs this
# in its build stage (the runtime image has no compiler); PROBE_SOURCE overrides the source location.
set -euo pipefail
output=${1:?Usage: build_probe.sh OUTPUT}
source=${PROBE_SOURCE:-$(cd "$(dirname "$0")" && pwd)/orbslam_probe.cc}
mkdir -p "$(dirname "$output")"
# shellcheck disable=SC2046  # pkg-config output is a list of flags
c++ -O2 -std=c++11 -DCOMPILEDWITHC11 -w \
  -I/opt/ORB_SLAM3 -I/opt/ORB_SLAM3/include \
  -I/opt/ORB_SLAM3/include/CameraModels -I/opt/ORB_SLAM3/Thirdparty/Sophus \
  -I/usr/include/eigen3 $(pkg-config --cflags opencv4) \
  "$source" -o "$output" \
  -L/opt/ORB_SLAM3/lib -lORB_SLAM3 $(pkg-config --libs opencv4) \
  -L/usr/local/lib -lpangolin -lGL -lGLEW -lpthread \
  -L/opt/ORB_SLAM3/Thirdparty/DBoW2/lib -lDBoW2 \
  -L/opt/ORB_SLAM3/Thirdparty/g2o/lib -lg2o -lboost_serialization -lcrypto
