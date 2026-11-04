# umi-excavator-processing: the ORB-SLAM3 reconstruction runtime. Pinned ORB-SLAM3 (+ Pangolin) and the repository's
# orbslam_probe, plus Python 3.13 with the `processing` dependency group and the policy/processing/ package (with
# policy/data/, the dataset-location leaf it imports).
#
# Build context: the repository root, limited by Dockerfile.dockerignore next to this file:
#   docker build -f policy/processing/docker/Dockerfile -t umi-excavator-processing .     (or: bash policy/processing/docker/build.sh)
# Run (the data directory at /data; the image calls the native runners directly, no Docker-in-Docker):
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data:/data" umi-excavator-processing \
#     python -m policy.processing.reconstruct_orbslam --manifest /data/... --data-root /data/... ...
# A repository bind-mounted at /workspace (with -w /workspace) takes precedence over the baked policy/processing/.
ARG BASE=ubuntu:22.04

FROM ${BASE} AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=8
ARG ORB_REV=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
ARG PANGOLIN_REV=1c620ff72a93854d1ab3cb42c6b7360073926673
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates pkg-config \
    libopencv-dev libeigen3-dev libssl-dev libboost-serialization-dev \
    libglew-dev libgl1-mesa-dev libegl1-mesa-dev libpng-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/Pangolin && cd /opt/Pangolin && \
    git remote add origin https://github.com/stevenlovegrove/Pangolin.git && \
    git fetch --depth 1 origin "${PANGOLIN_REV}" && git checkout --detach FETCH_HEAD && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF \
      -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_VIDEO=OFF && \
    cmake --build build --parallel "${BUILD_JOBS}" && cmake --install build && ldconfig
RUN git init /opt/ORB_SLAM3 && cd /opt/ORB_SLAM3 && \
    git remote add origin https://github.com/UZ-SLAMLab/ORB_SLAM3.git && \
    git fetch --depth 1 origin "${ORB_REV}" && git checkout --detach FETCH_HEAD
# g2o stays on C++11, matching ORB-SLAM3: Eigen aligned allocation differs in C++17 and crashed bundle adjustment.
# Only the libORB_SLAM3 library is built; orbslam_probe (below) is the one executable linked against it.
RUN cmake -S /opt/ORB_SLAM3/Thirdparty/DBoW2 -B /opt/ORB_SLAM3/Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/ORB_SLAM3/Thirdparty/DBoW2/build --parallel "${BUILD_JOBS}" && \
    cmake -S /opt/ORB_SLAM3/Thirdparty/g2o -B /opt/ORB_SLAM3/Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON && \
    cmake --build /opt/ORB_SLAM3/Thirdparty/g2o/build --parallel "${BUILD_JOBS}" && \
    tar -xf /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C /opt/ORB_SLAM3/Vocabulary && \
    cmake -S /opt/ORB_SLAM3 -B /opt/ORB_SLAM3/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/ORB_SLAM3/build --target ORB_SLAM3 --parallel "${BUILD_JOBS}"
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib
# The repository's probe, linked against that build (portable: no -march=native).
COPY policy/processing/slam/build_probe.sh policy/processing/slam/orbslam_probe.cc /src/processing/slam/
RUN bash /src/processing/slam/build_probe.sh /opt/umi-excavator/bin/orbslam_probe
# Stage the runtime tree and list the Ubuntu packages that own every shared library it links. No pipefail:
# the pipelines end in explicit checks (missing libraries, an empty package list).
# hadolint ignore=DL4006,SC2086
RUN set -eu; mkdir -p /stage/opt/ORB_SLAM3/Vocabulary \
      /stage/opt/ORB_SLAM3/Thirdparty/DBoW2 /stage/opt/ORB_SLAM3/Thirdparty/g2o; \
    cp -a /opt/ORB_SLAM3/lib /opt/ORB_SLAM3/LICENSE /stage/opt/ORB_SLAM3/; \
    cp -a /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /stage/opt/ORB_SLAM3/Vocabulary/; \
    cp -a /opt/ORB_SLAM3/Thirdparty/DBoW2/lib /stage/opt/ORB_SLAM3/Thirdparty/DBoW2/; \
    cp -a /opt/ORB_SLAM3/Thirdparty/g2o/lib /stage/opt/ORB_SLAM3/Thirdparty/g2o/; \
    cp -a /opt/umi-excavator /stage/opt/; \
    mkdir -p /stage/usr/local/lib && cp -a /usr/local/lib/libpangolin* /stage/usr/local/lib/; \
    objects=$(find /stage -type f \( -perm -u+x -o -name '*.so*' \)); \
    if ldd $objects | grep 'not found'; then exit 1; fi; \
    ldd $objects | awk '$2 == "=>" && $3 ~ /^\// {print $3}' | sort -u | while read -r lib; do \
      dpkg -S "*/$(basename "$lib")" 2>/dev/null || true; done \
      | sed 's/: \/.*//' | tr ',' '\n' | sed 's/:.*//; s/ //g' | sort -u > /stage/runtime-packages.txt; \
    grep -q '^libopencv-core' /stage/runtime-packages.txt; cat /stage/runtime-packages.txt

FROM ${BASE}
ARG DEBIAN_FRONTEND=noninteractive
COPY --from=build /stage/runtime-packages.txt /tmp/runtime-packages.txt
# ffmpeg re-encodes the review overlay videos to H.264 (policy/processing/slam/report.py); without it they stay mp4v.
RUN apt-get update && xargs -a /tmp/runtime-packages.txt apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg && rm -rf /var/lib/apt/lists/* /tmp/runtime-packages.txt
COPY --from=build /stage/opt/ /opt/
COPY --from=build /stage/usr/local/lib/ /usr/local/lib/
RUN printf '%s\n' /usr/local/lib /opt/ORB_SLAM3/lib /opt/ORB_SLAM3/Thirdparty/DBoW2/lib \
      /opt/ORB_SLAM3/Thirdparty/g2o/lib > /etc/ld.so.conf.d/umi-excavator-processing.conf && ldconfig
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock .python-version /opt/umi-excavator/
RUN uv sync --project /opt/umi-excavator --frozen --only-group processing && chmod -R a+rX /opt/python /opt/venv
COPY configs/camera.json /opt/umi-excavator/configs/camera.json
COPY policy/data /opt/umi-excavator/policy/data
COPY policy/processing /opt/umi-excavator/policy/processing
# Inside this image policy/processing/slam/orbslam.py picks these binaries instead of Docker.
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/umi-excavator PYTHONUNBUFFERED=1 HOME=/tmp \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    ORBSLAM3_PROBE=/opt/umi-excavator/bin/orbslam_probe \
    ORBSLAM3_VOCABULARY=/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt
RUN python -m policy.processing.docker.check_image
WORKDIR /data
CMD ["python", "-m", "policy.processing.slam", "--help"]
