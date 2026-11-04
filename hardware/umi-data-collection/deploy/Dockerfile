# UMI recording dashboard for the Raspberry Pi (arm64, Raspberry Pi OS bookworm).
# Build with the repository root as context (the page lives in hub/web/data-collection):
#   docker build -f hardware/umi-data-collection/deploy/Dockerfile -t umi-dashboard:latest .
# From a Mac, for the Pi:
#   docker buildx build --platform linux/arm64 -f hardware/umi-data-collection/deploy/Dockerfile -t umi-dashboard:latest .
# Debian bookworm matches the Pi's system Python 3.11 and gives python3-av with libx264.

FROM debian:bookworm-slim AS capture
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev linux-libc-dev \
 && rm -rf /var/lib/apt/lists/*
COPY hardware/umi-data-collection/recorder/capture.c /src/capture.c
RUN gcc -O2 -Wall -Wextra -Werror -std=c11 /src/capture.c -o /src/capture

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-smbus2 python3-pil python3-av i2c-tools v4l-utils \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY hardware/umi-data-collection/recorder/ /app/recorder/
COPY --from=capture /src/capture /app/recorder/capture
# The page: hub/web/data-collection (its theme.css symlink is excluded by the ignore file) + hub/web/theme.css.
COPY hub/web/data-collection/ /app/recorder/static/
COPY hub/web/theme.css /app/recorder/static/theme.css
RUN chmod -R a+rX /app \
 && python3 -c "import smbus2, PIL; from recorder.video_export import check_video_support; check_video_support()" \
 && python3 -m compileall -q /app
# The recorder's state lives on the /data volume (the Pi's ~/umi-dashboard): access-token, imus.json,
# time-calibration.json and recordings/. A missing imus.json or time-calibration.json reads as absent.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UMI_DATA_DIR=/data
EXPOSE 8000
ENTRYPOINT ["/bin/sh", "-c", "mkdir -p -m 700 /data/recordings && exec python3 -m recorder \"$@\"", "recorder"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--token-file", "/data/access-token"]
