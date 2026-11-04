# Preprocessing runtime (twin IK replays, label export, cohort freezing): the root project's default
# dependencies on Python 3.13, CPU only. The repository is bind-mounted at /workspace at run time.
# Build context: the repository root, limited by preprocess.Dockerfile.dockerignore to the lock files.
# `python -m policy labels|cohort <config>` with <stage>.runner: docker (or --docker) builds it when missing.
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libegl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock .python-version /opt/umi-excavator/
RUN uv sync --project /opt/umi-excavator --frozen --no-dev && chmod -R a+rX /opt/python /opt/venv
ENV PATH=/opt/venv/bin:$PATH HOME=/tmp MUJOCO_GL=egl PYTHONUNBUFFERED=1
WORKDIR /workspace
CMD ["python", "-m", "policy", "--help"]
