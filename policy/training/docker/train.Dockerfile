# CLOAK / OpenPI pi0.5 training and serving runtime for Lambda GPU boxes: the root project (Python 3.13)
# with its cloak and cuda dependency groups.
# Build context: the repository root, limited by train.Dockerfile.dockerignore to pyproject.toml, uv.lock and
# .python-version. `python -m policy image <config>` (and train / eval / serve) builds it on the box with CLOAK_REV = policy.training.cloak.UPSTREAM_REVISION. The repository itself is
# bind-mounted at /workspace at run time, and the OpenPI weight cache at /cache/openpi.
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG CLOAK_REV
RUN test -n "$CLOAK_REV" && apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libgl1 libegl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock .python-version /opt/umi-excavator/
RUN uv sync --project /opt/umi-excavator --frozen --no-dev --group cloak --group cuda && chmod -R a+rX /opt/python /opt/venv
# backend.activate() checks this exact revision with git, so keep the clone's .git.
RUN git clone https://github.com/Stanford-TML/cloak.git /opt/cloak && \
    git -C /opt/cloak checkout --detach "$CLOAK_REV" && \
    git config --system --add safe.directory /opt/cloak && chmod -R a+rX /opt/cloak
ENV PATH=/opt/venv/bin:$PATH OPENPI_DATA_HOME=/cache/openpi HOME=/tmp MUJOCO_GL=egl PYTHONUNBUFFERED=1
WORKDIR /workspace
CMD ["python", "-m", "policy", "--help"]
