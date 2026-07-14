# 基础镜像锁定 SHA256 digest（非 tag，tag 可被覆盖）。
# 更新 digest 时执行：
#   docker pull pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel
#   docker inspect pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel --format='{{index .RepoDigests 0}}'
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel \
    && pip install --break-system-packages uv

WORKDIR /workspace/graspo

# Layer 1: dependencies (cached when uv.lock unchanged)
# pyproject.toml is NOT copied here — version is managed by setuptools-scm
# (git tags), so version bumps don't invalidate the dependency cache.
# --mount=type=cache,target=/root/.cache/uv persists uv's download cache
# across builds, so even if this layer is rebuilt (e.g. uv.lock changes),
# packages are read from local cache instead of re-downloaded from PyPI.
COPY uv.lock .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --frozen --no-dev

# Layer 2: source code + build metadata
# SETUPTOOLS_SCM_PRETEND_VERSION is derived from the build arg VERSION —
# set via build.sh which reads it from `git describe`. This allows the
# Docker build to work without a full git history (important for CI
# shallow clones and COPY-only contexts).
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY pyproject.toml ./

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

# Install project into the same venv as dependencies
# --no-dev keeps the production image lean (dev deps not needed at runtime)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# torchvision required by Qwen3.5 VideoProcessor import path (even text-only)
RUN uv pip install torchvision

# Use venv Python as the entry point so all deps are available
ENV PATH="/workspace/graspo/.venv/bin:$PATH"

ENTRYPOINT ["graspo"]
CMD ["--help"]