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

# Layer 1: dependencies (cached when pyproject.toml and uv.lock unchanged)
# --no-install-project resolves only the locked dependencies, leaving the
# project itself for Layer 2 (so the source COPY doesn't break the cache).
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --no-install-project --frozen

# Layer 2: source code
COPY src ./src
COPY configs ./configs
COPY data ./data

# Install project into the same venv as dependencies (not system Python)
RUN uv sync --frozen

# torchvision required by Qwen3.5 VideoProcessor import path (even text-only)
# UV_SYSTEM_PYTHON=1 is set above, must override to install into venv
RUN UV_SYSTEM_PYTHON=0 uv pip install torchvision

# Use venv Python as the entry point so all deps are available
ENV PATH="/workspace/graspo/.venv/bin:$PATH"

CMD ["graspo", "--help"]