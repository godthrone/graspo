# 基础镜像：Python 3.12 + torch 2.12.1 + CUDA 13.2
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system deps + uv (official standalone installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/graspo

# Layer 1: dependencies (cached when uv.lock AND pyproject.toml unchanged)
# pyproject.toml IS copied here — version is managed by setuptools-scm
# (dynamic = ["version"]), so version bumps don't change this file's hash
# and don't invalidate this layer.
# --system-site-packages inherits torch + torchvision from the base image.
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --system-site-packages .venv \
    && uv sync --frozen --no-dev --no-install-project

# Layer 2: source code
COPY src ./src
COPY samples ./samples

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

RUN uv sync --frozen --no-dev

ENV PATH="/workspace/graspo/.venv/bin:/root/.local/bin:$PATH"

ENTRYPOINT ["graspo"]
CMD ["--help"]
