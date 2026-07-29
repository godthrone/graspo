# 基础镜像：Python 3.12 + torch 2.12.1 + CUDA 13.2
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (official standalone binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace/graspo

# Layer 1: dependencies (cached when uv.lock AND pyproject.toml unchanged)
# pyproject.toml IS copied here — version is managed by setuptools-scm
# (dynamic = ["version"]), so version bumps don't change this file's hash
# and don't invalidate this layer.
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Layer 2: source code
COPY src ./src
COPY samples ./samples

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

RUN uv sync --frozen --no-dev

ENV PATH="/workspace/graspo/.venv/bin:$PATH"

ENTRYPOINT ["graspo"]
CMD ["--help"]
