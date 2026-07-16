# 基础镜像：Python 3.12 + torch 2.12.1 + CUDA 13.2
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/graspo

# Layer 1: dependencies (cached when uv.lock AND pyproject.toml unchanged)
# pyproject.toml IS copied here — version is managed by setuptools-scm
# (dynamic = ["version"]), so version bumps don't change this file's hash
# and don't invalidate this layer.
# venv with --system-site-packages inherits torch from the base image,
# avoiding 2.5GB of CUDA downloads from PyPI.
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m venv --system-site-packages .venv \
    && .venv/bin/pip install --no-cache-dir \
      pydantic pyyaml safetensors "transformers>=5.9.0" pillow

# Layer 2: source code
COPY src ./src
COPY configs ./configs
COPY data ./data

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

RUN .venv/bin/pip install --no-cache-dir --no-deps -e .

ENV PATH="/workspace/graspo/.venv/bin:$PATH"

ENTRYPOINT ["graspo"]
CMD ["--help"]