# 基础镜像：Python 3.12 + torch 2.12.1 + CUDA 13.2
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    -o Acquire::Retries=10 -o Acquire::http::Timeout=30 \
    && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=10 -o Acquire::http::Timeout=30 \
    git curl ca-certificates libjpeg-turbo8 libpng16-16 \
    && curl --retry 5 --retry-max-time 120 -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/graspo

# Layer 1: dependencies (cached when uv.lock AND pyproject.toml unchanged)
# pyproject.toml pins torch==2.12.1, torchvision==0.27.1 matching the base image.
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Layer 2: source code
COPY src ./src
COPY samples ./samples

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

RUN uv sync --frozen --no-dev

ENV PATH="/workspace/graspo/.venv/bin:/root/.local/bin:$PATH"

ENTRYPOINT ["graspo"]
CMD ["--help"]
