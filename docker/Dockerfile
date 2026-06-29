# 生产构建请锁定 SHA256 digest 替代 tag：
#   docker pull pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel
#   docker inspect pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel --format='{{index .RepoDigests 0}}'
# 将输出结果替换下方 FROM 行，例如：
#   FROM pytorch/pytorch@sha256:abc123...
FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel \
    && pip install --break-system-packages uv

WORKDIR /workspace/graspo

# Layer 1: dependencies (cached when pyproject.toml unchanged)
COPY pyproject.toml README.md LICENSE ./
RUN uv pip install --system --break-system-packages \
    $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

# Layer 2: source code
COPY src ./src
COPY configs ./configs
COPY data ./data

RUN uv pip install --system --break-system-packages --no-deps -e .

CMD ["graspo", "--help"]