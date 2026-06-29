# 基础镜像锁定 SHA256 digest（非 tag，tag 可被覆盖）。
# 更新 digest 时执行：
#   docker pull pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel
#   docker inspect pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel --format='{{index .RepoDigests 0}}'
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

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