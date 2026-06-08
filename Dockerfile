FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# --- System dependencies (cached after first build) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade setuptools wheel

WORKDIR /workspace/graspo

# --- GRASPO pip dependencies (cached as long as pyproject.toml deps don't change) ---
COPY pyproject.toml ./
RUN python -m pip install \
    "pyyaml>=6.0.0" \
    "safetensors>=0.4.0" \
    "transformers>=4.53.0,<5.0.0" \
    "pillow>=10.0.0"

# --- Source code (only these layers rebuild on code change, no network needed) ---
COPY README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY data ./data

# --- Editable install without dependency resolution (no network needed) ---
RUN python -m pip install -e . --no-deps

CMD ["graspo", "--help"]
