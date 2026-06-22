FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel

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

# 升级 pip 并安装 uv（--break-system-packages 因为基础镜像的 Python 是 externally-managed）
RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel && \
    pip install --break-system-packages uv

WORKDIR /workspace/graspo

# --- 仅复制 pyproject.toml（及其引用的 README/LICENSE，因为有些构建后端可能需要） ---
COPY pyproject.toml README.md LICENSE ./

# --- 使用 uv 安装项目依赖（不包括项目本身）---
# 这一步会读取 pyproject.toml 中的 dependencies，生成临时锁定文件并安装
# 只要 pyproject.toml 不变，这一层就会被缓存，不会重新下载依赖
RUN uv pip install --system --break-system-packages \
    $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

# --- 复制源代码（只有这层及以下会在代码变动时重建）---
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY tests ./tests

# --- 以可编辑方式安装项目自身，但不再安装依赖（因为已经安装过了）---
RUN uv pip install --system --break-system-packages --no-deps -e .

# 设置入口
CMD ["graspo", "--help"]