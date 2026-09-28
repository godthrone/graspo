# GRASPO 产品镜像 —— 公开 base + 显式 PyTorch cu130 官方轮子索引（宪法 §4/§6/§14.3/§15.1）。
#
# **为什么是公开 base**：本仓库要对外发布，`FROM <内网镜像>` 外人拉不到（违反 §4 零步上手
# 与 §6 环境可复现），且内网镜像名属于环境信息，不得进入 tracked 文件（§15.1）。
# 「单一血统」由**版本**保证（下面的精确 pin + 版本门禁），不由 base 镜像相同来保证。
#
# **CUDA 渠道**：torch/torchvision 从 PyTorch 官方 cu130 轮子索引取，得到带 `+cu130`
# 本地标签的轮子；其余依赖从 PyPI 取。base 里预装的 torch 2.12.1 会被显式 pin 覆盖。
#
# ⚠️ 状态：**待上机/发布阶段验证**（不阻塞本期 —— 本期唯一获批执行环境是既有实验镜像）。
#    只读边界内无法确证 `download.pytorch.org` 从构建机可达，故本配方尚未定稿。
#    发布阶段验证命令（单一路径锁卡；宿主侧先断言）：
#      python3 scripts/gpu_lock_guard.py --visible 0 || exit 1
#      docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 <image> \
#        python /workspace/graspo/scripts/check_env_versions.py --local
#      docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 <image> \
#        python -c "import torch; print(torch.__version__, torch.version.cuda)"
#
# 代理与 pip 源（§14.3 取值规则）——**只构建时需要、运行时不需要**：
#   · **代理 ⇒ BuildKit secret**（`RUN --mount=type=secret`）。class A / 公开仓库
#     **禁止** `--build-arg` 路线（取值会留在 `docker history --no-trunc`，即使从不写 ENV），
#     也**禁止**写进 ENV（§7.1）。`bash docker/build.sh` 会把宿主 HTTP_PROXY/HTTPS_PROXY/NO_PROXY
#     转成 secret；不传即直连公网。读取逻辑见 `docker/proxy-env.sh`（单一真相源）。
#   · **pip 源 ⇒ 直接内联**：`download.pytorch.org`（cu130 官方轮子索引）与 `pypi.org/simple`
#     都是**公开地址**（非机密、非内网），不需要配置面；做成 `ARG` 只会多出一条可被
#     `--build-arg` 覆盖的渠道，反而与 §14.3 的取值规则冲突。

# 公开 base：pytorch/pytorch，digest 精确锁定（内容寻址、不可变，强于 tag）。
FROM pytorch/pytorch@sha256:91629be68c98490a987b243b176757e8549336b98296933448fe496427118569

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 代理读取器（§14.3）：每个需要网络的 RUN 用 `. /usr/local/bin/proxy-env.sh` 引入。
COPY docker/proxy-env.sh /usr/local/bin/proxy-env.sh

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    sh -eu -c '. /usr/local/bin/proxy-env.sh; \
    apt-get update \
      -o Acquire::Retries=10 -o Acquire::http::Timeout=30; \
    apt-get install -y --no-install-recommends \
      -o Acquire::Retries=10 -o Acquire::http::Timeout=30 \
      git curl ca-certificates; \
    curl --retry 5 --retry-max-time 120 -LsSf https://astral.sh/uv/install.sh | sh; \
    rm -rf /var/lib/apt/lists/*'

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/graspo

# ── Layer 1: 运行时依赖，显式精确锁定（§6）────────────────────────────────────
# 锁定值 = 实测血统；与 pyproject.toml 的声明由 scripts/check_env_versions.py
# 自动派生包集合并逐包比对（§2 防呆）。
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./

# ★ 下面两条 `sh -c '…'` 的**闭合单引号必须与最后一个版本号之间留一个空格**：门禁
# `scripts/check_env_versions.py::parse_dockerfile_pins()` 按空白切 token 再用
# `name==version` 正则取值，引号若紧贴 `torchvision==0.26.0` 会被并进版本（`0.26.0'`）
# ⇒ 声明点比对报不一致；且本文件 :104 的 `RUN python scripts/check_env_versions.py --local`
# 会在镜像内解析本文件 ⇒ 不止单测红，**镜像构建会直接失败**。改动此行前请先读该函数。
# torch/torchvision：走 cu130 官方索引（**公开常量，内联**），拿到 `+cu130` 轮子（渠道由断言兜底）。
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    sh -eu -c '. /usr/local/bin/proxy-env.sh; \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.0 torchvision==0.26.0 '

# 其余运行时依赖：走 PyPI（**公开常量，内联**）。闭合引号前同样留空格（见上方说明）。
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    sh -eu -c '. /usr/local/bin/proxy-env.sh; \
    pip install --no-cache-dir --index-url https://pypi.org/simple/ \
      transformers==5.12.1 pydantic==2.11.10 pyyaml==6.0.3 safetensors==0.8.0 \
      pillow==11.3.0 numpy==1.26.4 ms-swift==4.5.3 '

# CUDA 渠道断言（fail-closed）：release 段比较会剔除 `+cu130`，渠道单独断言。
RUN python -c "import torch, sys; cuda=torch.version.cuda or ''; print('torch', torch.__version__, 'cuda', cuda); sys.exit('CUDA CHANNEL FAILED: '+cuda) if not cuda.startswith('13.0') else None"

# ── Layer 2: 源码与项目安装 ───────────────────────────────────────────────────
COPY src ./src
COPY samples ./samples
COPY scripts/check_env_versions.py ./scripts/check_env_versions.py
COPY docker/Dockerfile ./docker/Dockerfile
COPY docker/proxy-env.sh ./docker/proxy-env.sh

ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
# 代码单向流动（§14.4）追溯锚点：`cat /opt/graspo-image/VERSION` 可确认构建版本。
RUN mkdir -p /opt/graspo-image && printf '%s\n' "${VERSION}" > /opt/graspo-image/VERSION

# --no-deps：依赖已由上面的显式 pin 覆盖，项目本身只做无依赖安装
# （避免 uv 重新解析依赖树把 torch 换成非 cu130 轮子，造成渠道分叉）。
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    sh -eu -c '. /usr/local/bin/proxy-env.sh; \
    uv pip install --system --break-system-packages --no-deps -e .'

# ── 构建期版本门禁（fail-closed）：与运行时同一脚本、自动派生包集合 ────────────
RUN python scripts/check_env_versions.py --local

# Smoke test: ensure the package imports and CLI is functional before the image is used.
RUN graspo --help

ENTRYPOINT ["graspo"]
CMD ["--help"]
