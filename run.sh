#!/usr/bin/env bash
# GRASPO 训练启动入口（防呆版）
#
# 唯一入口：从 GitHub clone 后，这是启动训练的唯一方式。
# 用法:
#   bash run.sh my_config.yaml                # 训练（自动选空闲 GPU）
#   bash run.sh my_config.yaml --smoke        # 冒烟：跑 1 步验证环境后停止
#   GPU_IDS=4,5 bash run.sh my_config.yaml    # 指定 GPU
#
# 防呆设计（宪法 §2.3）:
#   1. 自动选择空闲 GPU（nvidia-smi 检测显存占用为 0 的卡），无需手动数卡
#   2. 只传 --gpus device=<ids>，绝不注入 CUDA_VISIBLE_DEVICES——
#      两者混用会导致 NCCL 初始化死锁（实测卡死）
#   3. 固定 --ipc=host --shm-size=16g（NCCL 共享内存必需）
#   4. 镜像 tag 自动取 git describe，不硬编码版本
#   5. --smoke 走 CLI 参数（graspo launch --smoke），不修改用户 config 文件
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 参数解析 ────────────────────────────────────────────────────────────────
CONFIG="${1:-}"
SMOKE="${2:-}"
if [ -z "$CONFIG" ]; then
    echo "用法: bash run.sh <config.yaml> [--smoke]"
    echo "      GPU_IDS=4,5 bash run.sh <config.yaml>   # 指定 GPU"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: 配置文件不存在: $CONFIG"
    exit 1
fi
CONFIG_ABS="$(realpath "$CONFIG")"

# ── 防呆 1: 自动选择空闲 GPU ────────────────────────────────────────────────
if [ -z "${GPU_IDS:-}" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi 不可用，请确认 GPU 驱动已安装"
        exit 1
    fi
    GPU_IDS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F', ' '$2==0 {printf "%s%s", sep, $1; sep=","}')"
    if [ -z "$GPU_IDS" ]; then
        echo "ERROR: 没有空闲 GPU（所有卡都被占用）"
        echo "  可用: GPU_IDS=4,5 bash run.sh $CONFIG 指定要用的卡"
        exit 1
    fi
    echo "自动选择 GPU: $GPU_IDS"
else
    echo "使用指定 GPU: $GPU_IDS"
fi

# ── 防呆 4: 镜像 tag 自动取 git 版本 ───────────────────────────────────────
VERSION="$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.14.3")"
IMAGE="graspo:${VERSION}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "镜像 $IMAGE 不存在，先构建（bash docker/build.sh 或 docker/Dockerfile.from-local）"
    exit 1
fi

# ── 启动容器 ────────────────────────────────────────────────────────────────
CONTAINER_NAME="graspo-$(basename "$CONFIG" .yaml)"
CONFIG_DIR="$(dirname "$CONFIG_ABS")"
EXTRA_ARGS=()
if [ "$SMOKE" = "--smoke" ]; then
    EXTRA_ARGS+=(--smoke)
    echo "冒烟模式：跑 1 步验证环境"
fi

echo "启动容器: $CONTAINER_NAME"
echo "  镜像: $IMAGE"
echo "  配置: $CONFIG_ABS"
echo "  GPU:  $GPU_IDS"

docker run -d --name "$CONTAINER_NAME" \
    --gpus "device=$GPU_IDS" \
    --ipc=host --shm-size=16g \
    -v "$CONFIG_DIR:/data/configs" \
    -v "$ROOT_DIR/models:/workspace/graspo/models" \
    "$IMAGE" \
    launch --config "/data/configs/$(basename "$CONFIG")" "${EXTRA_ARGS[@]}"

echo "已启动。查看日志: docker logs -f $CONTAINER_NAME"
echo "停止训练:  docker stop $CONTAINER_NAME"
