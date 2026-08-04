#!/usr/bin/env bash
# GRASPO 训练启动入口（防呆版）
#
# 唯一入口：从 GitHub clone 后，这是启动训练的唯一方式。
# 用法:
#   bash run.sh my_config.yaml                # 训练（自动选空闲 GPU）
#   bash run.sh my_config.yaml --smoke        # 冒烟：跑 1 步验证环境后停止
#   bash run.sh my_config.yaml --gpus 4,5     # 指定 GPU
#   bash run.sh my_config.yaml --model-dir /path/to/models          # 模型挂载源
#
# 防呆设计:
#   1. 自动选择空闲 GPU（nvidia-smi 检测显存占用为 0 的卡），无需手动数卡
#   2. 只传 --gpus device=<ids>，绝不注入 CUDA_VISIBLE_DEVICES——
#      两者混用会导致 NCCL 初始化死锁（实测卡死）
#   3. 固定 --ipc=host --shm-size=16g（NCCL 共享内存必需）
#   4. 镜像 tag 自动取 git describe，不硬编码版本
#   5. --smoke 走 CLI 参数（graspo launch --smoke），不修改用户 config 文件
#   6. 参数全部走 CLI（--gpus/--model-dir），不使用自定义环境变量
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "用法: bash run.sh <config.yaml> [--smoke] [--gpus <ids>] [--model-dir <path>]"
    echo "      --gpus 4,5           指定 GPU（默认自动选空闲卡）"
    echo "      --model-dir <path>   模型目录挂载源（默认 \$ROOT_DIR/models）"
}

# ── 参数解析 ────────────────────────────────────────────────────────────────
CONFIG=""
SMOKE=0
GPU_IDS=""
MODEL_DIR="$ROOT_DIR/models"
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)
            SMOKE=1
            shift
            ;;
        --gpus)
            [ $# -ge 2 ] || { echo "ERROR: --gpus 需要参数"; usage; exit 1; }
            GPU_IDS="$2"
            shift 2
            ;;
        --model-dir)
            [ $# -ge 2 ] || { echo "ERROR: --model-dir 需要参数"; usage; exit 1; }
            MODEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$CONFIG" ]; then
                CONFIG="$1"
            else
                echo "ERROR: 未知参数 $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    usage
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: 配置文件不存在: $CONFIG"
    exit 1
fi
CONFIG_ABS="$(realpath "$CONFIG")"

# ── 防呆 1: 自动选择空闲 GPU（--gpus 未指定时）──────────────────────────────
if [ -z "$GPU_IDS" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi 不可用，请确认 GPU 驱动已安装"
        exit 1
    fi
    GPU_IDS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F', ' '$2==0 {printf "%s%s", sep, $1; sep=","}')"
    if [ -z "$GPU_IDS" ]; then
        echo "ERROR: 没有空闲 GPU（所有卡都被占用）"
        echo "  可用: bash run.sh $CONFIG --gpus 4,5 指定要用的卡"
        exit 1
    fi
    echo "自动选择 GPU: $GPU_IDS"
else
    echo "使用指定 GPU: $GPU_IDS"
fi

# ── 防呆 4: 镜像 tag 自动取 git 版本 ───────────────────────────────────────
VERSION="$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")"
IMAGE="graspo:${VERSION}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "镜像 $IMAGE 不存在，先构建（bash docker/build.sh）"
    exit 1
fi

# ── 防呆 5: 对齐宿主时区（动态获取，换机器自动适配）──────────────
# 容器默认 Etc/UTC，不设 TZ 则日志/date 显示与宿主差时区。镜像内有
# tzdata 即可运行时注入，无需重建镜像。优先级：/etc/timezone →
# /etc/localtime 符号链接 → UTC 兜底（覆盖 Debian/macOS/Alpine）。
if [ -f /etc/timezone ]; then
    TZ_VALUE="$(cat /etc/timezone)"
elif [ -L /etc/localtime ]; then
    TZ_VALUE="$(readlink /etc/localtime | sed 's|^.*zoneinfo/||')"
else
    TZ_VALUE="UTC"
fi
echo "容器时区: $TZ_VALUE"

# ── 启动容器 ────────────────────────────────────────────────────────────────
CONTAINER_NAME="graspo-$(basename "$CONFIG" .yaml)"
CONFIG_DIR="$(dirname "$CONFIG_ABS")"
EXTRA_ARGS=()
if [ "$SMOKE" = 1 ]; then
    EXTRA_ARGS+=(--smoke)
    echo "冒烟模式：跑 1 步验证环境"
fi

echo "启动容器: $CONTAINER_NAME"
echo "  镜像: $IMAGE"
echo "  配置: $CONFIG_ABS"
echo "  GPU:  $GPU_IDS"
echo "  模型: $MODEL_DIR → /workspace/graspo/models"

docker run -d --name "$CONTAINER_NAME" \
    --gpus "\"device=$GPU_IDS\"" \
    --ipc=host --shm-size=16g \
    -e "TZ=${TZ_VALUE}" \
    -v "$CONFIG_DIR:/data/configs" \
    -v "$MODEL_DIR:/workspace/graspo/models" \
    "$IMAGE" \
    launch --config "/data/configs/$(basename "$CONFIG")" "${EXTRA_ARGS[@]}"

echo "已启动。查看日志: docker logs -f $CONTAINER_NAME"
echo "停止训练:  docker stop $CONTAINER_NAME"
