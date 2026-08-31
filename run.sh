#!/usr/bin/env bash
# GRASPO 训练启动入口（防呆版）
#
# 唯一入口：从 GitHub clone 后，这是启动训练的唯一方式。
# 用法:
#   bash run.sh my_config.yaml                # 训练（自动选空闲 GPU）
#   bash run.sh my_config.yaml --smoke        # 冒烟：跑 1 步验证环境后停止
#   bash run.sh my_config.yaml --gpus 4,5     # 指定 GPU
#   bash run.sh my_config.yaml --image graspo:v0.22.0   # 指定镜像
#
# 防呆设计:
#   1. 自动选择空闲 GPU（nvidia-smi 检测显存占用为 0 的卡），无需手动数卡
#   2. 只传 --gpus device=<ids>，绝不注入 CUDA_VISIBLE_DEVICES——
#      两者混用会导致 NCCL 初始化死锁（实测卡死）
#   3. 固定 --ipc=host --shm-size=16g（NCCL 共享内存必需）
#   4. 镜像 tag 默认取 git describe，可通过 --image 覆盖
#   5. --smoke 走 CLI 参数（graspo launch --smoke），不修改用户 config 文件
#   6. 挂载目录从 YAML config 自动推导（model_path / train_path / output_dir 的父目录
#      + config 文件所属目录），无需手动指定 --model-dir
#   7. 参数全部走 CLI（--gpus / --image），不使用自定义环境变量
#   8. `NCCL_P2P_DISABLE` 按拓扑自动判定：A800 PCIe 拓扑（GPU 以 NVLink pair 成对、跨 pair 走
#      PCIe bridge）下，NCCL 的 P2P/CUMEM 路径会 hang，需禁用 P2P 走中间内存拷贝；全 NVLink
#      mesh（全 NVLink 互联）无需禁用，禁用反而引入非对称显存/效率损耗。详见 README FAQ。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "用法: bash run.sh <config.yaml> [--smoke] [--gpus <ids>] [--image <name>]"
    echo "      --gpus 4,5           指定 GPU（默认自动选空闲卡）"
    echo "      --image <name>       指定镜像（默认从 git describe 自动推导）"
}

# ── 参数解析 ────────────────────────────────────────────────────────────────
CONFIG=""
SMOKE=0
GPU_IDS=""
IMAGE=""
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
        --image)
            [ $# -ge 2 ] || { echo "ERROR: --image 需要参数"; usage; exit 1; }
            IMAGE="$2"
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

# ── 防呆 2: 从 YAML 配置自动推导挂载目录 ────────────────────────────────────
# 读取 model.model_path / data.train_path / training.output_dir 的父目录，
# 加上 config 文件所属目录，去重后全部挂载。
# 用户只需写 YAML，不需要关心容器挂载。

_yaml_value() {
    # 从 YAML 中提取 key: value 行的值（简单解析，适合顶层和第二层字段）
    # TODO(P0): 当前为 grep 启发式，不支持嵌套映射/引号/多行值等边界情况。
    # 未来可改用 python3 -c "import yaml; ..."（需确保宿主有 pyyaml）或引入 yq。
    grep -E "^\s*${1}:" "$CONFIG_ABS" 2>/dev/null | head -1 | sed 's/^[^:]*:\s*//' | xargs
}

# 收集需要挂载的目录（去重）
_mount_dirs=()
_add_mount_dir() {
    local d="$1"
    if [ -z "$d" ] || [ ! -d "$d" ]; then
        return 0
    fi
    for existing in "${_mount_dirs[@]}"; do
        if [ "$existing" = "$d" ]; then
            return 0
        fi
    done
    _mount_dirs+=("$d")
}

_add_mount_dir "$(dirname "$CONFIG_ABS")"

model_path="$(_yaml_value "model_path")"
if [ -n "$model_path" ] && [ -d "$(dirname "$model_path")" ]; then
    _add_mount_dir "$(realpath "$(dirname "$model_path")")"
fi

train_path="$(_yaml_value "train_path")"
if [ -n "$train_path" ] && [ -d "$(dirname "$train_path")" ]; then
    _add_mount_dir "$(realpath "$(dirname "$train_path")")"
    # 数据集根（祖父目录）——仅在图像引用需要时挂载（挂载面积最小化）：
    # data/ 与 images/ 平级的数据集，图像相对引用 ../images/x.jpg 需要
    # 数据集根可达（v0.21 smoke 实测发现）。读取训练数据首行 image 字段，
    # 以 ../ 开头才挂祖父目录；否则父目录已覆盖（单层结构）。
    _image_ref="$(
        head -c 8192 "$train_path" 2>/dev/null \
        | grep -o '"image"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 || true \
        | sed 's/.*:[[:space:]]*"//; s/"$//'
    )"
    if [ -n "$_image_ref" ] && [ "${_image_ref#../}" != "$_image_ref" ]; then
        _add_mount_dir "$(realpath "$(dirname "$(dirname "$train_path")")")"
    fi
fi

output_dir="$(_yaml_value "output_dir")"
if [ -n "$output_dir" ] && [ -d "$(dirname "$output_dir")" ]; then
    _add_mount_dir "$(realpath "$(dirname "$output_dir")")"
fi

if [ ${#_mount_dirs[@]} -eq 0 ]; then
    echo "ERROR: 无法从配置中推导任何挂载目录"
    exit 1
fi

# 构建 -v 参数
_mount_args=()
for d in "${_mount_dirs[@]}"; do
    _mount_args+=(-v "$d:$d")
done

echo "挂载目录: ${_mount_dirs[*]}"

# ── 防呆 3: 镜像 tag 默认取 git 版本，可通过 --image 覆盖 ─────────────────
if [ -z "$IMAGE" ]; then
    IMAGE="graspo:$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")"
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: 镜像 $IMAGE 不存在"
    echo "  构建: bash docker/build.sh"
    echo "  或指定: bash run.sh $CONFIG --image graspo:v0.22.0"
    exit 1
fi

# ── 防呆 4: 对齐宿主时区（动态获取，换机器自动适配）────────────────────────
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

# ── 防呆 8: NCCL_P2P_DISABLE 按拓扑条件化 ──────────────────────────────────
# 在 A800 PCIe 拓扑（GPU 以 NVLink pair 成对、跨 pair 走 PCIe bridge）下，
# NCCL 的 P2P/CUMEM 路径在小张量 all-reduce 或跨 pair 的 P2P send/recv 会 hang，
# 需禁用 P2P 走中间内存拷贝。但全 NVLink mesh（所有 GPU 通过 NVLink 全互联）无需禁用，
# 禁用反而引入非对称显存/效率损耗（默认组缓冲集中到 device0、GPU0 先 OOM）。
# 按选中 GPU 的拓扑自动判定：选中 GPU 间存在 PXB/PHB/SYS（跨 PCIe bridge / 跨
# NUMA）路径 → 需禁用 P2P；全部为 NVLink（NV#）→ 保留 P2P（更快、显存对称）。
_p2p_disable_required() {
    local gpus="$1"
    local topo row link i j gi gj
    local -a garr
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[run.sh] WARNING: nvidia-smi 不可用，回退为禁用 P2P（安全优先，避免 hang）"
        return 0
    fi
    topo="$(nvidia-smi topo -m 2>/dev/null || true)"
    if [ -z "$topo" ]; then
        echo "[run.sh] WARNING: 无法读取 GPU 拓扑，回退为禁用 P2P（安全优先，避免 hang）"
        return 0
    fi
    IFS=',' read -ra garr <<< "$gpus"
    local n=${#garr[@]}
    for ((i=0;i<n;i++)); do
        gi="${garr[i]}"
        for ((j=i+1;j<n;j++)); do
            gj="${garr[j]}"
            row="$(echo "$topo" | awk -v idx="GPU${gi}" '$1==idx {print; exit}')"
            if [ -z "$row" ]; then
                echo "[run.sh] WARNING: 找不到 GPU${gi} 拓扑行，回退为禁用 P2P"
                return 0
            fi
            link="$(echo "$row" | awk -v col=$((gj+2)) '{print $col}')"
            case "$link" in
                *PXB*|*PHB*|*SYS*) return 0 ;;
            esac
        done
    done
    return 1
}

DOCKER_ENV_ARGS=(-e "TZ=${TZ_VALUE}" -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
if _p2p_disable_required "$GPU_IDS"; then
    DOCKER_ENV_ARGS+=(-e "NCCL_P2P_DISABLE=1")
    echo "GPU 拓扑含跨 PCIe bridge/NUMA 路径，禁用 NCCL P2P（NCCL_P2P_DISABLE=1）"
else
    echo "GPU 拓扑全 NVLink，保留 NCCL P2P（不设 NCCL_P2P_DISABLE）"
fi

# ── 启动容器 ────────────────────────────────────────────────────────────────
CONTAINER_NAME="graspo-$(basename "$CONFIG" .yaml)"
# 防呆（§2.4 操作防呆 / §2.5 运维边界）：同名容器若已存在（上次训练未清理），旧
# worker 可能仍占用 GPU，重启会叠出"每 GPU 多进程 + 显存不均"。这里不自动删除，
# 而是列出目标并明确提示，由用户确认后手动清理，避免误删。
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "错误：容器 $CONTAINER_NAME 已存在（可能是上次训练未清理）。"
    echo "  旧训练进程可能仍占用 GPU，将导致多进程叠加与显存不均。"
    echo "  请先确认该训练已停止，然后删除旧容器再重试："
    echo "      docker rm -f $CONTAINER_NAME"
    exit 1
fi
EXTRA_ARGS=()
if [ "$SMOKE" = 1 ]; then
    EXTRA_ARGS+=(--smoke)
    echo "冒烟模式：跑 1 步验证环境"
fi

echo "启动容器: $CONTAINER_NAME"
echo "  镜像: $IMAGE"
echo "  配置: $CONFIG_ABS"
echo "  GPU:  $GPU_IDS"

docker run -d --name "$CONTAINER_NAME" \
    --gpus '"device='"$GPU_IDS"'"' \
    --ipc=host --shm-size=16g \
    "${DOCKER_ENV_ARGS[@]}" \
    "${_mount_args[@]}" \
    "$IMAGE" \
    launch --config "$CONFIG_ABS" "${EXTRA_ARGS[@]}"

echo "已启动。查看日志: docker logs -f $CONTAINER_NAME"
echo "停止训练:  docker stop $CONTAINER_NAME"