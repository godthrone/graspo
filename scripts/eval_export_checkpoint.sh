#!/usr/bin/env bash
# 评测链路：checkpoint → merged-hf 导出（显式锁卡 + fail-closed）
#
# 职责：把训练 checkpoint 转成可被 vLLM 服务的 HF 目录。
#   - PEFT adapter（ms-swift 产物）→ `graspo eval export`（合并进 base）
#   - native GRASPO checkpoint      → 既有 `graspo export --config`（格式自校验）
#   - 本身就是 base 模型            → 无需动作
# 不负责：起服务（eval_serve_vllm.sh）、跑评测（graspo eval run）、清理。
#
# 用法:
#   EVAL_CONFIG=/abs/path/eval.yaml \\
#   EVAL_GPUS=0 \\
#   EVAL_IMAGE=graspo:v0.24.0 \\        # 可选；默认从 git tag 推导
#     bash scripts/eval_export_checkpoint.sh
#
# 防呆设计（宪法 §2.3 / §2.4）:
#   1. EVAL_GPUS 未显式给出 → 退出，无默认值（同 eval_serve_vllm.sh）。
#   2. 卡取值 ⊆ {0,1,2,3,4,5}，卡数 ≤ 4，含 6/7 一律拒绝。
#      （历史教训：ELAM v3 脚本硬编码 EXPORT_GPU=6——严禁照抄。）
#   3. 输出目录若已有完整 merged-hf 产物 → **直接复用**，不重复合并
#      （省时间且避免覆盖既有产物）。需要强制重做时用 --allow-overwrite。
#   4. 显存检查只查选中卡（nvidia-smi --id）。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_GPUS="${EVAL_GPUS:-}"
EVAL_IMAGE="${EVAL_IMAGE:-}"
ALLOW_OVERWRITE_FLAG=""

ALLOWED_GPUS="0 1 2 3 4 5"
MAX_GPU_COUNT=4

die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --allow-overwrite) ALLOW_OVERWRITE_FLAG="--allow-overwrite"; shift ;;
        -h|--help) echo "usage: EVAL_CONFIG=<yaml> EVAL_GPUS=0 bash scripts/eval_export_checkpoint.sh [--allow-overwrite]"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

# 顺序有意如此：**绑卡检查先于配置文件检查**。安全边界必须在最前面 ——
# 否则配置文件路径写错会把"你正在碰生产卡"这条更严重的错误盖掉。
[ -n "$EVAL_GPUS" ] || die "EVAL_GPUS is required (e.g. EVAL_GPUS=0). No default: GPU 6/7 are occupied by the production vLLM on the target GPU host."

# ── 卡集合校验 ────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARR <<< "$EVAL_GPUS"
GPU_COUNT=0
NORMALIZED_GPUS=()
for gpu in "${GPU_ARR[@]}"; do
    gpu="$(echo "$gpu" | xargs)"
    [ -n "$gpu" ] || continue
    case "$gpu" in
        ''|*[!0-9]*) die "invalid GPU index '$gpu' in EVAL_GPUS='$EVAL_GPUS'" ;;
    esac
    echo " $ALLOWED_GPUS " | grep -q " $gpu " \
        || die "GPU $gpu is outside the permitted set {$ALLOWED_GPUS}; GPU 6/7 belong to the production vLLM"
    dup=0
    for seen in ${NORMALIZED_GPUS[@]+"${NORMALIZED_GPUS[@]}"}; do
        [ "$seen" = "$gpu" ] && dup=1
    done
    [ "$dup" = 1 ] || NORMALIZED_GPUS+=("$gpu")
done
GPU_COUNT=${#NORMALIZED_GPUS[@]}
[ "$GPU_COUNT" -gt 0 ] || die "EVAL_GPUS='$EVAL_GPUS' parsed to zero devices"
[ "$GPU_COUNT" -le "$MAX_GPU_COUNT" ] || die "requested $GPU_COUNT GPUs but at most $MAX_GPU_COUNT are allowed"

IFS=',' GPU_CSV="${NORMALIZED_GPUS[*]}"
echo "export will use GPU(s): $GPU_CSV (verified ⊆ {$ALLOWED_GPUS}, count $GPU_COUNT ≤ $MAX_GPU_COUNT)"

[ -n "$EVAL_CONFIG" ] || die "EVAL_CONFIG is required (path to the eval config YAML)"
[ -f "$EVAL_CONFIG" ] || die "EVAL_CONFIG does not exist: $EVAL_CONFIG"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not available; run on the GPU host"
echo "--- memory on the selected card(s) only (nvidia-smi --id=$GPU_CSV) ---"
nvidia-smi --id="$GPU_CSV" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    || die "nvidia-smi --id=$GPU_CSV failed"

# ── 只读预检：先问清楚要做什么（复用 Python 侧的识别逻辑，保持单一真相源）──
echo "--- graspo eval prepare (read-only) ---"
(cd "$ROOT_DIR" && uv run graspo eval prepare --eval-config "$EVAL_CONFIG") || die "prepare failed"

CONFIG_ABS="$(realpath "$EVAL_CONFIG")"
if [ -z "$EVAL_IMAGE" ]; then
    EVAL_IMAGE="graspo:$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")"
fi
docker image inspect "$EVAL_IMAGE" >/dev/null 2>&1 || die "image $EVAL_IMAGE not found"

# 复用 Python 侧的 mount 推导：把 config 里出现的路径都挂进容器。
_mount_args=(-v "$(dirname "$CONFIG_ABS")":"$(dirname "$CONFIG_ABS")")
for key in dataset_path train_dataset_path base_model_path checkpoint_path merged_output_dir output_dir; do
    value="$(grep -E "^[[:space:]]*${key}:" "$CONFIG_ABS" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | xargs || true)"
    [ -n "$value" ] || continue
    [ -d "$value" ] || continue
    directory="$(realpath "$value")"
    if [ -d "$directory" ]; then
        _mount_args+=(-v "$directory":"$directory")
    elif [ -d "$(dirname "$directory")" ]; then
        parent="$(realpath "$(dirname "$directory")")"
        _mount_args+=(-v "$parent":"$parent")
    fi
done
# 去重（简单方式：交给 docker，重复 -v 同一路径是合法的）
echo "mounts: ${_mount_args[*]}"

echo "--- merging (PEFT path) / delegating (native path) ---"
docker run --rm \
    --gpus '"device='"$GPU_CSV"'"' \
    --ipc=host \
    --ulimit memlock=-1 \
    "${_mount_args[@]}" \
    -v "$ROOT_DIR":/workspace:ro \
    "$EVAL_IMAGE" \
    eval export --eval-config "$CONFIG_ABS" $ALLOW_OVERWRITE_FLAG

echo "export step finished. verify the merged directory, then start the server:"
echo "    EVAL_MODEL_DIR=<merged_output_dir> EVAL_GPUS=$GPU_CSV bash scripts/eval_serve_vllm.sh"
