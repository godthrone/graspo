#!/usr/bin/env bash
# 评测链路：启动 vLLM 服务（评测专用，显式锁卡 + fail-closed）
#
# 职责：为 `graspo eval run` 起一个 vLLM OpenAI 兼容服务，只服务一个模型目录。
# 不负责：合并权重、跑评测、算 Δ、清理容器（见同目录其它脚本）。
#
# 用法:
#   EVAL_MODEL_DIR=/abs/path/to/merged \\   # 必需：要被服务的模型目录
#   EVAL_GPUS=4,5 \\                        # 必需：显式卡列表（允许集 4–7）
#   EVAL_PORT=18889 \\                      # 可选，默认 18889
#   EVAL_SERVED_NAME=graspo-eval \\         # 可选，须与 eval 配置的 served_model_name 一致
#     bash scripts/eval_serve_vllm.sh
#
# 防呆设计（宪法 §2.3 边界校验、§2.4 操作防呆）:
#   1. EVAL_GPUS 未显式给出 → 直接退出（**没有自动选卡**）。
#      任何"默认用个空闲卡"的启发式都可能踩上别人的卡。
#      宁可让人显式写，也不给默认值。
#   2. 卡取值必须 ⊆ {4,5,6,7}，且卡数 ≤ 4。
#      （历史教训：ELAM v3 的 v3_eval_pipeline_v2.sh 硬编码 EXPORT_GPU=6 /
#        VLLM_GPU=7——"硬编码卡号"仍严禁照抄；但 6/7 不再是禁地，见下。）
#      ★ 口径变更 2026-09-30：旧规则"含 6/7 一律拒绝"的理由是"GPU 6/7 被常驻
#        生产 vLLM 占死（各 ~74.7 GiB）"——**该容器已于 2026-09-23 停止，理由
#        失效**；GPU 4–7 现为 graspo 专用卡池 ⇒ 允许集改为 {4,5,6,7}。
#   3. 显存检查**只查选中的卡**（nvidia-smi --id=<ids>），不带 -i 的全量采样会把
#      生产卡读数混进来，是已确证的教训。
#   4. 只传 --gpus '"device=...",' 形式，**不注入 CUDA_VISIBLE_DEVICES**——
#      与训练入口 run.sh 同一约定，避免 NCCL 初始化路径分叉。
#   5. 容器名冲突时不自动删除，列出目标并交人工确认（§2.4）。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EVAL_MODEL_DIR="${EVAL_MODEL_DIR:-}"
EVAL_GPUS="${EVAL_GPUS:-}"
EVAL_PORT="${EVAL_PORT:-18889}"
EVAL_SERVED_NAME="${EVAL_SERVED_NAME:-graspo-eval}"
EVAL_IMAGE="${EVAL_IMAGE:-vllm-openai:v0.22.0}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-1}"

# 口径变更 2026-09-30：原值 "0 1 2 3 4 5"（且遇 6/7 一律拒绝）。其理由——"GPU 6/7 被
# 常驻生产 vLLM 占死（各 ~74.7 GiB）"——**已失效**：该容器 2026-09-23 已停、卡已空；
# GPU 4–7 现为 graspo 专用卡池 ⇒ 允许集改为 4–7。
ALLOWED_GPUS="4 5 6 7"
MAX_GPU_COUNT=4

die() { echo "ERROR: $*" >&2; exit 1; }

# ── 防呆 1: 必需的显式输入 ────────────────────────────────────────────────
# 顺序有意如此：**绑卡检查先于模型目录内容检查**。安全边界必须在最前面 ——
# 否则模型路径写错会把"你正在碰生产卡"这条更严重的错误盖掉。
[ -n "$EVAL_MODEL_DIR" ] || die "EVAL_MODEL_DIR is required (the model directory to serve). No default."
[ -n "$EVAL_GPUS" ] || die "EVAL_GPUS is required (e.g. EVAL_GPUS=4,5). There is no auto-selection: name the cards explicitly."
[ -d "$EVAL_MODEL_DIR" ] || die "EVAL_MODEL_DIR does not exist: $EVAL_MODEL_DIR"

# ── 防呆 2: 卡集合校验（取值 + 数量）──────────────────────────────────────
IFS=',' read -ra GPU_ARR <<< "$EVAL_GPUS"
GPU_COUNT=0
NORMALIZED_GPUS=()
for gpu in "${GPU_ARR[@]}"; do
    gpu="$(echo "$gpu" | xargs)"
    [ -n "$gpu" ] || continue
    case "$gpu" in
        ''|*[!0-9]*) die "invalid GPU index '$gpu' in EVAL_GPUS='$EVAL_GPUS'; expected comma-separated integers" ;;
    esac
    if ! echo " $ALLOWED_GPUS " | grep -q " $gpu "; then
        die "GPU $gpu is outside the permitted set {$ALLOWED_GPUS}. The eval card pool is 4–7 (graspo-dedicated)."
    fi
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
GPU_FLAG='"device='"$GPU_CSV"'"'
echo "eval vLLM will use GPU(s): $GPU_CSV (explicit, verified ⊆ {$ALLOWED_GPUS}, count $GPU_COUNT ≤ $MAX_GPU_COUNT)"

# ── 防呆 3: 模型目录必须真能服务 ──────────────────────────────────────────
[ -f "$EVAL_MODEL_DIR/config.json" ] || die "EVAL_MODEL_DIR has no config.json — not a servable HF model dir: $EVAL_MODEL_DIR"

# ── 防呆 3: 只查选中的卡（必须带 --id）─────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not available; run this script on the GPU host"
echo "--- memory on the selected card(s) only (nvidia-smi --id=$GPU_CSV) ---"
nvidia-smi --id="$GPU_CSV" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || die "nvidia-smi --id=$GPU_CSV failed"

busy="$(nvidia-smi --id="$GPU_CSV" --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 > 20000 {printf "%s ", $1}')"
if [ -n "$busy" ]; then
    die "selected GPU(s) look occupied (>20000 MiB): $busy — pick different cards; do not free a card another user is serving on"
fi

# ── 防呆 4: 容器名冲突 → 列出并交人工，不自动删 ───────────────────────────
CONTAINER_NAME="vllm-eval-${EVAL_SERVED_NAME}"
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "ERROR: container $CONTAINER_NAME already exists." >&2
    echo "  A previous eval server may still be running on these GPUs." >&2
    echo "  Review and remove it yourself (no auto-delete):" >&2
    echo "      docker ps -a --filter name=^${CONTAINER_NAME}$" >&2
    echo "      docker rm -f ${CONTAINER_NAME}" >&2
    exit 1
fi

docker image inspect "$EVAL_IMAGE" >/dev/null 2>&1 \
    || die "image $EVAL_IMAGE not found; build or load it first"

# ── 启动 ──────────────────────────────────────────────────────────────────
# --reasoning-parser / --tool-call-parser 与 v3 一致：评测取的是 tool_calls，
# 解析器不对齐会让 tool_calls 变成空列表，准确率直接归零（静默踩坑）。
echo "starting container $CONTAINER_NAME (port $EVAL_PORT → 8000, image $EVAL_IMAGE)"
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus "$GPU_FLAG" \
    -p "${EVAL_PORT}:8000" \
    --ipc=host \
    --ulimit memlock=-1 \
    -v "$EVAL_MODEL_DIR":/model:ro \
    "$EVAL_IMAGE" \
    /model \
    --tensor-parallel-size "$EVAL_TP_SIZE" \
    --max-model-len "$EVAL_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --served-model-name "$EVAL_SERVED_NAME"

cat <<EOF

started. wait for readiness, then run:
    docker logs -f $CONTAINER_NAME
    curl -s http://127.0.0.1:${EVAL_PORT}/v1/models
    cd $ROOT_DIR && uv run graspo eval run --eval-config <your eval config>.yaml

NOTE: temperature is locked to 0 inside graspo.eval — do not pass any override.
EOF
