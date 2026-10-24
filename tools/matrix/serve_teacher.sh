#!/usr/bin/env bash
# serve_teacher.sh — OPD 档「外部教师服务」起服脚本（vLLM / OpenAI 兼容）
#
# 职责：把 OPD 档的**教师**（27B 的 SFT/LoRA/dp1 产物）以 vLLM 起成一个外部服务，
#       供训练侧 `distill.teacher_model_server` 指向它（ms-swift `--teacher_model_server`）。
# 不负责：训练、评测、择卡决策、容器清理（见 docs/run-discipline.md §21）。
#
# ★ 卡数口径（主席 2026-10-24 原话）：
#   「OPD×27B 三档（M066–M068），教师用 vLLM 部署，单独弄别的卡，不算在测试表的卡数里面，
#     测试表那个卡数是针对被训练的模型的。」
#   ⇒ 教师占的卡 **不计入**矩阵该档 `cards`；本脚本的 `--gpus` 与被训练档的卡集是两码事。
#
# ⛔ 默认 dry-run：只打印将要执行的 docker 命令，**不做任何外部调用**（不起容器、不探网、不查卡）。
#    真正起服必须显式 `--apply`。
#
# 用法:
#   bash tools/matrix/serve_teacher.sh --gpus 3 --model /models/teacher/merged-hf --port 18901
#       ⇒ 只打印命令（dry-run）
#   bash tools/matrix/serve_teacher.sh --gpus 3 --model /models/teacher/merged-hf --port 18901 --apply
#       ⇒ 真起容器
#
# 参数:
#   --gpus <n[,m...]>          必需。教师独占的卡号（⊆ 本机 0–7；逗号分隔可多张）。
#   --model <宿主机目录>        必需。教师产物目录（HF 形态；LoRA 需先 merge）。
#   --port <宿主端口>           必需。宿主侧端口（容器内固定 8000）。
#   --served-model-name <名>    可选。默认 = 模型目录名。
#   --image <镜像>              可选。默认 vllm-openai:v0.22.0。
#   --gpu-memory-utilization <f> 可选。默认 0.85。
#   --max-logprobs <n>          可选。默认 65 = `gkd_logits_topk`(64) + 1 —— ★ 外部教师 API 只回
#                               top-k logprobs，服务侧 `--max-logprobs` 必须 >= topk+1，否则
#                               训练侧请求会被拒（见 core/schema.py 的同名校验与 message）。
#   --dry-run | --apply         二者互斥；缺省 = dry-run。
#   -h | --help
#
# 起服前必须先取卡（见 docs/run-discipline.md §17.2 / §21）:
#   python3 scripts/run_guards.py cards-busy --ps-file /tmp/ps.txt --inspect-file /tmp/ins.txt
#   python3 scripts/gpu_idle_assert.py --visible <卡列表> --position-mib <头寸>
#   ⛔ 本脚本**不**自动选卡、不判卡是否空闲（那是设施层的判据，单一真相源）。
#
# 退出码: 0 = 打印成功 / 起服命令返回 0；1 = 参数或前置校验失败；docker 侧失败按其 rc。
set -uo pipefail

ALLOWED_GPUS="0 1 2 3 4 5 6 7"

GPU_SPEC=""
MODEL_DIR=""
PORT=""
SERVED_NAME=""
IMAGE="vllm-openai:v0.22.0"
GPU_MEM_UTIL="0.85"
MAX_LOGPROBS="65"
APPLY=0

die() { printf '⛔ %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus)   [[ $# -ge 2 ]] || die "--gpus 缺参数值"; GPU_SPEC="$2"; shift 2 ;;
    --model)  [[ $# -ge 2 ]] || die "--model 缺参数值"; MODEL_DIR="$2"; shift 2 ;;
    --port)   [[ $# -ge 2 ]] || die "--port 缺参数值"; PORT="$2"; shift 2 ;;
    --served-model-name) [[ $# -ge 2 ]] || die "--served-model-name 缺参数值"; SERVED_NAME="$2"; shift 2 ;;
    --image)  [[ $# -ge 2 ]] || die "--image 缺参数值"; IMAGE="$2"; shift 2 ;;
    --gpu-memory-utilization) [[ $# -ge 2 ]] || die "--gpu-memory-utilization 缺参数值"; GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-logprobs) [[ $# -ge 2 ]] || die "--max-logprobs 缺参数值"; MAX_LOGPROBS="$2"; shift 2 ;;
    --dry-run) APPLY=0; shift ;;
    --apply)   APPLY=1; shift ;;
    *) die "未识别参数：$1（⛔ 不猜；见 --help）" ;;
  esac
done

[[ -n "$GPU_SPEC" ]] || die "--gpus 必需（教师独占的卡号，如 --gpus 3）。⛔ 本脚本不自动选卡。"
[[ -n "$MODEL_DIR" ]] || die "--model 必需（教师产物目录，如 --model /models/Qwen3.8-27B-teacher-merged）。"
[[ -n "$PORT" ]] || die "--port 必需（宿主端口，如 --port 18901）。"

# ── 卡号校验（取值域 = 本机全卡 0–7；⛔ 不判卡数、⛔ 不查占用）────────────────
GPU_NORMALIZED=""
for gpu in ${GPU_SPEC//,/ }; do
  case "$gpu" in
    ''|*[!0-9]*) die "卡号非法：'$gpu'（--gpus='$GPU_SPEC'；期望逗号分隔的整数）" ;;
  esac
  echo " $ALLOWED_GPUS " | grep -q " $gpu " \
    || die "卡号 $gpu 超出允许集 {$ALLOWED_GPUS}（本机只有 0–7）"
  case ",$GPU_NORMALIZED," in
    *",$gpu,"*) die "卡号重复：$gpu（--gpus='$GPU_SPEC'）" ;;
  esac
  GPU_NORMALIZED="${GPU_NORMALIZED:+$GPU_NORMALIZED,}$gpu"
done
[[ -n "$GPU_NORMALIZED" ]] || die "--gpus='$GPU_SPEC' 解析出 0 张卡"

# ── 端口校验（纯数字、1–65535）────────────────────────────────────────────
case "$PORT" in
  ''|*[!0-9]*) die "端口非法：'$PORT'（期望 1–65535 的整数）" ;;
esac
((PORT >= 1 && PORT <= 65535)) || die "端口越界：$PORT（期望 1–65535）"

# ── 显存利用率校验（0.0 < f ≤ 1.0）────────────────────────────────────────
case "$GPU_MEM_UTIL" in
  ''|*[!0-9.]*) die "--gpu-memory-utilization 非法：'$GPU_MEM_UTIL'（期望小数，如 0.85）" ;;
esac
case "$MAX_LOGPROBS" in
  ''|*[!0-9]*) die "--max-logprobs 非法：'$MAX_LOGPROBS'（期望正整数，且 >= gkd_logits_topk + 1）" ;;
esac
((MAX_LOGPROBS >= 1)) || die "--max-logprobs 必须 >= 1"  

MODEL_DIR="${MODEL_DIR%/}"
[[ -n "$MODEL_DIR" ]] || die "--model 不能是根目录 '/'"
SERVED_NAME="${SERVED_NAME:-$(basename -- "$MODEL_DIR")}"
[[ -n "$SERVED_NAME" ]] || die "无法从 --model 推导 --served-model-name，请显式给出"

# ── 命令拼装（唯一真相源：本数组；dry-run 打印的就是它）────────────────────
GPUS_FLAG="\"device=$GPU_NORMALIZED\""
CMD=(docker run -d
     --gpus "$GPUS_FLAG"
     -v "$MODEL_DIR":/model:ro
     -p "$PORT":8000
     "$IMAGE"
     --model /model
     --served-model-name "$SERVED_NAME"
     --gpu-memory-utilization "$GPU_MEM_UTIL"
     --max-logprobs "$MAX_LOGPROBS")

# 可复制粘贴的一行（dry-run 的主产物）
printf -v CMD_LINE '%q ' "${CMD[@]}"
CMD_LINE="${CMD_LINE% }"

cat <<EOF
# ── OPD 外部教师服务（vLLM）─────────────────────────────────────────────
# 教师身份 : $(basename -- "$MODEL_DIR")
# 教师独占卡: $GPU_NORMALIZED   （★ 不计入矩阵该档 cards —— 矩阵卡数只算被训练的模型）
# 宿主端口  : $PORT  →  容器内 8000
# 容器内模型: /model（宿主 $MODEL_DIR，只读挂载）
# 服务端点  : http://127.0.0.1:$PORT/v1
# 训练侧接线: distill.teacher_model_server = http://127.0.0.1:$PORT
#             （数据件该档行 => tools/matrix/gen_configs.py 重生成 yaml；⛔ 不手改 yaml）
# 镜像      : $IMAGE
# max-logprobs: $MAX_LOGPROBS   （★ 必须 >= 训练侧 distill.gkd_logits_topk + 1，否则请求被拒）
EOF

say ""
say "# 将要执行的命令（唯一产物；--apply 时原样执行）："
say "$CMD_LINE"
say ""

if ((APPLY == 0)); then
  say "# ★ dry-run：未执行任何外部调用。真要起服请追加 --apply。"
  # 仅本机文件系统检查（不是外部调用）；缺目录只提示，不影响 dry-run 的 rc
  if [[ -d "$MODEL_DIR" ]]; then
    if [[ -f "$MODEL_DIR/config.json" ]]; then say "# 预检：$MODEL_DIR/config.json 存在（可服务的 HF 目录形态）"
    else say "# ⚠ 预检：$MODEL_DIR 存在但无 config.json（LoRA 未 merge？vLLM 起不来）"; fi
  else
    say "# ⚠ 预检：$MODEL_DIR 在宿主上不存在（容器内 /model 会挂空 ⇒ 起服必失败）"
  fi
  exit 0
fi

# ── --apply：真起服 ───────────────────────────────────────────────────────
[[ -d "$MODEL_DIR" ]] || die "--apply 前置校验失败：模型目录不存在：$MODEL_DIR"
[[ -f "$MODEL_DIR/config.json" ]] || die "--apply 前置校验失败：$MODEL_DIR/config.json 不存在（LoRA 需先 merge 成 merged-hf）"
command -v docker >/dev/null 2>&1 || die "--apply 需要 docker，但当前环境找不到 docker 命令"

say "# ★ --apply：执行上面的命令（下方即为 docker 的真实输出）"
CID="$("${CMD[@]}")" || die "docker run 失败（rc=$?）；排查：docker images | grep $IMAGE / docker logs <容器>"
say "$CID"
cat <<EOF

# 已起容器：$CID
# 就绪探针（⛔ 不要凭"容器起来了"判就绪）:
#     docker ps --filter publish=$PORT
#     curl -s http://127.0.0.1:$PORT/v1/models
#     docker logs -f $CID
# 停机（本脚本不自动清理；⛔ 别 pkill）:
#     docker stop $CID
EOF
