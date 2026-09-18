#!/usr/bin/env bash
# 评测链路：安全清理入口（只列清单 + 打印命令，**绝不自动删除**）
#
# 职责：把本次评测产生的大体积中间物逐项列出来，并打印**可逐项核对**的删除命令
#       交给人工执行。
# 不负责：起服务、跑评测、导出。它是评测结束后的收尾辅助。
#
# 为什么默认不自动删（宪法 §2.4 操作防呆）:
#   破坏性操作必须在执行前让操作者看见"即将发生什么"。通配符展开的结果在执行前
#   不可见，是误删事故的根源。因此本脚本
#     1. 只用 `ls` / `du` 逐项列出目标；
#     2. **不使用任何通配符**（`*`、`?`、`[..]`）构造删除命令；
#     3. 默认只打印，不执行。确认无误后由人把命令复制出去执行。
#
# 用户已定：「跑完了测好了就删」——但**留可追溯记录**：评测产物
# （eval_report.json / environment.json）与 Δ 结论要保留，只有大体积中间物
# （merged 权重）测完即删。
#
# 用法:
#   EVAL_CONFIG=/abs/path/eval.yaml bash scripts/eval_cleanup.sh
#   EVAL_CONFIG=/abs/path/eval.yaml bash scripts/eval_cleanup.sh --include-reports
#       （--include-reports 会把产物目录也列入待删清单——只有在你已把结论
#         抄进台账后才用这个开关。）
set -euo pipefail

EVAL_CONFIG="${EVAL_CONFIG:-}"
INCLUDE_REPORTS=0

ALLOWED_GPUS="0 1 2 3 4 5"

die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --include-reports) INCLUDE_REPORTS=1; shift ;;
        -h|--help)
            echo "usage: EVAL_CONFIG=<yaml> bash scripts/eval_cleanup.sh [--include-reports]"
            exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -n "$EVAL_CONFIG" ] || die "EVAL_CONFIG is required (path to the eval config YAML)"
[ -f "$EVAL_CONFIG" ] || die "EVAL_CONFIG does not exist: $EVAL_CONFIG"

value_of() {
    grep -E "^[[:space:]]*$1:" "$EVAL_CONFIG" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | xargs || true
}

MERGED_OUTPUT_DIR="$(value_of merged_output_dir)"
OUTPUT_DIR="$(value_of output_dir)"
SERVED_NAME="$(value_of served_model_name)"
CONTAINER_NAME="vllm-eval-${SERVED_NAME:-graspo-eval}"

echo "=============================================================="
echo " eval cleanup plan (NOT executed — review, then copy & run)"
echo " config: $EVAL_CONFIG"
echo "=============================================================="

echo
echo "--- 1. containers (list first, never auto-remove) ---"
if command -v docker >/dev/null 2>&1; then
    docker ps -a --filter "name=^${CONTAINER_NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
else
    echo "(docker not available here — nothing listed)"
fi

echo
echo "--- 2. merged weight directory (large intermediate; delete after the run) ---"
if [ -n "$MERGED_OUTPUT_DIR" ] && [ -d "$MERGED_OUTPUT_DIR" ]; then
    ls -la "$MERGED_OUTPUT_DIR"
    du -sh "$MERGED_OUTPUT_DIR" 2>/dev/null || true
    echo
    echo "to remove it (explicit path, no wildcard):"
    echo "    rm -rf '$MERGED_OUTPUT_DIR'"
    echo "  (verified: only that directory is targeted; nothing else is touched)"
else
    echo "no merged_output_dir recorded in the config, or it does not exist — nothing to delete"
fi

echo
echo "--- 3. eval artifacts (KEEP by default — they are the traceable record) ---"
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    ls -la "$OUTPUT_DIR"
    du -sh "$OUTPUT_DIR" 2>/dev/null || true
    echo
    echo "  KEEP: eval_report.json / environment.json are the audit trail for the conclusion."
    echo "        Make sure the numbers are written into the ledger before deleting anything."
    if [ "$INCLUDE_REPORTS" = 1 ]; then
        echo "  --include-reports was given, so the deletion command is printed as well:"
        echo "    rm -rf '$OUTPUT_DIR'"
    else
        echo "  (not listed for deletion; re-run with --include-reports if you really want it gone)"
    fi
else
    echo "no output_dir recorded in the config, or it does not exist — nothing listed"
fi

echo
echo "--- 4. container removal command (after you confirm it is stopped) ---"
echo "    docker ps -a --filter name=^${CONTAINER_NAME}\$"
echo "    docker stop ${CONTAINER_NAME}"
echo "    docker rm ${CONTAINER_NAME}"
echo
echo "verify the GPU(s) really went back to idle, listing ONLY the eval cards"
echo "(never sample all cards — production reads get mixed in):"
echo "    nvidia-smi --id=<your eval cards, e.g. 0,1> --query-gpu=index,memory.used --format=csv"
echo
echo "allowed eval cards are only: {$ALLOWED_GPUS}. GPU 6/7 are production — leave them alone."
