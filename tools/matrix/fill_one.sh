#!/usr/bin/env bash
# fill_one.sh — **单档回填**：一条命令把某一档的实测读数单独写进「当轮日期副本」。
#
#   用法: bash tools/matrix/fill_one.sh <M档号> [--dry-run]
#   例:   bash tools/matrix/fill_one.sh M041 --dry-run   # 只打印该行 before/after，不落盘
#         bash tools/matrix/fill_one.sh M041            # 真写（幂等：同一档重复跑 sha256 不变）
#
# ★ 为什么本文件这么短：真正的逻辑在 `tools/matrix/fill_one.py`（同目录），本文件只做两件事：
#   ① 选解释器；② **容器门槛**——落盘链 `backfill_matrix.py → verify_matrix.py → build_official.py
#      /gen_configs.py` 要 `import graspo`（⇒ pydantic 等）。宿主机（228）**没有**装 graspo 包，
#      宿主 python3 会在 `import pydantic` 处 `ModuleNotFoundError` ⇒ 这一步必须进容器跑。
#      ⇒ 宿主上直接跑本脚本时，若发现缺依赖且 docker 可用，就**自动**把自身送进训练镜像里再跑一遍
#      （同样的绝对路径挂载，产物仍落宿主同一份树；`--user $(id -u):$(id -g)` ⇒ 不产生 root 文件）。
#      ⛔ 不自动改任何配置/数据；容器只读挂载树的父目录（用于找 eval 归档），树本身可写（写副本）。
#
# ★ 退出码：0 成功 / 2 用法或非法档号 / 3 证据不足（保持「待测试」）/ 4 闸门失败（已回滚）
#            / 5 基准件三锁不成立 / 9 解释器环境不足且无法进容器（下面会打印可直接复制的容器命令）
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器\n' >&2; exit 1; }

IMAGE="${GRASPO_IMAGE:-graspo:v0.28.11-cu130fix-v2-msgspec-20261012-run}"
ENTRY="bash $REPO/tools/matrix/fill_one.sh"

# 容器门槛：只有**落盘链**需要 graspo/pydantic；读数派生（make_result_json）是纯标准库。
if ! "$PY" -c 'import pydantic' >/dev/null 2>&1; then
  if [[ "${FILL_ONE_IN_CONTAINER:-0}" == "1" ]]; then
    printf '⛔ 已在容器内但 import pydantic 仍失败 ⇒ 镜像/挂载不对，拒绝继续（rc=9）\n' >&2
    exit 9
  fi
  DOCKER_CMD=(docker run --rm
    --user "$(id -u):$(id -g)"
    --ipc=host --shm-size=2g
    -e "PYTHONPATH=$REPO/src"
    -e HOME=/tmp/graspo-fill-one-home
    -e FILL_ONE_IN_CONTAINER=1
    -v "$REPO:$REPO"
    -v "$(dirname -- "$REPO")":"$(dirname -- "$REPO")":ro
    -w "$REPO")
  # ★ GRASPO_FILL_RECOVER_TIMEOUT_ARTIFACT 必须一起带进容器：读数派生在容器里跑
  #   （见 fill_one.py 的同名开关；缺这条转发 ⇒ 开关在宿主设了、容器里却看不到）。
  for v in GRASPO_ROUND_DATE GRASPO_RUN_ROOT GRASPO_EVAL_ROOT GRASPO_FILL_LEDGER \
           GRASPO_FILL_RECOVER_TIMEOUT_ARTIFACT GRASPO_FILL_RECOVER_OPERATOR_STOP; do
    [[ -n "${!v:-}" ]] && DOCKER_CMD+=(-e "$v=${!v}")
  done
  DOCKER_CMD+=("$IMAGE" $ENTRY "$@")
  printf '→ 宿主解释器 %s 缺 graspo/pydantic（落盘链要 import graspo）⇒ 送进容器跑：\n' "$PY"
  printf '   %s\n' "${DOCKER_CMD[*]}"
  if [[ "${GRASPO_FILL_NO_DOCKER:-0}" == "1" ]] || ! command -v docker >/dev/null 2>&1; then
    printf '⛔ 未进容器（GRASPO_FILL_NO_DOCKER=1 或无 docker）⇒ 未做任何事（rc=9）。\n' >&2
    printf '   请复制上面那条 docker run 命令直接执行。\n' >&2
    exit 9
  fi
  exec "${DOCKER_CMD[@]}"
fi

exec "$PY" "$REPO/tools/matrix/fill_one.py" "$@"
