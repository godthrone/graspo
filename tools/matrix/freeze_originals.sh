#!/usr/bin/env bash
# freeze_originals.sh — 把矩阵链的**基准两件**设为物理只读（chmod 444），幂等可复跑。
#   ★ 2026-10-23（基准件唯一 + 按轮副本）：本脚本**只**管基准两件
#     （docs/GRASPO训练框架功能测试报告.html + docs/capability-matrix.data.json）；
#     按轮副本（docs/GRASPO训练框架功能测试报告-<date>.html / capability-matrix-<date>.data.json）
#     由 `new_round.sh` 建为 644 且**不进本脚本**（副本要可写才能回填；旧布局里那条带日期副本路径已随改名删除）。
# 为什么：19:20 一次无 --dry-run 的回填直调真实改写了这两件（随后逐字节恢复）。权限位是最后一道物理防线：
#   写路径 = 直调 backfill_matrix.py（wrapper 的路径闸门管不到）⇒ 只有文件系统层能兜住。
# 只做 chmod 444 + 打印 sha；不写内容、不碰 git、不复制、不改属主。
# ★ 合法更新流程（必须经批准）：chmod 644 两件 → 走正常链（回填/渲染/verify）→ 改完 chmod 444 复原。
#   恢复/更新后务必复跑 `bash tools/matrix/verify_dated.sh`（其中 ⑧ 查权限、① 查基准件 sha）。
# 用法：bash tools/matrix/freeze_originals.sh
#   ★ 路径取自 tools/matrix/paths.py（GRASPO_BASELINE_HTML 可覆盖基准人读件路径）。
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$REPO/tools/matrix"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器\n' >&2; exit 1; }
eval "$("$PY" "$M/paths.py" --sh)" || { printf '⛔ paths.py 取路径失败\n' >&2; exit 1; }

ORIG_HTML="$PATHS_BASELINE_HTML"
ORIG_DATA="$PATHS_BASELINE_DATA"

rc=0
for f in "$ORIG_HTML" "$ORIG_DATA"; do
  [[ -f "$f" ]] || { printf '✗ 缺失：%s\n' "$f" >&2; rc=1; continue; }
  chmod "$PATHS_FROZEN_MODE" -- "$f" || { printf '✗ chmod %s 失败：%s\n' "$PATHS_FROZEN_MODE" "$f" >&2; rc=1; continue; }
  printf '%s  %s  %s\n' "$(stat -c%a -- "$f")" "$(sha256sum -- "$f" | awk '{print $1}')" "$(basename -- "$f")"
done
printf '# 基准两件已设为 %s（只读）；合法更新：chmod 644 → 改 → chmod 444 → 复跑 verify_dated.sh\n' "$PATHS_FROZEN_MODE"
printf '# 按轮副本（本轮：%s）不在本脚本范围内 —— 副本须可写才能回填，由 new_round.sh 建为 %s\n' \
  "$PATHS_ROUND_HTML_NAME" "$PATHS_ROUND_MODE"
exit "$rc"
