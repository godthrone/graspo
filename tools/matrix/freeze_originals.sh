#!/usr/bin/env bash
# freeze_originals.sh — 2026-10-22 基线轮：把矩阵链的**原版两件**设为物理只读（chmod 444），幂等可复跑。
# 为什么：19:20 一次无 --dry-run 的回填直调真实改写了这两件（随后逐字节恢复）。权限位是最后一道物理防线：
#   写路径 = 直调 backfill_matrix.py（wrapper 的路径闸门管不到）⇒ 只有文件系统层能兜住。
# 只做 chmod 444 + 打印 sha；不写内容、不碰 git、不复制、不改属主。
# ★ 合法更新流程（必须经批准）：chmod 644 两件 → 走正常链（回填/渲染/verify）→ 改完 chmod 444 复原。
#   恢复/更新后务必复跑 `bash tools/matrix/verify_dated.sh`（其中 ⑧ 查权限、① 查原版 sha）。
# 用法：bash tools/matrix/freeze_originals.sh
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG_HTML="$REPO/docs/GRASPO 训练框架功能测试报告.html"
ORIG_DATA="$REPO/docs/capability-matrix.data.json"

rc=0
for f in "$ORIG_HTML" "$ORIG_DATA"; do
  [[ -f "$f" ]] || { printf '✗ 缺失：%s\n' "$f" >&2; rc=1; continue; }
  chmod 444 -- "$f" || { printf '✗ chmod 444 失败：%s\n' "$f" >&2; rc=1; continue; }
  printf '%s  %s  %s\n' "$(stat -c%a -- "$f")" "$(sha256sum -- "$f" | awk '{print $1}')" "$(basename -- "$f")"
done
printf '# 原版两件已设为 444（只读）；合法更新：chmod 644 → 改 → chmod 444 → 复跑 verify_dated.sh\n'
exit "$rc"
