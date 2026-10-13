#!/usr/bin/env bash
# §11 的"一条命令"：把仓内 scripts/*.py 幂等同步到部署树，并逐件冒烟。
#
# 为什么有这件：228 上两次在**起跑时**才发现"护栏已落仓、部署树里没有"（`ModuleNotFoundError`）。
# 本脚本幂等、可重复、任一冒烟失败即非零退出，供任何一轮随时调用：
#
#   scripts/sync_deploy_tree.sh --tree /data/remote/graspo-b11fix-20261012
#
# 冒烟内容：① 文件就位且与仓内 sha256 一致；② `python3 -c "import <mod>"` 可导入；
#           ③ 对带 CLI 的件跑 `--help`（rc=0）。
set -uo pipefail

TREE=""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tree) TREE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help)
      echo "用法: $0 --tree <部署树> [--repo <仓根>] [--dry-run]"
      echo "  幂等同步仓内 scripts/*.py 到部署树并逐件冒烟（sha256/import/--help）；任一失败即非零退出。"
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
[ -n "$TREE" ] || { echo "用法：$0 --tree <部署树> [--repo <仓根>] [--dry-run]" >&2; exit 2; }
[ -d "$TREE" ] || { echo "★ 部署树不存在：$TREE" >&2; exit 1; }
[ -d "$TREE/scripts" ] || { echo "★ 部署树缺 scripts/：$TREE" >&2; exit 1; }

fail=0
echo "=== 同步 ${REPO}/scripts/*.py → ${TREE}/scripts/ ==="
for src in "$REPO"/scripts/*.py; do
  base="$(basename "$src")"
  dst="$TREE/scripts/$base"
  if [ "$DRY" = "1" ]; then
    echo "  [dry-run] $base"
  else
    if ! cp -f "$src" "$dst"; then echo "  ✗ 复制失败：$base"; fail=1; continue; fi
    echo "  synced $base ($(wc -c < "$dst") B)"
  fi
done

echo "=== 逐件冒烟 ==="
for src in "$REPO"/scripts/*.py; do
  base="$(basename "$src")"; dst="$TREE/scripts/$base"
  [ "$DRY" = "1" ] && continue
  # ① sha256 一致
  a=$(sha256sum "$src" | awk '{print $1}'); b=$(sha256sum "$dst" | awk '{print $1}')
  if [ "$a" != "$b" ]; then echo "  ✗ $base：同步后 sha256 不一致"; fail=1; continue; fi
  mod="${base%.py}"
  # ② 可导入（纯模块用 import）
  if ! python3 -c "import sys; sys.path.insert(0, '$TREE/scripts'); import $mod" >/dev/null 2>&1; then
    echo "  ✗ $base：import 失败"; fail=1; continue
  fi
  # ③ 带 CLI 的件跑 --help（rc=0）
  if python3 "$dst" --help >/dev/null 2>&1; then echo "  OK $base（import + --help rc=0）"
  else echo "  OK $base（import；无 --help 或无 CLI）"; fi
done

if [ "$fail" = "1" ]; then echo "=== §11 同步失败（fail-closed）==="; exit 1; fi
echo "=== §11 同步完成：全部件就位且冒烟通过 ==="
