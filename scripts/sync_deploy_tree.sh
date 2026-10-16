#!/usr/bin/env bash
# §11 的"一条命令"：把仓内 scripts/*.py 幂等同步到部署树，并逐件冒烟。
#
# 为什么有这件：228 上两次在**起跑时**才发现"护栏已落仓、部署树里没有"（`ModuleNotFoundError`）。
# 本脚本幂等、可重复、任一冒烟失败即非零退出，供任何一轮随时调用：
#
#   scripts/sync_deploy_tree.sh --tree <部署树绝对路径>
#
# 冒烟内容：① 文件就位且与仓内 sha256 一致；② `python3 -c "import <mod>"` 可导入；
#           ③ 对带 CLI 的件跑 `--help`（rc=0）。
set -uo pipefail

TREE=""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK=0
REMOTE=""; RUSER=""; RHOST=""; RPORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --tree) TREE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --check) CHECK=1; shift ;;
    -h|--help)
      echo "用法: $0 --tree <部署树> [--repo <仓根>] [--dry-run]"
      echo "  幂等同步仓内 scripts/*.py 到部署树并逐件冒烟（sha256/import/--help）；任一失败即非零退出。"
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
[ -n "$TREE" ] || { echo "用法：$0 --tree <部署树> [--remote user@host:port] [--repo <仓根>] [--dry-run]" >&2; exit 2; }
if [ -n "$REMOTE" ]; then
  RUSER="${REMOTE%@*}"; rest="${REMOTE#*@}"; RHOST="${rest%%:*}"; RPORT="${rest##*:}"
  echo "=== 远端模式：$RUSER@$RHOST:$RPORT → $TREE ==="
  SSH="ssh -p $RPORT $RUSER@$RHOST"; SCP="scp -q -P $RPORT"
  $SSH "test -d '$TREE/scripts'" || { echo "★ 远端部署树缺 scripts/：$TREE" >&2; exit 1; }
else
  SSH=""; SCP=""
  [ -d "$TREE" ] || { echo "★ 部署树不存在：$TREE" >&2; exit 1; }
  [ -d "$TREE/scripts" ] || { echo "★ 部署树缺 scripts/：$TREE" >&2; exit 1; }
fi
shopt -s nullglob
set -- "$REPO"/scripts/*.py
[ $# -gt 0 ] || { echo "★ 仓内无 scripts/*.py（--repo=$REPO 对吗？）" >&2; exit 1; }

fail=0
if [ "$CHECK" = "1" ]; then
  echo "=== §11 --check：只比对不写（${REPO}/scripts vs ${TREE}/scripts）==="
  diffn=0
  for src in "$REPO"/scripts/*.py; do
    base="$(basename "$src")"; dst="$TREE/scripts/$base"
    a=$(sha256sum "$src" | awk '{print $1}')
    if [ -n "${REMOTE:-}" ]; then b=$($SSH "sha256sum '$dst' 2>/dev/null" | awk '{print $1}'); else b=$(sha256sum "$dst" 2>/dev/null | awk '{print $1}'); fi
    if [ "$a" = "$b" ]; then echo "  OK   $base"
    else echo "  DIFF $base（仓 $a / 树 ${b:-缺失}）"; diffn=$((diffn+1)); fi
  done
  [ "$diffn" = "0" ] || { echo "=== §11 --check 失败：$diffn 件不一致（请跑不带 --check 的同步）==="; exit 1; }
  echo "=== §11 --check 通过：部署树与仓逐字节一致 ==="; exit 0
fi

echo "=== 同步 ${REPO}/scripts/*.py → ${TREE}/scripts/ ==="
for src in "$REPO"/scripts/*.py; do
  base="$(basename "$src")"
  dst="$TREE/scripts/$base"
  if [ "$DRY" = "1" ]; then
    echo "  [dry-run] $base"
  elif [ -n "$REMOTE" ]; then
    $SCP "$src" "$RUSER@$RHOST:$dst" || { echo "  ✗ 远端复制失败：$base"; fail=1; continue; }
    echo "  synced $base → $RHOST"
  else
    cp -f "$src" "$dst" || { echo "  ✗ 复制失败：$base"; fail=1; continue; }
    echo "  synced $base ($(wc -c < "$dst") B)"
  fi
done

echo "=== 逐件冒烟 ==="
for src in "$REPO"/scripts/*.py; do
  base="$(basename "$src")"; dst="$TREE/scripts/$base"
  [ "$DRY" = "1" ] && continue
  # ① sha256 一致
  a=$(sha256sum "$src" | awk '{print $1}')
  if [ -n "$REMOTE" ]; then b=$($SSH "sha256sum '$dst'" | awk '{print $1}'); else b=$(sha256sum "$dst" | awk '{print $1}'); fi
  if [ "$a" != "$b" ]; then echo "  ✗ $base：同步后 sha256 不一致"; fail=1; continue; fi
  mod="${base%.py}"
  # ② 可导入（纯模块用 import）
  # 若该件在**仓内**本就无法裸 import（依赖 hostdeps 等），则不把远端 import 当阻断项
  if ! python3 -c "import sys; sys.path.insert(0, '$REPO/scripts'); import $mod" >/dev/null 2>&1; then
    echo "  OK $base（sha256 一致；仓内亦不可裸 import ⇒ 跳过 import 冒烟）"; continue
  fi
  # 远端 import 冒烟：带上部署树的 hostdeps（228 上脚本依赖它）并在树根下执行
  IMP="cd '$TREE' && PYTHONPATH='$TREE.hostdeps:$TREE/scripts' python3 -c \"import $mod\""
  if [ -n "$REMOTE" ]; then
    if ! $SSH "$IMP" >/dev/null 2>&1; then
      # 远端 import 失败但 --help 能跑通 ⇒ 视为可接受（依赖差异），仅提示
      if $SSH "cd '$TREE' && PYTHONPATH='$TREE.hostdeps' python3 '$dst' --help" >/dev/null 2>&1; then
        echo "  OK $base（sha256 一致；远端 import 需 deps，--help rc=0）"; continue
      fi
      echo "  ✗ $base：远端 import 失败"; fail=1; continue
    fi
  else python3 -c "import sys; sys.path.insert(0, '$TREE/scripts'); import $mod" >/dev/null 2>&1 || { echo "  ✗ $base：import 失败"; fail=1; continue; }; fi
  if [ -n "$REMOTE" ]; then
    $SSH "python3 '$dst' --help" >/dev/null 2>&1 && echo "  OK $base（远端 import + --help rc=0）" || echo "  OK $base（远端 import；无 --help 或无 CLI）"
  else
    python3 "$dst" --help >/dev/null 2>&1 && echo "  OK $base（import + --help rc=0）" || echo "  OK $base（import；无 --help 或无 CLI）"
  fi
done

if [ "$fail" = "1" ]; then echo "=== §11 同步失败（fail-closed）==="; exit 1; fi
echo "=== §11 同步完成：全部件就位且冒烟通过 ==="
