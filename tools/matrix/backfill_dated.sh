#!/usr/bin/env bash
# backfill_dated.sh — 回填防呆 wrapper（2026-10-22 基线轮）：产物只落「带日期副本对」，原版只读。
# 钉死：--data-in=副本数据件 / --html-out=副本人读件 / --template=共用模板 / --ledger-out=.local 台账
# 放行：--runs-dir / --only / --dry-run（原样透传）
# ⛔ 把 --data-in/--html-out/--data-out 指向仓内原版 ⇒ fail-closed 退出 2（这就是"不许动原版"的技术闸门）
# 用法：bash tools/matrix/backfill_dated.sh --runs-dir <dir> [--only M001] [--dry-run]
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
TOOL="$REPO/tools/matrix/backfill_matrix.py"
TEMPLATE="$REPO/tools/matrix/template.html"
DATED_DATA="$REPO/docs/capability-matrix-20261022.data.json"
DATED_HTML="$REPO/docs/GRASPO 训练框架功能测试报告-20261022.html"
ORIG_DATA="$REPO/docs/capability-matrix.data.json"
ORIG_HTML="$REPO/docs/GRASPO 训练框架功能测试报告.html"
LEDGER="$REPO/.local/hb-workspace/20261022-baseline-r1/backfill-ledger.md"

usage() {
  cat <<TXT
用法: bash tools/matrix/backfill_dated.sh --runs-dir <跑次日录> [选项]

  钉死（不可改）: --data-in  $DATED_DATA
                  --html-out $DATED_HTML
                  --template $TEMPLATE
                  --ledger-out $LEDGER
  透传:           --runs-dir <dir>  --only M001,M002  --dry-run
  例外:           --data-in/--html-out 可指向**非原版**路径（演练用，如 /tmp 副本）；
                  --ledger-out 可指向 .local/ 下路径（演练台账）。
  ⛔ 指向原版（$ORIG_DATA / $ORIG_HTML）一律拒绝并退出 2。
TXT
}

die() { printf '⛔ backfill_dated.sh 拒绝执行：%s\n' "$*" >&2; exit 2; }

# 规范化：相对路径按仓根解析；realpath -m（文件可不存在）
canon() { local p="$1"; [[ "$p" = /* ]] || p="$REPO/$p"; realpath -m -- "$p"; }
ORIG_DATA_C="$(canon "$ORIG_DATA")"; ORIG_HTML_C="$(canon "$ORIG_HTML")"

data_in="$DATED_DATA"; html_out="$DATED_HTML"; ledger="$LEDGER"; passthru=()
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --data-in|--html-out|--data-out|--template|--ledger-out)
      opt="$1"; [[ $# -ge 2 ]] || die "$opt 缺参数值"
      val="$2"; c="$(canon "$val")"; shift 2
      [[ "$opt" == --data-out ]] && die "--data-out 被钉死（缺省=--data-in，原地回填副本；⛔ 不允许另指）"
      [[ "$opt" == --template ]] && die "--template 被钉死为共用模板 $TEMPLATE（模板不复制）"
      [[ "$c" == "$ORIG_DATA_C" ]] && die "$opt 指向**原版数据件** $ORIG_DATA（本工作包只许回填副本；原版是冻结源）"
      [[ "$c" == "$ORIG_HTML_C" ]] && die "$opt 指向**原版人读件** $ORIG_HTML（只读冻结件，sha 31024648…470d9）"
      case "$opt" in
        --data-in) data_in="$c" ;;
        --html-out) html_out="$c" ;;
        --ledger-out) [[ "$c" == "$REPO/.local/"* ]] || die "--ledger-out 只许落 .local/ 下（⛔ 不得在 docs/ 新增台账）：$val"
                      ledger="$c" ;;
      esac ;;
    --data-in=*|--html-out=*|--data-out=*|--template=*|--ledger-out=*)
      die "本 wrapper 不解析 ${1%%=*}=<值> 形式（避免绕过路径闸门）；请用「${1%%=*} <值>」" ;;
    --) shift; passthru+=(--); passthru+=("$@"); break ;;
    *) passthru+=("$1"); shift ;;
  esac
done

[[ -f "$data_in" ]] || die "数据件不存在：$data_in（副本需在工作包 A 落盘后创建，见 T1）"
printf '→ data-in   %s\n→ html-out  %s\n→ template  %s\n→ ledger-out %s\n→ 透传      %s\n' \
  "$data_in" "$html_out" "$TEMPLATE" "$ledger" "${passthru[*]:-（无）}"
exec "$PY" "$TOOL" --data-in "$data_in" --html-out "$html_out" --template "$TEMPLATE" \
     --ledger-out "$ledger" "${passthru[@]}"
