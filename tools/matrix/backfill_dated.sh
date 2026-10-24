#!/usr/bin/env bash
# backfill_dated.sh — 回填防呆 wrapper：产物只落「按轮副本对」，**基准件只读**。
# 钉死：--data-in=副本数据件 / --html-out=副本人读件 / --template=共用模板 / --ledger-out=.local 台账
# ★ 2026-10-23（基准件唯一 + 按轮副本）：副本由 `tools/matrix/new_round.sh` 派生。
#   本轮副本**不存在** ⇒ 本 wrapper 先自动跑 `new_round.sh --date <轮次>` 派生（cp -p 基准件 + chmod 644
#   + 副本数据件的 human_view_path 指向副本），再回填；⛔ 任何情况下都不写基准件。
#   轮次取自 `--date YYYYMMDD` 或 `$GRASPO_ROUND_DATE`（缺省 = 今天）；路径全部来自 `tools/matrix/paths.py`。
# ★ R1-2（对应审查 🔴-2）：**取消自由透传**。本 wrapper 只认下面这张白名单，其余一律 rc=2 拒起：
#   透传白名单：--runs-dir <dir>  --only M001,M002  --dry-run  --date YYYYMMDD  -h/--help
#   演练例外（值只许指向**非基准**路径）：--data-in <非基准>  --html-out <非基准>  --ledger-out <.local/*>
#   ⛔ `=` 赋值形式（--data-in=… / --html=…）一律 rc=2；⛔ argparse 缩写（--data-i / --html / --le）一律 rc=2；
#   ⛔ 任何未登记参数（含 `--`、短选项、位置参数）一律 rc=2 —— 旧版此处把未识别 `--*` 塞进 passthru，
#      而 backfill_matrix.py 的 argparse 默认 allow_abbrev=True 且钉死值排在 argv 前面 ⇒ 后到的缩写会覆盖钉死值
#      （实测 `--html docs/…基准件.html` 能写到冻结件上）。现三道闸门：
#        ①本 wrapper 白名单（用户串永远不进选项位）
#        ②全部参数由本 wrapper **显式给全**（不再有任何自由透传片段；Python 3.12 的 argparse 连尾部 `--` 都判
#          unrecognized，故不用 `--` 而用「显式给全」这条路，等价达到"用户串不被当选项解析"）
#        ③backfill_matrix.py 的 ArgumentParser(allow_abbrev=False)（根治，绕过 wrapper 直调也不认缩写）
# ⛔ 基准件只读：值经 realpath -m 规范化（消解相对路径/`..`/软链）＋ 同 inode（-ef，堵硬链接）比对基准两件 ⇒ rc=2
# ⛔ 台账只许落 .local/ 下（⛔ 不得在 docs/ 新增台账）
# ⛔ --runs-dir 内没有任何 <档号>.json ⇒ rc=3（"无待回填 json，疑似落点错误"，防静默零写入）
# 用法：bash tools/matrix/backfill_dated.sh --runs-dir <dir> [--date YYYYMMDD] [--only M001] [--dry-run]
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$REPO/tools/matrix"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器\n' >&2; exit 1; }
TOOL="$M/backfill_matrix.py"
NEW_ROUND="$M/new_round.sh"

die() { printf '⛔ backfill_dated.sh 拒绝执行：%s\n' "$*" >&2; exit 2; }

# ★ 预扫 --date（**只**取日期，其余参数仍走下面的白名单）：路径常量依赖轮次日期，
#   必须在 `paths.py --sh` 之前拿到。格式不符 ⇒ rc=2（⛔ 不猜）。
date_arg=""
prev=""
for a in "$@"; do
  if [[ "$prev" == "--date" ]]; then date_arg="$a"; prev=""; continue; fi
  case "$a" in
    --date=*) die "--date 不认「=<值>」形式，请用「--date <YYYYMMDD>」" ;;
  esac
  prev="$a"
done
if [[ -n "$date_arg" ]]; then
  [[ "$date_arg" =~ ^[0-9]{8}$ ]] || die "--date 须为 YYYYMMDD，实际 $date_arg"
  export GRASPO_ROUND_DATE="$date_arg"
fi
eval "$("$PY" "$M/paths.py" --sh)" || die "paths.py 取路径失败"

TEMPLATE="$PATHS_TEMPLATE"
DATED_DATA="$PATHS_ROUND_DATA"
DATED_HTML="$PATHS_ROUND_HTML"
ORIG_DATA="$PATHS_BASELINE_DATA"
ORIG_HTML="$PATHS_BASELINE_HTML"
LEDGER="$REPO/.local/hb-workspace/20261022-baseline-r1/backfill-ledger.md"

usage() {
  cat <<TXT
用法: bash tools/matrix/backfill_dated.sh --runs-dir <跑次目录> [选项]

  钉死（不可改）: --data-in  $DATED_DATA
                  --html-out $DATED_HTML
                  --template $TEMPLATE
                  --ledger-out $LEDGER
  白名单:         --runs-dir <dir>  --date YYYYMMDD  --only M001,M002  --dry-run  -h/--help
  例外:           --data-in/--html-out 可指向**非基准**路径（演练用，如 /tmp 副本）；
                  --ledger-out 可指向 .local/ 下路径（演练台账）。
  ⛔ 指向基准件（$ORIG_DATA / $ORIG_HTML）一律拒绝并退出 2（相对/绝对/上跳/软链/硬链接同判）。
  ⛔ 「=」赋值形式、argparse 缩写、其它任何未登记参数，一律拒绝并退出 2（本 wrapper 无自由透传）。
  ⛔ --runs-dir 内没有任何 <档号>.json ⇒ 退出 3（无待回填 json，疑似落点错误：应指向 <run-root>/collect）。
  ★ 本轮副本不存在 ⇒ 先自动派生（cp -p 基准件 + chmod $PATHS_ROUND_MODE + human_view_path 指向副本），再回填。
    ⛔ 基准件（$PATHS_FROZEN_MODE 只读）任何情况下都不写。
TXT
}

# 规范化：相对路径按仓根解析；realpath -m（文件可不存在）
canon() { local p="$1"; [[ "$p" = /* ]] || p="$REPO/$p"; realpath -m -- "$p"; }
ORIG_DATA_C="$(canon "$ORIG_DATA")"; ORIG_HTML_C="$(canon "$ORIG_HTML")"

# 基准件只读判定：规范化路径相等，或与基准件同 inode（硬链接/绑定挂载 ⇒ 写它等于改写冻结件）
check_not_orig() {
  local opt="$1" val="$2" c="$3"
  [[ "$c" == "$ORIG_DATA_C" ]] && die "$opt 指向**基准数据件** $ORIG_DATA（只许回填副本；基准件是冻结源）"
  [[ "$c" == "$ORIG_HTML_C" ]] && die "$opt 指向**基准人读件** $ORIG_HTML（只读冻结件，sha ${PATHS_BASELINE_HTML_SHA256:0:8}…）"
  if [[ -e "$c" ]] && { [[ "$c" -ef "$ORIG_DATA_C" ]] || [[ "$c" -ef "$ORIG_HTML_C" ]]; }; then
    die "$opt 与基准件同 inode（硬链接/绑定挂载）：$val ⇒ 写它等于改写冻结件"
  fi
}

data_in="$DATED_DATA"; html_out="$DATED_HTML"; ledger="$LEDGER"
runs_dir=""; only=""; dry_run=0
data_in_override=0; html_out_override=0
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --date) [[ $# -ge 2 ]] || die "--date 缺参数值"; shift 2 ;;   # 值已在预扫里校验
    --data-in|--html-out|--data-out|--template|--ledger-out|--runs-dir|--only)
      opt="$1"; [[ $# -ge 2 ]] || die "$opt 缺参数值"
      [[ "$2" == -* ]] && die "$opt 的值不得以 '-' 开头（$2）——值会落到选项位上"
      val="$2"; shift 2
      case "$opt" in
        --data-out) die "--data-out 被钉死（缺省=--data-in，原地回填副本；⛔ 不允许另指）" ;;
        --template) die "--template 被钉死为共用模板 $TEMPLATE（模板不复制）" ;;
        --data-in)  c="$(canon "$val")"; check_not_orig "$opt" "$val" "$c"; data_in="$c"; data_in_override=1 ;;
        --html-out) c="$(canon "$val")"; check_not_orig "$opt" "$val" "$c"; html_out="$c"; html_out_override=1 ;;
        --ledger-out) c="$(canon "$val")"
                      check_not_orig "$opt" "$val" "$c"
                      [[ "$c" == "$REPO/.local/"* ]] || die "--ledger-out 只许落 .local/ 下（⛔ 不得在 docs/ 新增台账）：$val"
                      ledger="$c" ;;
        --runs-dir) [[ -d "$val" ]] || die "跑次目录不存在：$val" ; runs_dir="$(canon "$val")" ;;
        --only)     [[ "$val" =~ ^M[0-9]{3}(,M[0-9]{3})*$ ]] \
                      || die "--only 只接受逗号分隔的 M 档号（如 M001,M011），实际：$val" ; only="$val" ;;
      esac ;;
    --dry-run) dry_run=1; shift ;;
    --*=*) die "不解析 ${1%%=*}=<值> 形式（含缩写/全名；避免绕过路径闸门）；请用「${1%%=*} <值>」" ;;
    --*|-*) die "未识别参数 $1：本 wrapper 只放行 --runs-dir/--date/--only/--dry-run/-h/--help 与演练用 --data-in/--html-out/--ledger-out；⛔ 无自由透传（argparse 缩写如 --data-i/--html/--le 一律拒）" ;;
    *) die "多余位置参数 $1：本 wrapper 不接受位置参数（可能是把 --only 的值放错了位置）" ;;
  esac
done

[[ -n "$runs_dir" ]] || die "--runs-dir 必填（本 wrapper 无自由透传，见用法）"

# ★ 按轮副本缺失 ⇒ 先派生（⛔ 永不写基准件；派生器自己校验基准件 sha 与 444）
if (( ! data_in_override && ! html_out_override )) && [[ ! -f "$data_in" || ! -f "$html_out" ]]; then
  printf '→ 本轮副本不全（%s）⇒ 先派生：bash %s --date %s\n' \
    "$PATHS_ROUND_HTML_NAME" "$NEW_ROUND" "$PATHS_ROUND_DATE"
  if (( dry_run )); then
    bash "$NEW_ROUND" --date "$PATHS_ROUND_DATE" --dry-run || die "派生 dry-run 失败"
  else
    bash "$NEW_ROUND" --date "$PATHS_ROUND_DATE" || die "派生副本失败（见上；⛔ 零写入回填）"
  fi
fi
# ★ dry-run 时派生器同样只打印计划（不落盘）⇒ 副本可能仍不存在：允许这次空跑，但明说零写入
if (( dry_run )) && [[ ! -f "$data_in" || ! -f "$html_out" ]]; then
  printf '→ 【dry-run】本轮副本尚未落盘；正式跑时本 wrapper 会先派生再回填 —— 本次**零写入**（rc=0）\n'
  exit 0
fi
[[ -f "$data_in" ]] || die "数据件不存在：$data_in（先 bash $NEW_ROUND --date $PATHS_ROUND_DATE 派生副本）"
[[ -f "$html_out" ]] || die "人读件不存在：$html_out（先 bash $NEW_ROUND --date $PATHS_ROUND_DATE 派生副本）"

# ★ 空落点闸门（对应审查 🟡-3）：目录里一个 <档号>.json 都没有 ⇒ 不是"成功"，是落点错了
shopt -s nullglob
jsons=("$runs_dir"/*.json)
shopt -u nullglob
n_json=0
for j in "${jsons[@]}"; do [[ -f "$j" ]] && n_json=$((n_json + 1)); done
if ((n_json == 0)); then
  printf '⛔ backfill_dated.sh 拒绝执行：无待回填 json，疑似落点错误（应指向 <run-root>/collect）\n' >&2
  printf '   --runs-dir %s 下未找到任何 <档号>.json；本次**零写入**（rc=3，⛔ 不是成功）\n' "$runs_dir" >&2
  exit 3
fi

printf '→ date      %s\n→ data-in   %s\n→ html-out  %s\n→ template  %s\n→ ledger-out %s\n→ runs-dir  %s（%d 份 json）\n→ only      %s\n→ dry-run   %s\n' \
  "$PATHS_ROUND_DATE" "$data_in" "$html_out" "$TEMPLATE" "$ledger" "$runs_dir" "$n_json" "${only:-（全部）}" "$dry_run"

# ★ 显式给全所有参数（用户串只出现在「值」位，永不进入选项位；不再有 passthru 片段）
argv=(--data-in "$data_in" --html-out "$html_out" --template "$TEMPLATE"
      --ledger-out "$ledger" --runs-dir "$runs_dir")
if [[ -n "$only" ]]; then argv+=(--only "$only"); fi
if ((dry_run)); then argv+=(--dry-run); fi
exec "$PY" "$TOOL" "${argv[@]}"
