#!/usr/bin/env bash
# sync_results.sh — 把「当轮已回填的两份结果件」从 228 在役树拉回本仓，并在落盘前后各做一遍硬校验。
#
# ★ 一句话：228 那边 `fill_one.sh` 填完一档，就在**本仓**跑一次本脚本 ⇒ 主席在本机直接看见进度。
#
# 用法：
#   bash tools/matrix/sync_results.sh                    # 当轮 = 今天（paths.py 缺省）
#   bash tools/matrix/sync_results.sh --date 20261025    # 指定轮次
#   GRASPO_ROUND_DATE=20261025 bash tools/matrix/sync_results.sh
#
# 拉什么（<date> = 轮次日期 YYYYMMDD；落点 = 本仓 docs/ 同名文件，即「按轮副本」）：
#   228:<远端 docs>/GRASPO训练框架功能测试报告-<date>.html   -> docs/GRASPO训练框架功能测试报告-<date>.html
#   228:<远端 docs>/capability-matrix-<date>.data.json       -> docs/capability-matrix-<date>.data.json
#   ⛔ 基准件（docs/GRASPO训练框架功能测试报告.html · 444 · 冻结）**只读不写**：脚本落盘前按
#      realpath/inode 拒写，并前后各查一次「sha / 字节 / 权限」三锁（任一不符 ⇒ rc≠0 且不落盘）。
#
# 自动校验（相当于工作包 2a/2b/2c）：
#   2a 逐字节相同   远端 `sha256sum` vs 暂存区 sha vs 落盘后 sha，三者必须全等（外加字节数）；
#   2b 基准件三锁   sha256 = paths.BASELINE_HTML_SHA256 · 字节 = paths.BASELINE_HTML_BYTES · 权限 = 444；
#   2c 写范围闸门   check_fill_scope.py --base <基准件> --filled <暂存的副本 html> --data <暂存的副本数据件>
#                   ⇒ 必须 rc=0 且输出含 `SCOPE PASS`。
#
# 失败语义（fail-closed）：**先全部校验，后一次性落盘**。任一校验不过 ⇒ 本仓 docs/ 一个字节都不改，
#   rc≠0（暂存区在 .local/tmp/ 下，退出时自动清理）。
# 幂等：远端 sha 与本仓在位 sha 相同 ⇒ 直接跳过写入（不 cp、不动 mtime），因此同一状态重复跑不产生 diff。
#
# 退出码：0 = 校验通过且本仓已与 228 一致（含「本来就已经一致」）；1 = 校验不过 / 拉取失败（本仓未改）。
#
# 可覆盖的环境变量（⛔ 不增删其它）：
#   GRASPO_RESULT_HOST   远端 SSH 目标      缺省 user@192.0.2.10
#   GRASPO_RESULT_PORT   远端 SSH 端口      缺省 22
#   GRASPO_RESULT_DOCS   远端 docs 绝对路径 缺省 /data/remote/graspo-baseline-r1-ebd5968/docs
#   GRASPO_ROUND_DATE    轮次日期 YYYYMMDD  （同 --date；缺省 = paths.py 的今天）
#   GRASPO_BASELINE_HTML 基准人读件路径     缺省 = 仓内基准件（仅供演练覆盖）
#
# 依赖：ssh / scp / sha256sum / stat + python3（走仓内 .venv，缺失则退回系统 python3）。
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$REPO/tools/matrix"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器\n' >&2; exit 1; }

HOST="${GRASPO_RESULT_HOST:-user@192.0.2.10}"
PORT="${GRASPO_RESULT_PORT:-22}"
REMOTE_DOCS="${GRASPO_RESULT_DOCS:-/data/remote/graspo-baseline-r1-ebd5968/docs}"
SSH_COMMON=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
ssh() { command ssh "${SSH_COMMON[@]}" -p "$PORT" "$@"; }
scp() { command scp "${SSH_COMMON[@]}" -P "$PORT" "$@"; }

date_arg=""
while (($#)); do
  case "$1" in
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --date) [[ $# -ge 2 ]] || { printf '⛔ --date 缺参数值\n' >&2; exit 2; }; date_arg="$2"; shift 2 ;;
    --date=*) printf '⛔ 不解析 --date=<值>，请用「--date <值>」\n' >&2; exit 2 ;;
    *) printf '⛔ 未识别参数 %s（见 --help）\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$date_arg" ]] && export GRASPO_ROUND_DATE="$date_arg"
eval "$("$PY" "$M/paths.py" --sh)" || { printf '⛔ paths.py 取路径失败\n' >&2; exit 1; }

# ★ 2026-10-28：写范围闸门的比对基线 = **模板基线**（当前模板 + 空白种子渲染）；
#   冻结原件退为历史基准件（三锁仍由 verify_dated ① 断言）。GRASPO_BASELINE_HTML 仅演练覆盖用。
BASE_HTML="${GRASPO_BASELINE_HTML:-$PATHS_TEMPLATE_BASELINE_HTML}"
ROUND_HTML="$PATHS_ROUND_HTML"
ROUND_DATA="$PATHS_ROUND_DATA"
HTML_NAME="$PATHS_ROUND_HTML_NAME"
DATA_NAME="$PATHS_ROUND_DATA_NAME"
SHA_EXPECT="$PATHS_BASELINE_HTML_SHA256"
BYTES_EXPECT="$PATHS_BASELINE_HTML_BYTES"
MODE_EXPECT="$PATHS_FROZEN_MODE"
MODE_ROUND="$PATHS_ROUND_MODE"

fail=0
say() { printf '%s\n' "$*"; }
bad() { fail=1; printf '  ✗ FAIL: %s\n' "$*" >&2; }
ok()  { printf '  ✓ %s\n' "$*"; }

# ---- 0. 防呆：落点不得是基准件（真实路径 / inode 双保险）-------------------------
for dst in "$ROUND_HTML" "$ROUND_DATA"; do
  [[ "$(realpath -m "$dst")" != "$(realpath -m "$BASE_HTML")" ]] || {
    printf '⛔ 落点与基准件同一路径（%s），拒绝执行\n' "$dst" >&2; exit 1; }
done
[[ -f "$BASE_HTML" ]] || { printf '⛔ 基准件不存在：%s\n' "$BASE_HTML" >&2; exit 1; }
[[ "$(stat -c %i "$BASE_HTML")" != "$(stat -c %i "$ROUND_HTML" 2>/dev/null || echo -1)" ]] || {
  printf '⛔ 副本 html 与基准件同 inode（硬链接），拒绝执行\n' >&2; exit 1; }

# ---- 1. 基准件三锁（落盘前）---------------------------------------------------
check_baseline() {           # $1 = 阶段名
  local phase="$1" sha bytes mode
  sha="$(sha256sum "$BASE_HTML" | awk '{print $1}')"
  bytes="$(stat -c %s "$BASE_HTML")"
  mode="$(stat -c %a "$BASE_HTML")"
  say "【2b · 基准件三锁 / $phase】$BASE_HTML"
  [[ "$sha" == "$SHA_EXPECT" ]]   && ok "sha256 $sha" || bad "sha256 $sha ≠ 期望 $SHA_EXPECT"
  [[ "$bytes" == "$BYTES_EXPECT" ]] && ok "字节 $bytes B" || bad "字节 $bytes B ≠ 期望 $BYTES_EXPECT B"
  [[ "$mode" == "$MODE_EXPECT" ]]  && ok "权限 $mode" || bad "权限 $mode ≠ 期望 $MODE_EXPECT"
}
check_baseline "落盘前"
((fail == 0)) || { say ''; say '⛔ 基准件三锁已破 ⇒ 不落盘、不动本仓任何文件（rc=1）'; exit 1; }

# ---- 2. 远端读取 + 拉到暂存区（同仓文件系统，便于原子 mv）----------------------
mkdir -p "$REPO/.local/tmp"
STAGE="$(mktemp -d "$REPO/.local/tmp/sync-results-${PATHS_ROUND_DATE}.XXXXXX")" || {
  printf '⛔ 建暂存区失败\n' >&2; exit 1; }
trap 'rm -rf "$STAGE"' EXIT

say "【拉取】$HOST:$REMOTE_DOCS  →  暂存区"
remote_meta="$(ssh "$HOST" \
  "cd '$REMOTE_DOCS' && sha256sum -- '$HTML_NAME' '$DATA_NAME' && stat -c '%n %s' -- '$HTML_NAME' '$DATA_NAME'")" || {
  bad "远端读取失败（$HOST:$REMOTE_DOCS）—— 远端是否已开跑本轮？"; say '⛔ 本仓未改任何文件（rc=1）'; exit 1; }

r_sha_html="$(printf '%s\n' "$remote_meta" | awk -v n="$HTML_NAME" '$2==n{print $1}')"
r_sha_data="$(printf '%s\n' "$remote_meta" | awk -v n="$DATA_NAME" '$2==n{print $1}')"
r_by_html="$(printf '%s\n' "$remote_meta" | awk -v n="$HTML_NAME" '$1==n{print $2}')"
r_by_data="$(printf '%s\n' "$remote_meta" | awk -v n="$DATA_NAME" '$1==n{print $2}')"
for v in r_sha_html r_sha_data r_by_html r_by_data; do
  [[ -n "${!v}" ]] || { bad "远端回执缺字段 $v"; printf '%s\n' "$remote_meta" >&2; say '⛔ 本仓未改任何文件（rc=1）'; exit 1; }
done
say "  远端 $HTML_NAME  $r_by_html B  $r_sha_html"
say "  远端 $DATA_NAME  $r_by_data B  $r_sha_data"

scp -q "$HOST:$REMOTE_DOCS/$HTML_NAME" "$HOST:$REMOTE_DOCS/$DATA_NAME" "$STAGE/" || {
  bad "scp 拉取失败"; say '⛔ 本仓未改任何文件（rc=1）'; exit 1; }
[[ -f "$STAGE/$HTML_NAME" && -f "$STAGE/$DATA_NAME" ]] || {
  bad "暂存区缺文件"; say '⛔ 本仓未改任何文件（rc=1）'; exit 1; }

# ---- 3. 2a 逐字节相同（远端 vs 暂存）------------------------------------------
s_sha_html="$(sha256sum "$STAGE/$HTML_NAME" | awk '{print $1}')"
s_sha_data="$(sha256sum "$STAGE/$DATA_NAME" | awk '{print $1}')"
s_by_html="$(stat -c %s "$STAGE/$HTML_NAME")"
s_by_data="$(stat -c %s "$STAGE/$DATA_NAME")"
say "【2a · 逐字节相同】远端 sha ↔ 暂存区 sha"
[[ "$s_sha_html" == "$r_sha_html" && "$s_by_html" == "$r_by_html" ]] \
  && ok "$HTML_NAME  $s_by_html B  $s_sha_html" \
  || bad "$HTML_NAME 暂存 $s_by_html B/$s_sha_html ≠ 远端 $r_by_html B/$r_sha_html"
[[ "$s_sha_data" == "$r_sha_data" && "$s_by_data" == "$r_by_data" ]] \
  && ok "$DATA_NAME  $s_by_data B  $s_sha_data" \
  || bad "$DATA_NAME 暂存 $s_by_data B/$s_sha_data ≠ 远端 $r_by_data B/$r_sha_data"

# ---- 4. 2c 写范围闸门（在暂存区上跑；不过则本仓一字节没动）---------------------
say "【2c · §22 写范围闸门】check_fill_scope.py（base=基准件 / filled=暂存的副本 html）"
scope_out="$(cd "$REPO" && "$PY" "$M/check_fill_scope.py" \
  --base "$BASE_HTML" --filled "$STAGE/$HTML_NAME" --data "$STAGE/$DATA_NAME" 2>&1)"
scope_rc=$?
printf '%s\n' "$scope_out" | sed 's/^/    /'
case "$scope_rc" in
  0) printf '%s' "$scope_out" | grep -q 'SCOPE PASS' \
       && ok "SCOPE PASS（rc=0）" || bad "rc=0 但未见 SCOPE PASS 结论行（fail-closed）" ;;
  2) bad "SCOPE VIOLATION（rc=2）：副本 html 越出「第四章实验数据单元格文本」允许面" ;;
  *) bad "闸门用法级失败（rc=$scope_rc）" ;;
esac

((fail == 0)) || { say ''; say '⛔ 校验未全过 ⇒ 本仓 docs/ 未改一字节（rc=1）'; exit 1; }

# ---- 5. 落盘（幂等：sha 已一致则不写）-----------------------------------------
install_one() {              # $1 = 暂存文件  $2 = 目标  $3 = 远端 sha
  local src="$1" dst="$2" want="$3"
  if [[ -f "$dst" && "$(sha256sum "$dst" | awk '{print $1}')" == "$want" ]]; then
    chmod "$MODE_ROUND" "$dst" 2>/dev/null
    ok "已在位且一致，跳过写入：$(basename "$dst")"
    return 0
  fi
  cp -f -- "$src" "$dst" && chmod "$MODE_ROUND" "$dst" \
    && ok "已落盘：$(basename "$dst")（$MODE_ROUND）" \
    || { bad "落盘失败：$dst"; return 1; }
}
say "【落盘】→ 本仓 docs/（权限 $MODE_ROUND）"
install_one "$STAGE/$HTML_NAME" "$ROUND_HTML" "$r_sha_html"
install_one "$STAGE/$DATA_NAME" "$ROUND_DATA" "$r_sha_data"

# ---- 6. 落盘后复验：2a（本仓 vs 远端）+ 2b（三锁）------------------------------
say "【2a · 复验】本仓 sha ↔ 远端 sha"
for pair in "$ROUND_HTML:$r_sha_html:$HTML_NAME" "$ROUND_DATA:$r_sha_data:$DATA_NAME"; do
  IFS=: read -r dst want name <<<"$pair"
  got="$(sha256sum "$dst" | awk '{print $1}')"
  [[ "$got" == "$want" ]] && ok "$name  $(stat -c %s "$dst") B  $got" \
                          || bad "$name 本仓 $got ≠ 远端 $want"
done
check_baseline "落盘后"

say ''
if ((fail == 0)); then
  say "# 结论：SYNC PASS（rc=0）—— 本仓 docs/ 两份结果件已与 228 在役树逐字节一致，基准件三锁完好。"
  say "#   台账：$PATHS_REL_ROUND_HTML / $PATHS_REL_ROUND_DATA"
  exit 0
fi
say '# 结论：SYNC FAIL（rc=1）'
exit 1
