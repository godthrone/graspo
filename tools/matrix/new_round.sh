#!/usr/bin/env bash
# new_round.sh — **开新一轮**入口（非交互 · 幂等）：从基准件派生当日副本，⛔ 永不写基准件。
# ★ 规则（2026-10-23 起）：基准件只有一份（无日期 · 444 只读 · 结果永不写入）；
#   配额与填数只发生在**副本**上 ⇒ 每轮开跑前必须先跑本脚本建出副本。
# 动作（5 步，逐条打日志）：
#   ① 校验基准件：存在 / sha256 == 冻结值 / 权限 444（基准数据件同样查 444）；不符 ⇒ rc=1，零写入
#   ② 人读副本：不存在 ⇒ `cp -p` 基准件 + `chmod 644`；已存在 ⇒ **只校验不覆盖**
#   ③ 数据副本：不存在 ⇒ `cp -p` 基准数据件 + `chmod 644` + `meta.authority.human_view_path`
#      指向**该副本**；已存在 ⇒ 只校验该字段是否已指向副本（不符 ⇒ rc=1，⛔ 不自动覆盖）
#   ④ 打印下一步命令（`backfill_dated.sh` / `verify_dated.sh`）
#   ⑤ 幂等：同参重复跑 rc=0 且**零字节变化**（可用 `--dry-run` 先看计划）
# ⛔ 本脚本不写基准件、不做任何 git 操作、不跑训练、不占卡。
# 用法：bash tools/matrix/new_round.sh --date YYYYMMDD [--dry-run]
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$REPO/tools/matrix"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器（.venv/bin/python 与 python3 都不可用）\n' >&2; exit 1; }

usage() {
  cat <<'TXT'
用法: bash tools/matrix/new_round.sh --date YYYYMMDD [--dry-run]

  开新一轮（非交互 · 幂等）：从**基准件**派生当日副本，⛔ 永不写基准件。
    ① 校验基准件 sha256 与权限（444）
    ② 建副本人读件 docs/GRASPO训练框架功能测试报告-<date>.html（cp -p + chmod 644）
    ③ 建副本数据件 docs/capability-matrix-<date>.data.json（human_view_path 指向副本）
    ④ 打印下一步命令
    ⑤ 幂等：副本已存在则**只校验不覆盖**

  --dry-run   只做 ① 与计划打印，不落任何文件
  -h/--help   本帮助

★ 路径一律取自 tools/matrix/paths.py（可用 GRASPO_BASELINE_HTML / GRASPO_ROUND_DATE 覆盖）。
⛔ 不做任何 git 操作；⛔ 不改基准件内容与权限。
TXT
}

date_arg=""; dry_run=0
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --date) [[ $# -ge 2 ]] || { printf '⛔ --date 缺参数值\n' >&2; exit 2; }
            date_arg="$2"; shift 2 ;;
    --date=*) printf '⛔ 不解析 --date=<值> 形式，请用「--date <值>」\n' >&2; exit 2 ;;
    --dry-run) dry_run=1; shift ;;
    --*) printf '⛔ 未识别参数 %s（见 --help）\n' "$1" >&2; exit 2 ;;
    *) printf '⛔ 不接受位置参数：%s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$date_arg" ]] || { printf '⛔ --date 必填（YYYYMMDD）\n' >&2; exit 2; }
[[ "$date_arg" =~ ^[0-9]{8}$ ]] || { printf '⛔ --date 须为 YYYYMMDD，实际 %s\n' "$date_arg" >&2; exit 2; }

eval "$("$PY" "$M/paths.py" --sh --date "$date_arg")" || { printf '⛔ paths.py 取路径失败\n' >&2; exit 1; }

BASE_HTML="$PATHS_BASELINE_HTML"
BASE_DATA="$PATHS_BASELINE_DATA"
R_HTML="$PATHS_ROUND_HTML"
R_DATA="$PATHS_ROUND_DATA"
REL_R_HTML="$PATHS_REL_ROUND_HTML"

fail=0
say() { printf '%s\n' "$*"; }
bad() { fail=1; printf '  ✗ %s\n' "$*"; }
ok()  { printf '  ✓ %s\n' "$*"; }

say "# new_round.sh — 轮次 $PATHS_ROUND_DATE（仓根 $PATHS_REPO）$([[ $dry_run -eq 1 ]] && echo '【dry-run：零写入】')"

# ① 基准件校验 ---------------------------------------------------------------
say ""; say "① 基准件校验（sha256 + 权限 $PATHS_FROZEN_MODE）"
if [[ ! -f "$BASE_HTML" ]]; then
  bad "基准人读件不存在：$BASE_HTML"
else
  b_sha="$(sha256sum -- "$BASE_HTML" | awk '{print $1}')"
  b_perm="$(stat -c%a -- "$BASE_HTML")"
  b_size="$(stat -c%s -- "$BASE_HTML")"
  say "  $b_sha  ${b_size} B  $(basename "$BASE_HTML")"
  [[ "$b_sha" == "$PATHS_BASELINE_HTML_SHA256" ]] \
    && ok "sha256 == 冻结值（${PATHS_BASELINE_HTML_SHA256:0:12}…）" \
    || bad "sha256 与冻结值不符 ⇒ 基准件被改写（⛔ 本脚本不修，先查谁写的）"
  [[ "$b_perm" == "$PATHS_FROZEN_MODE" ]] && ok "人读基准件权限 $b_perm" \
    || bad "人读基准件权限 $b_perm，期望 $PATHS_FROZEN_MODE（bash $M/freeze_originals.sh 可复原）"
fi
if [[ ! -f "$BASE_DATA" ]]; then
  bad "基准数据件不存在：$BASE_DATA"
else
  d_perm="$(stat -c%a -- "$BASE_DATA")"
  [[ "$d_perm" == "$PATHS_FROZEN_MODE" ]] && ok "数据基准件权限 $d_perm" \
    || bad "数据基准件权限 $d_perm，期望 $PATHS_FROZEN_MODE（bash $M/freeze_originals.sh 可复原）"
fi
if ((fail)); then
  say ""; say "# 结论：① 未过 ⇒ **零写入**（rc=1）；先修基准件再开轮"
  exit 1
fi

# ② 人读副本 ----------------------------------------------------------------
say ""; say "② 人读副本 $PATHS_ROUND_HTML_NAME"
if [[ -e "$R_HTML" ]]; then
  ok "已存在 ⇒ 只校验不覆盖（权限 $(stat -c%a -- "$R_HTML")，$(stat -c%s -- "$R_HTML") B）"
else
  if ((dry_run)); then
    say "  [dry-run] 将执行：cp -p '$BASE_HTML' '$R_HTML' && chmod $PATHS_ROUND_MODE"
  else
    cp -p -- "$BASE_HTML" "$R_HTML" || { bad "复制失败：$R_HTML"; }
    chmod "$PATHS_ROUND_MODE" -- "$R_HTML" || bad "chmod $PATHS_ROUND_MODE 失败：$R_HTML"
    [[ -f "$R_HTML" && "$(stat -c%a -- "$R_HTML")" == "$PATHS_ROUND_MODE" ]] \
      && ok "已新建（chmod $PATHS_ROUND_MODE · $(stat -c%s -- "$R_HTML") B · 与基准件逐字节相同的起点）"
  fi
fi

# ③ 数据副本 ----------------------------------------------------------------
say ""; say "③ 数据副本 $PATHS_ROUND_DATA_NAME"
if [[ -e "$R_DATA" ]]; then
  cur_hv="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print(((d.get("meta") or {}).get("authority") or {}).get("human_view_path") or "")' "$R_DATA")"
  if [[ "$cur_hv" == "$REL_R_HTML" ]]; then
    ok "已存在 ⇒ 只校验不覆盖（human_view_path 已指向副本）"
  else
    bad "已存在但 human_view_path = ${cur_hv:-（缺失）}，期望 $REL_R_HTML"
    bad "⛔ 不自动覆盖（该副本可能已开跑/已回填）⇒ 人工裁定后重跑"
  fi
else
  if ((dry_run)); then
    say "  [dry-run] 将执行：cp -p '$BASE_DATA' '$R_DATA' && chmod $PATHS_ROUND_MODE"
    say "  [dry-run] 将执行：paths.py --set-human-view-path '$R_DATA' --to $REL_R_HTML"
  else
    cp -p -- "$BASE_DATA" "$R_DATA" || bad "复制失败：$R_DATA"
    if [[ -f "$R_DATA" ]]; then
      chmod "$PATHS_ROUND_MODE" -- "$R_DATA" || bad "chmod $PATHS_ROUND_MODE 失败：$R_DATA"
      if out="$("$PY" "$M/paths.py" --set-human-view-path "$R_DATA" --to "$REL_R_HTML" 2>&1)"; then
        say "  $out"
        ok "已新建（chmod $PATHS_ROUND_MODE · human_view_path → $REL_R_HTML）"
      else
        say "  $out"; bad "human_view_path 改写失败：$R_DATA"
      fi
    fi
  fi
fi

# ④ 下一步命令 --------------------------------------------------------------
say ""; say "④ 下一步命令"
say "  # 回填（副本不存在时本 wrapper 会自动跑 new_round.sh 派生）："
say "  bash $M/backfill_dated.sh --date $PATHS_ROUND_DATE --runs-dir <run-root>/collect"
say "  # 收尾自检（无副本时副本项自动跳过，rc 仍为 0）："
say "  bash $M/verify_dated.sh --date $PATHS_ROUND_DATE"

say ""
if ((fail)); then
  say "# 结论：FAIL（rc=1）$([[ $dry_run -eq 1 ]] && echo '【dry-run】')"
  exit 1
fi
say "# 结论：PASS（rc=0）$([[ $dry_run -eq 1 ]] && echo '【dry-run：本轮零写入】')；副本就绪，配额与填数只落在副本上"
