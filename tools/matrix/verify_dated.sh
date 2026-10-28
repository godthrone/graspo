#!/usr/bin/env bash
# verify_dated.sh — 矩阵链收尾自检（8 步，可复跑）。★ 2026-10-23 改为「基准件唯一 + 按轮副本」新布局。
# 只读：不写 docs/、不改数据件、不跑训练、不连服务器、不动 git 历史。
# 只写：.local/hb-workspace/20261022-baseline-r1/verify-dated-last.txt（本次读数台账）。
# 退出码：0 = 全部通过；1 = 有步骤失败。
# ★ 「本轮还没开跑」是**正常态**：副本两件不存在 ⇒ ③④⑥ 自动跳过并记 PASS（rc 仍为 0）。
#   副本存在 ⇒ 照旧校验「与基准件相异行数 = N（结果列）」+ 权限断言（基准件 444）。
# 用法：bash tools/matrix/verify_dated.sh [--date YYYYMMDD]
#   ★ 路径一律取自 tools/matrix/paths.py：--date 或 GRASPO_ROUND_DATE 指定轮次，
#     GRASPO_BASELINE_HTML 可覆盖基准人读件路径（缺省 = 仓内基准件）；
#     GRASPO_DATED_DATA / GRASPO_DATED_HTML 可覆盖「副本对」指向（仅用于 /tmp 演练场演示"已回填"态）。
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$REPO/tools/matrix"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY:-}" ]] || { printf '⛔ 找不到 python 解释器\n' >&2; exit 1; }

date_arg=""
while (($#)); do
  case "$1" in
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --date) [[ $# -ge 2 ]] || { printf '⛔ --date 缺参数值\n' >&2; exit 2; }; date_arg="$2"; shift 2 ;;
    --date=*) printf '⛔ 不解析 --date=<值>，请用「--date <值>」\n' >&2; exit 2 ;;
    *) printf '⛔ 未识别参数 %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$date_arg" ]] && export GRASPO_ROUND_DATE="$date_arg"

# ★ 2026-10-28（审查警告 7）：轮次缺省值 = 今天，会把**实际存在的副本**（如 -20261025）判成"本轮无副本"
#   而静默跳过 ③④⑥ 三项（实测 17:54 那次就是这样漏掉的）。⇒ 未显式指定轮次时，**按仓内实际副本**定位：
#     唯一副本 ⇒ 用它；多份 ⇒ 取最新并打印全部候选（⛔ 不静默挑一个）；一份都没有 ⇒ 维持"今天 = 未开跑"。
ROUND_AUTO_NOTE=""
if [[ -z "${GRASPO_ROUND_DATE:-}" ]]; then
  mapfile -t _rounds < <(ls -1 "$REPO/docs/" 2>/dev/null \
    | sed -nE 's/^GRASPO.*报告-([0-9]{8})\.html$/\1/p' | sort -u)
  if (( ${#_rounds[@]} == 1 )); then
    export GRASPO_ROUND_DATE="${_rounds[0]}"
    ROUND_AUTO_NOTE="（自动识别轮次：仓内唯一副本 ${_rounds[0]}）"
  elif (( ${#_rounds[@]} > 1 )); then
    export GRASPO_ROUND_DATE="${_rounds[-1]}"
    ROUND_AUTO_NOTE="（自动识别轮次：仓内 ${#_rounds[@]} 份副本，取最新 ${_rounds[-1]}；候选 ${_rounds[*]}）"
  fi
fi

eval "$("$PY" "$M/paths.py" --sh)" || { printf '⛔ paths.py 取路径失败\n' >&2; exit 1; }

ORIG_HTML="$PATHS_BASELINE_HTML"
ORIG_DATA="$PATHS_BASELINE_DATA"
TEMPLATE="$PATHS_TEMPLATE"
DATED_HTML="${GRASPO_DATED_HTML:-$PATHS_ROUND_HTML}"
DATED_DATA="${GRASPO_DATED_DATA:-$PATHS_ROUND_DATA}"
# ★ 基准人读件的冻结值来自 paths.py（唯一真相源）；2026-10-23 第 2 章按主席原文重写后定稿。
#   ⛔ 不是放宽：① 仍逐字节比、⑧ 仍查 444；任何再次改写即红。
ORIG_SHA_EXPECT="$PATHS_BASELINE_HTML_SHA256"
OUT="$REPO/.local/hb-workspace/20261022-baseline-r1/verify-dated-last.txt"
mkdir -p "$(dirname "$OUT")"; : >"$OUT"

fail=0
say() { printf '%s\n' "$*" | tee -a "$OUT"; }
bad() { fail=1; say "  ✗ FAIL: $*"; }
ok()  { say "  ✓ $*"; }
# 副本状态：1 = 两件都在（本轮已开跑，照旧全查）；0 = 两件都不在（本轮未开跑，正常）；
#   2 = 只有数据件（html 是渲染派生物、可由 new_round.sh 补齐 ⇒ 视作"未开跑"，只提示不算失败）；
#  -1 = 只有 html（机读源丢了 = 真异常：渲染产物在、数据件不在 ⇒ fail-closed）
HAS_ROUND=0
if [[ -f "$DATED_DATA" && -f "$DATED_HTML" ]]; then
  HAS_ROUND=1
elif [[ -f "$DATED_DATA" && ! -e "$DATED_HTML" ]]; then
  HAS_ROUND=2
elif [[ -f "$DATED_HTML" && ! -e "$DATED_DATA" ]]; then
  HAS_ROUND=-1
fi

say "# verify_dated.sh 读数 — $(date '+%Y-%m-%d %H:%M:%S')（仓根 $REPO · 轮次 $PATHS_ROUND_DATE）${ROUND_AUTO_NOTE}"

# ① 报告冻结断言（sha + 字节 + 权限 444；唯一实现 = paths.py --verify-frozen）--------
#   ★ 2026-10-25 主席令「这个报告就定稿冻结了，没有我的允许，不能修改报告」（§22）：
#     三项**任一**不符即红 —— sha 抓"内容变了"，字节抓"换个等长内容"，权限抓"还能不能被就地改写"。
say ""; say "① 报告冻结断言（sha256 + 字节 + 权限 ${PATHS_FROZEN_MODE}；冻结日 ${PATHS_FROZEN_DATE} · §22）"
if out_frozen="$("$PY" "$M/paths.py" --verify-frozen 2>&1)"; then
  printf '%s\n' "$out_frozen" >>"$OUT"
  grep -E '^  (sha256|字节|权限)' <<<"$out_frozen" | sed 's/^/  /'
  ok "基准件冻结完好（sha256 / 字节 ${PATHS_BASELINE_HTML_BYTES} B / 权限 ${PATHS_FROZEN_MODE} 三项全符）"
else
  printf '%s\n' "$out_frozen" >>"$OUT"
  grep -E '^  (sha256|字节|权限|✗)' <<<"$out_frozen" | sed 's/^/  /'
  bad "报告冻结断言不符（sha/字节/权限任一）⇒ 基准件被改写过或权限被放开（⛔ 无主席许可不得修）"
fi
orig_sha="$(sha256sum "$ORIG_HTML" | awk '{print $1}')"; orig_size="$(stat -c%s "$ORIG_HTML")"
say "  $orig_sha  ${orig_size} B  $(basename "$ORIG_HTML")"

# ② 基准件 + 基准数据件 ⇒ verify PASS（④X-b 生效，最强那道机核） ------------------
say ""; say "② verify（基准对：基准数据件 + 基准人读件）"
if out="$("$PY" "$M/verify_matrix.py" --template "$TEMPLATE" --data "$ORIG_DATA" --html "$ORIG_HTML" 2>&1)"; then
  printf '%s\n' "$out" >>"$OUT"
  grep -E "^results:" <<<"$out" | sed 's/^/  /' | tee -a "$OUT"
  grep -q "VERIFY PASS" <<<"$out" && ok "rc=0 VERIFY PASS" || bad "rc=0 但输出无 'VERIFY PASS'"
  grep -q "已回填 0 档" <<<"$out" && ok "基准件仍 0 档已回填（未回填态）" || say "  注意：基准件 results 非 0 档（历史回填）"
else
  printf '%s\n' "$out" >>"$OUT"; bad "基准对 verify rc≠0（基准件或基准数据件漂移）"
fi

# ③ 副本 + 副本数据件 ⇒ verify PASS + 打印已回填档号 ---------------------------
say ""; say "③ verify（副本对：副本数据件 + 副本人读件，$PATHS_ROUND_HTML_NAME）"
filled="?" ; filled_list=""
if ((HAS_ROUND == -1)); then
  bad "副本只有人读件、机读数据件缺失：$DATED_DATA（渲染产物在、源丢了 ⇒ ⛔ 不猜，先查谁删的）"
elif ((HAS_ROUND == 2)); then
  say "  本轮无人读副本（只有数据件 $PATHS_ROUND_DATA_NAME）⇒ 视作未开跑，跳过副本项"
  say "  补齐：bash $M/new_round.sh --date $PATHS_ROUND_DATE（幂等：已有数据件只校验不覆盖）"
  ok "副本项跳过（rc 计入 0）"
elif ((HAS_ROUND == 0)); then
  say "  本轮无副本，跳过副本项（未开跑 = 正常态；开轮用 bash $M/new_round.sh --date $PATHS_ROUND_DATE）"
  ok "副本项跳过（rc 计入 0）"
elif out="$("$PY" "$M/verify_matrix.py" --template "$TEMPLATE" --data "$DATED_DATA" --html "$DATED_HTML" 2>&1)"; then
  printf '%s\n' "$out" >>"$OUT"
  res="$(grep -E "^results:" <<<"$out")"; say "  $res"
  filled="$(sed -nE 's/.*已回填 ([0-9]+) 档.*/\1/p' <<<"$res")"; [[ -z "$filled" ]] && filled="0"
  filled_list="$(sed -nE 's/.*已回填 [0-9]+ 档 \[(.*)\].*/\1/p' <<<"$res")"
  ok "rc=0 VERIFY PASS"
else
  printf '%s\n' "$out" >>"$OUT"; bad "副本对 verify rc≠0（副本数据件与副本 HTML 不同源）"
fi

# ④ 副本数据件 sha 记录 + 与基准件的关系 --------------------------------------
say ""; say "④ sha256 记录"
say "  基准数据件      $(sha256sum "$ORIG_DATA" | awk '{print $1}')"
if ((HAS_ROUND != 1)); then
  say "  （本轮无完整副本对，④ 跳过）"; ok "副本 sha 记录跳过（rc 计入 0）"
else
  dsha="$(sha256sum "$DATED_DATA" | awk '{print $1}')"
  say "  副本数据件      $dsha"
  say "  副本人读件      $(sha256sum "$DATED_HTML" | awk '{print $1}')"
  n_diff="$(diff "$ORIG_DATA" "$DATED_DATA" | grep -cE '^[<>]')"
  [[ "$filled" == "0" ]] && say "  数据件相异行数  $n_diff（期望 2 = 仅 human_view_path 的一对 < >）" \
                        || say "  数据件相异行数  $n_diff（已回填：含 6 列 + 状态 + meta.backfill）"
  if [[ "$filled" == "0" ]]; then
    if cmp -s "$ORIG_HTML" "$DATED_HTML"; then ok "未回填状态：副本与基准件逐字节相同（HTML）"
    else bad "未回填态但副本 HTML ≠ 基准件"; fi
    [[ "$n_diff" -le 2 ]] && ok "未回填状态：副本数据件仅路径字段一处不同" || say "  注意：数据件相异行数 $n_diff > 2（可能已回填）"
  else
    ok "已回填状态：副本已回填 $filled 档 [$filled_list]（副本 HTML 与基准件必然不同）"
  fi
fi

# ⑤ git status 只列预期文件 ---------------------------------------------------
say ""; say "⑤ git status --porcelain（docs/ 面只许出现下列「预期」行；非 docs/ 改动记为「他工作包」不计失败）"
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  warn_n=0
  # ★ 2026-10-23：主席改名（删带空格基准件 + 删带日期副本 + 新增无空格基准件）由主席自行提交，
  #   本工作包**不提交**那两条 ⇒ 此处按「预期·主席改名」放行（⛔ 不算成本包的意外改动）。
  LEGACY_HTML_REL="docs/GRASPO 训练框架功能测试报告.html"
  LEGACY_DATED_REL="docs/GRASPO 训练框架功能测试报告-20261022.html"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # porcelain 状态两列里可能带空格（" D" = 工作树删除、未入暂存）⇒ 去空格再比对
    st="${line:0:2}"; st="${st// /}"; p="${line:3}"; p="${p#\"}"; p="${p%\"}"
    case "$st|$p" in
      "D|$LEGACY_HTML_REL"|"D|$LEGACY_DATED_REL")
        say "  [预期·主席改名·删除] $line（⛔ 本工作包不提交）" ;;
      "??|$PATHS_REL_BASELINE_HTML"|"A|$PATHS_REL_BASELINE_HTML")
        say "  [预期·主席改名·新增基准件] $line（⛔ 本工作包不提交）" ;;
      "M|$PATHS_REL_BASELINE_HTML")
        bad "基准人读件被改（M）：$line —— 基准件结果永不写入" ;;
      "M|$PATHS_REL_BASELINE_DATA")
        say "  [允许·本工作包·基准数据件 human_view_path] $line" ;;
      *"|$PATHS_REL_ROUND_DATA"|*"|$PATHS_REL_ROUND_HTML"|\
*"|docs/capability-matrix-"[0-9]*".data.json"|*"|docs/GRASPO训练框架功能测试报告-"[0-9]*".html")
        say "  [允许·本工作包·按轮副本] $line" ;;
      "M|docs/run-discipline.md")
        say "  [预期·本工作包·文档指针] $line" ;;
      *"|docs/"*)
        bad "docs/ 下意外改动（本工作包只许动 paths.py/脚本/两份数据件的 human_view_path）：$line" ;;
      *) warn_n=$((warn_n+1)); [[ $warn_n -le 3 ]] && say "  [非 docs/ 改动·他工作包] $line"
         [[ $warn_n -eq 3 ]] && say "  …（其余非 docs/ 改动略，计数见末行）" ;;
    esac
  # ★ 2026-10-23：中文路径必须关掉 quotepath（否则 porcelain 输出 "docs/GRASPO \345..." 八进制转义，
  #   ⑤ 的 case 分支永远匹配不上 → 连"预期件"也会被判成意外改动）。
  done < <(git -C "$REPO" -c core.quotepath=false status --porcelain)
  ok "docs/ 面核对完成；非 docs/ 改动 $warn_n 项（他工作包在飞，不影响本项判定）"
else
  bad "非 git 仓，无法核对工作树"
fi

# ⑥ 泄漏扫描：显式点名副本两件（默认目标不含副本） --------------------------
say ""; say "⑥ scan_matrix_leaks（显式点名副本两件）"
if ((HAS_ROUND != 1)); then
  say "  （本轮无完整副本对，⑥ 跳过）"; ok "副本泄漏扫描跳过（rc 计入 0）"
elif out="$("$PY" "$M/scan_matrix_leaks.py" "$DATED_HTML" "$DATED_DATA" 2>&1)"; then
  printf '%s\n' "$out" >>"$OUT"; say "  $(tail -1 <<<"$out")"; ok "rc=0 无命中"
else
  printf '%s\n' "$out" >>"$OUT"; bad "副本件泄漏扫描有命中"
fi

# ⑦ 无临时件残留 --------------------------------------------------------------
say ""; say "⑦ docs/.backfill-tmp-* 残留"
leftover="$(ls -d "$REPO"/docs/.backfill-tmp-* 2>/dev/null | tr '\n' ' ')"
[[ -z "$leftover" ]] && ok "无残留" || bad "存在残留：$leftover"

# ⑧ 基准两件必须是物理只读（chmod 444） ----------------------------------------
say ""; say "⑧ 基准两件权限（期望 444；加固脚本 tools/matrix/freeze_originals.sh）"
for f in "$ORIG_HTML" "$ORIG_DATA"; do
  perm="$(stat -c%a -- "$f" 2>/dev/null)"
  [[ "$perm" == "$PATHS_FROZEN_MODE" ]] && ok "$perm  $(basename "$f")" \
    || bad "权限为 ${perm:-?}，期望 $PATHS_FROZEN_MODE（基准件须物理只读；bash $M/freeze_originals.sh 可复原）"
done
say ""; say "# 结论：$([[ $fail -eq 0 ]] && echo 'ALL PASS（rc=0）' || echo 'FAIL（rc=1）')；读数已存 $OUT"
exit "$fail"
