#!/usr/bin/env bash
# verify_dated.sh — 2026-10-22 基线轮「结果只进带日期副本」收尾自检（8 步，可复跑）。
# 只读：不写 docs/、不改数据件、不跑训练、不连服务器、不动 git 历史。
# 只写：.local/hb-workspace/20261022-baseline-r1/verify-dated-last.txt（本次读数台账）。
# 退出码：0 = 全部通过（"未回填"与"已回填"两态都应 rc=0）；1 = 有步骤失败。
# 用法：bash tools/matrix/verify_dated.sh
#   ★ GRASPO_DATED_DATA / GRASPO_DATED_HTML 可覆盖「副本对」指向（仅用于 /tmp 演练场演示"已回填"态；
#     缺省 = 仓内副本两件。⛔ 原版路径不可覆盖，① 恒查仓内原版）。
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"; M="$REPO/tools/matrix"
TEMPLATE="$M/template.html"
ORIG_HTML="$REPO/docs/GRASPO 训练框架功能测试报告.html"
DATED_HTML="${GRASPO_DATED_HTML:-$REPO/docs/GRASPO 训练框架功能测试报告-20261022.html}"
ORIG_DATA="$REPO/docs/capability-matrix.data.json"
DATED_DATA="${GRASPO_DATED_DATA:-$REPO/docs/capability-matrix-20261022.data.json}"
ORIG_SHA_EXPECT="31024648cfd4c0cd229365fcc1f2c1b92689db9414cba211b99c4e2d160470d9"
OUT="$REPO/.local/hb-workspace/20261022-baseline-r1/verify-dated-last.txt"
mkdir -p "$(dirname "$OUT")"; : >"$OUT"

fail=0
say() { printf '%s\n' "$*" | tee -a "$OUT"; }
bad() { fail=1; say "  ✗ FAIL: $*"; }
ok()  { say "  ✓ $*"; }

say "# verify_dated.sh 读数 — $(date '+%Y-%m-%d %H:%M:%S')（仓根 $REPO）"

# ① 原版 HTML 逐字节未变 -------------------------------------------------------
say ""; say "① 原版人读件 sha256（期望 ${ORIG_SHA_EXPECT:0:12}…）"
orig_sha="$(sha256sum "$ORIG_HTML" | awk '{print $1}')"; orig_size="$(stat -c%s "$ORIG_HTML")"
say "  $orig_sha  ${orig_size} B  $(basename "$ORIG_HTML")"
[[ "$orig_sha" == "$ORIG_SHA_EXPECT" ]] && ok "原版未变" || bad "原版 sha 与冻结值不符（原版被写过）"

# ② 原版 + 原版数据件 ⇒ verify PASS（④X-b 生效，最强那道机核） ------------------
say ""; say "② verify（原版对：原版数据件 + 原版人读件）"
if out="$("$PY" "$M/verify_matrix.py" --template "$TEMPLATE" --data "$ORIG_DATA" --html "$ORIG_HTML" 2>&1)"; then
  printf '%s\n' "$out" >>"$OUT"
  grep -E "^results:" <<<"$out" | sed 's/^/  /' | tee -a "$OUT"
  grep -q "VERIFY PASS" <<<"$out" && ok "rc=0 VERIFY PASS" || bad "rc=0 但输出无 'VERIFY PASS'"
  grep -q "已回填 0 档" <<<"$out" && ok "原版仍 0 档已回填（未回填态）" || say "  注意：原版 results 非 0 档（历史回填）"
else
  printf '%s\n' "$out" >>"$OUT"; bad "原版对 verify rc≠0（原版或原版数据件漂移）"
fi

# ③ 副本 + 副本数据件 ⇒ verify PASS + 打印已回填档号 ---------------------------
say ""; say "③ verify（副本对：副本数据件 + 副本人读件）"
filled="?" ; filled_list=""
if [[ -f "$DATED_DATA" && -f "$DATED_HTML" ]]; then
  if out="$("$PY" "$M/verify_matrix.py" --template "$TEMPLATE" --data "$DATED_DATA" --html "$DATED_HTML" 2>&1)"; then
    printf '%s\n' "$out" >>"$OUT"
    res="$(grep -E "^results:" <<<"$out")"; say "  $res"
    filled="$(sed -nE 's/.*已回填 ([0-9]+) 档.*/\1/p' <<<"$res")"; [[ -z "$filled" ]] && filled="0"
    filled_list="$(sed -nE 's/.*已回填 [0-9]+ 档 \[(.*)\].*/\1/p' <<<"$res")"
    ok "rc=0 VERIFY PASS"
  else
    printf '%s\n' "$out" >>"$OUT"; bad "副本对 verify rc≠0（副本数据件与副本 HTML 不同源）"
  fi
else
  bad "副本件缺失：$DATED_DATA / $DATED_HTML（T1 未执行）"
fi

# ④ 副本数据件 sha 记录 + 与原件的关系 ----------------------------------------
say ""; say "④ sha256 记录"
say "  原版数据件      $(sha256sum "$ORIG_DATA" | awk '{print $1}')"
if [[ -f "$DATED_DATA" ]]; then
  dsha="$(sha256sum "$DATED_DATA" | awk '{print $1}')"
  say "  副本数据件      $dsha"
  say "  副本人读件      $(sha256sum "$DATED_HTML" | awk '{print $1}')"
  n_diff="$(diff "$ORIG_DATA" "$DATED_DATA" | grep -cE '^[<>]')"
  [[ "$filled" == "0" ]] && say "  数据件相异行数  $n_diff（期望 2 = 仅 human_view_path 的一对 < >）" \
                        || say "  数据件相异行数  $n_diff（已回填：含 6 列 + 状态 + meta.backfill）"
  if [[ "$filled" == "0" ]]; then
    if cmp -s "$ORIG_HTML" "$DATED_HTML"; then ok "未回填状态：副本与原件逐字节相同（HTML）"
    else bad "未回填态但副本 HTML ≠ 原件"; fi
    [[ "$n_diff" -le 2 ]] && ok "未回填状态：副本数据件仅路径字段一处不同" || say "  注意：数据件相异行数 $n_diff > 2（可能已回填）"
  else
    ok "已回填状态：副本已回填 $filled 档 [$filled_list]（副本 HTML 与原件必然不同）"
  fi
else
  bad "副本数据件缺失，无法记录 sha"
fi

# ⑤ git status 只列预期文件 ---------------------------------------------------
say ""; say "⑤ git status --porcelain（docs/ 面必须只有预期件；非 docs/ 改动记为「他工作包」不计失败）"
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  warn_n=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    p="${line:3}"; p="${p#\"}"; p="${p%\"}"
    case "$p" in
      "docs/capability-matrix-20261022.data.json"|"docs/GRASPO 训练框架功能测试报告-20261022.html")
        ok "[预期·本工作包副本] $line" ;;
      "tools/matrix/backfill_dated.sh"|"tools/matrix/verify_dated.sh"|"tools/matrix/backfill_matrix.py")
        ok "[预期·本工作包新增脚本] $line" ;;
      "docs/capability-matrix.data.json")
        say "  [允许·工作包 A 的官方数据件改动] $line（本工作包从不写它）" ;;
      "docs/GRASPO 训练框架功能测试报告.html")
        bad "原版人读件出现在 git status（违反冻结）：$line" ;;
      docs/*)
        bad "docs/ 下意外改动（本工作包只许动副本两件）：$line" ;;
      *) warn_n=$((warn_n+1)); [[ $warn_n -le 3 ]] && say "  [非 docs/ 改动·他工作包] $line"
         [[ $warn_n -eq 3 ]] && say "  …（其余非 docs/ 改动略，计数见末行）" ;;
    esac
  done < <(git -C "$REPO" status --porcelain)
  ok "docs/ 面核对完成；非 docs/ 改动 $warn_n 项（他工作包在飞，不影响本项判定）"
else
  bad "非 git 仓，无法核对工作树"
fi

# ⑥ 泄漏扫描：显式点名两个副本件（默认目标不含副本） --------------------------
say ""; say "⑥ scan_matrix_leaks（显式点名副本两件）"
if [[ -f "$DATED_DATA" && -f "$DATED_HTML" ]]; then
  if out="$("$PY" "$M/scan_matrix_leaks.py" "$DATED_HTML" "$DATED_DATA" 2>&1)"; then
    printf '%s\n' "$out" >>"$OUT"; say "  $(tail -1 <<<"$out")"; ok "rc=0 无命中"
  else
    printf '%s\n' "$out" >>"$OUT"; bad "副本件泄漏扫描有命中"
  fi
else
  bad "副本件缺失，无法扫描"
fi

# ⑦ 无临时件残留 --------------------------------------------------------------
say ""; say "⑦ docs/.backfill-tmp-* 残留"
leftover="$(ls -d "$REPO"/docs/.backfill-tmp-* 2>/dev/null | tr '\n' ' ')"
[[ -z "$leftover" ]] && ok "无残留" || bad "存在残留：$leftover"

# ⑧ 原版两件必须是物理只读（chmod 444） ----------------------------------------
say ""; say "⑧ 原版两件权限（期望 444；加固脚本 tools/matrix/freeze_originals.sh）"
for f in "$ORIG_HTML" "$ORIG_DATA"; do
  perm="$(stat -c%a -- "$f" 2>/dev/null)"
  [[ "$perm" == "444" ]] && ok "$perm  $(basename "$f")" \
    || bad "权限为 ${perm:-?}，期望 444（原版须物理只读；bash $M/freeze_originals.sh 可复原）"
done
say ""; say "# 结论：$([[ $fail -eq 0 ]] && echo 'ALL PASS（rc=0）' || echo 'FAIL（rc=1）')；读数已存 $OUT"
exit "$fail"
