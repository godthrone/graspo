#!/usr/bin/env bash
# tests/e2e/rig_synth/verify_build.sh —— 端到端**可复跑**自证（纯本机，不需要 GPU / ms-swift / 真实数据）
#
# 作用：用一份结构化 fixture 把整条"生成器链"跑一遍，证明本包的工具**能复跑、且守卫真会失败**。
# 这是"生成器是否落盘、能否复跑"这条验收标准的**可执行证据**（不是口头声明）。
#
# 用法：bash tests/e2e/rig_synth/verify_build.sh [产物目录]
#   默认产物目录 = .local/hb-workspace/20260915-152747/task-synth-seq/selfcheck
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RIG="$REPO_ROOT/tests/e2e/rig_synth"
OUT="${1:-$REPO_ROOT/.local/hb-workspace/20260915-152747/task-synth-seq/selfcheck}"
BASE_CONFIG="$REPO_ROOT/samples/configs/matrix54/T013.yaml"
BASE_MANIFEST="$REPO_ROOT/tests/e2e/matrix54_manifest.json"

FAIL=0
step() { echo; echo "=== $* ==="; }
check() { # check <描述> <期望 rc> <实际 rc>
  if [ "$2" -eq "$3" ]; then echo "  OK   $1 (rc=$3)"; else echo "  FAIL $1 (期望 rc=$2，实际 rc=$3)"; FAIL=1; fi
}

rm -rf "$OUT"; mkdir -p "$OUT/evidence"
FIXTURE="$OUT/evidence/fixture_T013_subset.jsonl"

step "0) 造结构化 fixture（ELAM 形态：system + 多轮 tool + 末轮 user 双图）"
python3 - "$FIXTURE" <<'PY'
import json, sys
sysmsg = {"role": "system", "content": [{"type": "text", "text": "你是一个机器人。"}]}
def turn(step, extra=0):
    blocks = [{"type": "image", "image": f"../images/f{step}_left.jpg"},
              {"type": "image", "image": f"../images/f{step}_right.jpg"},
              {"type": "text", "text": f"第 {step} 步：捡起电池。"}]
    blocks += [{"type": "text", "text": f"补充{i}：" + "细节。" * 20} for i in range(extra)]
    return {"role": "user", "content": blocks}
rows = [
  {"messages": [sysmsg, turn(0),
     {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
       "function": {"name": "extend_arm", "arguments": {"action_type": "收缩手臂"}}}]},
     {"role": "tool", "content": "OK", "tool_call_id": "c1"}, turn(1)],
   "tools": [{"type": "function", "function": {"name": "extend_arm"}}],
   "targets": [{"id": "expected", "output": {"content": {"action_type": "伸长手臂"}}}],
   "id": "fx_short", "source": "fixture"},
  {"messages": [sysmsg, turn(0, 30),
     {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function",
       "function": {"name": "lift_arm", "arguments": {"action_type": "抬高手臂"}}}]},
     {"role": "tool", "content": "OK", "tool_call_id": "c2"}, turn(1, 30)],
   "tools": [{"type": "function", "function": {"name": "lift_arm"}}],
   "targets": [{"id": "expected", "output": {"content": {"action_type": "抬高手臂"}}}],
   "id": "fx_long", "source": "fixture"},
]
with open(sys.argv[1], "w", encoding="utf-8") as h:
    for r in rows:
        h.write(json.dumps(r, ensure_ascii=False) + "\n")
print("fixture lines:", len(rows))
PY

step "1) 桩标定（明确标注不是真实测量；仅用于本机跑通链路）"
cat > "$OUT/evidence/fixture_calibration.STUB.json" <<'JSON'
{"mode":"calibrate","measured":true,
 "measured_by":"STUB —— 本机自测用，**不是**真实测量值（本机无法量 token）",
 "stub":true,"base_index":0,"base_id":"fx_short","base_tokens_measured":1941,
 "points":[{"copies":1,"tokens":1941},{"copies":2,"tokens":3862},{"copies":4,"tokens":7705}]}
JSON
echo "  OK   桩标定已写（带 stub:true 标记）"

step "2) 生成两个变体（build_synth_seq.py）—— 期望 rc=0"
python3 "$RIG/build_synth_seq.py" --source "$FIXTURE" --base-index 0 \
  --calibration "$OUT/evidence/fixture_calibration.STUB.json" --variant v1-image-copy \
  --out-dir "$OUT/synth-host/T013/v1-image-copy" > "$OUT/evidence/build_v1.log" 2>&1
check "build v1-image-copy" 0 $?
python3 "$RIG/build_synth_seq.py" --source "$FIXTURE" --base-index 0 \
  --calibration "$OUT/evidence/fixture_calibration.STUB.json" --variant v2-text-only \
  --out-dir "$OUT/synth-host/T013/v2-text-only" > "$OUT/evidence/build_v2.log" 2>&1
check "build v2-text-only" 0 $?

step "3) 配置生成（emit_configs.py）—— 期望 rc=0，且只改白名单键"
python3 "$RIG/emit_configs.py" --base "$BASE_CONFIG" --out-dir "$OUT/rig_synth/configs" \
  --diff-out "$OUT/evidence/config_diff.txt" > "$OUT/evidence/emit_configs.log" 2>&1
check "emit_configs" 0 $?

step "4) 清单生成（emit_manifests.py）—— 期望 rc=0"
python3 "$RIG/emit_manifests.py" --src "$BASE_MANIFEST" --out-dir "$OUT/rig_synth/manifests" \
  --config-root "v1-image-copy=v1-image-copy,v2-text-only=v2-text-only" \
  > "$OUT/evidence/emit_manifests.log" 2>&1
check "emit_manifests" 0 $?

step "5) 产物守卫核验（verify_synth_artifacts.py）—— 期望 rc=0"
python3 "$RIG/verify_synth_artifacts.py" \
  --root "$OUT/synth-host/T013/v1-image-copy" --root "$OUT/synth-host/T013/v2-text-only" \
  > "$OUT/evidence/verify_ok.log" 2>&1
check "verify 正常路径" 0 $?

step "6) ★ 负向测试：篡改 v2 的 fills_context_column=true ⇒ 期望 rc=1（守卫真会失败）"
PROV="$OUT/synth-host/T013/v2-text-only/subsets/T013-synth-128k.provenance.json"
cp -f "$PROV" "$OUT/evidence/v2_prov.bak"
python3 - "$PROV" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["fills_context_column"] = True
p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
print("已篡改 fills_context_column -> True")
PY
python3 "$RIG/verify_synth_artifacts.py" --root "$OUT/synth-host/T013/v2-text-only" \
  > "$OUT/evidence/verify_negative.log" 2>&1
NEG_RC=$?
cp -f "$OUT/evidence/v2_prov.bak" "$PROV"
check "verify 负向（必须 rc=1）" 1 "$NEG_RC"
grep -q "G2" "$OUT/evidence/verify_negative.log" && echo "  OK   负向日志含 G2 断言失败" \
  || { echo "  FAIL 负向日志缺 G2"; FAIL=1; }

step "6b) ★ 裁定 1 负向：单跑档(128k)回填 A4=pass ⇒ 期望 rc=2（守卫必须拒绝）"
V1ROOT="$OUT/synth-host/T013/v1-image-copy"
python3 "$RIG/build_synth_seq.py" --out-dir "$V1ROOT" --tier-id T013 \
  --backfill-a4-tier 128k --backfill-a4-status pass \
  > "$OUT/evidence/a4_negative.log" 2>&1
A4NEG_RC=$?
check "单跑档 A4=pass 必须被拒" 2 "$A4NEG_RC"
grep -q "单跑档" "$OUT/evidence/a4_negative.log" && echo "  OK   负向日志含「单跑档」守卫断言" \
  || { echo "  FAIL 负向日志缺「单跑档」"; FAIL=1; }

step "6c) 裁定 1 正向：单跑档回填 A4=not_applicable_single_run ⇒ 期望 rc=0"
python3 "$RIG/build_synth_seq.py" --out-dir "$V1ROOT" --tier-id T013 \
  --backfill-a4-tier 128k --backfill-a4-status not_applicable_single_run \
  > "$OUT/evidence/a4_positive.log" 2>&1
check "单跑档 A4=未适用（单跑）" 0 $?
python3 - "$V1ROOT" <<'PY2'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "subsets" / "T013-synth-128k.provenance.json"
d = json.loads(p.read_text())
assert d["a4_dual_run"] is False, d["a4_dual_run"]
assert d["a4_status"] == "not_applicable_single_run", d["a4_status"]
assert d["a4_status_display"] == "未适用（单跑）", d["a4_status_display"]
assert "A4" not in d["applicable_criteria"], d["applicable_criteria"]
d4 = json.loads((pathlib.Path(sys.argv[1]) / "subsets" / "T013-synth-4k.provenance.json").read_text())
assert d4["a4_dual_run"] is True and d4["a4_status"] is None, (d4["a4_dual_run"], d4["a4_status"])
assert "A4" in d4["applicable_criteria"], d4["applicable_criteria"]
print("  OK   128k=单跑/未适用（单跑）/判据不含A4；4k=双跑/待跑/判据含A4")
PY2
[ $? -eq 0 ] || { echo "  FAIL A4 字段断言"; FAIL=1; }

step "6d) ★ 裁定 1 负向（产物层）：把单跑档 provenance 的 a4_status 篡改成 pass ⇒ 校验器期望 rc=1"
A4PROV="$V1ROOT/subsets/T013-synth-128k.provenance.json"
cp -f "$A4PROV" "$OUT/evidence/a4_prov.bak"
python3 - "$A4PROV" <<'PY3'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["a4_status"] = "pass"; d["a4_status_display"] = "通过"
p.write_text(json.dumps(d, ensure_ascii=False, indent=1)); print("已篡改 a4_status -> pass")
PY3
python3 "$RIG/verify_synth_artifacts.py" --root "$V1ROOT" \
  > "$OUT/evidence/a4_verify_negative.log" 2>&1
A4V_RC=$?
cp -f "$OUT/evidence/a4_prov.bak" "$A4PROV"
check "产物层 G6（单跑档 pass）必须 rc=1" 1 "$A4V_RC"
grep -q "G6" "$OUT/evidence/a4_verify_negative.log" && echo "  OK   负向日志含 G6 断言失败" \
  || { echo "  FAIL 负向日志缺 G6"; FAIL=1; }

step "7) 守卫单元测试（test_synth_guards.py + test_config_guards.py）—— 期望 rc=0"
python3 -m pytest "$RIG/test_synth_guards.py" "$RIG/test_config_guards.py" -q \
  > "$OUT/evidence/pytest_guards.log" 2>&1
check "pytest 守卫测试" 0 $?
tail -1 "$OUT/evidence/pytest_guards.log" | sed 's/^/  /'

step "7b) ★ 裁定 3：派生配置只偏离基线在**恰 3 个白名单键**上（含 save_steps）"
python3 - "$BASE_CONFIG" "$OUT/rig_synth/configs" <<'PY4'
import sys, pathlib, json
base_path, cfg_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
base = base_path.read_text(encoding="utf-8").splitlines()
marker = "# ==== 以下为基线逐字内容 ===="
allowed = ("max_prompt_length", "run_name", "save_steps")
bad, n, seen = 0, 0, []
for f in sorted(cfg_root.glob("*/*.yaml")):
    n += 1
    body = f.read_text(encoding="utf-8").split(marker, 1)[1].strip("\n").splitlines()
    for i, (a, b) in enumerate(zip(base, body)):
        if a != b:
            key = a.strip().split(":")[0]
            if key not in allowed:
                print(f"  违规差异 {f.name}:{i+1}: {a!r} -> {b!r}"); bad += 1
            seen.append(key)
    for k in ("save_steps", "max_prompt_length"):
        if not any(line.startswith(f"  {k}:") for line in body):
            print(f"  缺失键 {f.name}: {k}"); bad += 1
prov = json.loads((cfg_root / "baseline_provenance.json").read_text(encoding="utf-8"))
assert prov["allowed_patch_keys"] == ["max_prompt_length", "run_name", "save_steps"], prov
assert set(seen) <= set(allowed), seen
print(f"  OK   {n} 份配置；白名单外差异={bad}；实际偏离键={sorted(set(seen))}")
print(f"  OK   基线 sha256={prov['base_sha256']}")
raise SystemExit(1 if bad else 0)
PY4
check "配置白名单（3 键）" 0 $?

step "7c) ★ §15.1：tracked rig 文件里**不得出现宿主/内网绝对路径**（回归守卫）"
# 为什么有这一步：第一版 `run_synth.sh` **硬编码了 3 个宿主路径**，
# 被 `tools/gate_content.py` 以 `internal_absolute_path` **阻断 5 条**（rc=1）。
# 修法是**参数化环境变量 + fail-closed**（宪法 §15.1/§16：宿主路径不入 tracked 文件），
# **不是**加白名单。此处把该规则做成**可跑的自证**，防止回归。
HOSTPATH_HITS=0
for f in "$RIG"/build_synth_seq.py "$RIG"/measure_synth_tokens.py "$RIG"/emit_configs.py \
         "$RIG"/emit_manifests.py "$RIG"/verify_synth_artifacts.py \
         "$RIG"/test_synth_guards.py "$RIG"/test_config_guards.py \
         "$RIG"/verify_build.sh "$RIG"/run_synth.sh; do
  if grep -nE "/home/[a-z]|/mnt/[a-z]|10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\." "$f" >/dev/null 2>&1; then
    echo "  FAIL 内网路径混入：$f"
    grep -nE "/home/[a-z]|/mnt/[a-z]|10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\." "$f" | head -3
    HOSTPATH_HITS=1
  fi
done
[ "$HOSTPATH_HITS" -eq 0 ] && echo "  OK   9 个 rig 文件零宿主/内网绝对路径" || FAIL=1

step "8) shell 语法检查"
bash -n "$RIG/run_synth.sh" && { echo "  OK   run_synth.sh bash -n"; } || { echo "  FAIL bash -n"; FAIL=1; }

echo
if [ "$FAIL" -eq 0 ]; then
  echo "===== SELFCHECK OK：生成器链可复跑，守卫真会失败 ====="
  echo "产物与证据：$OUT"
else
  echo "===== SELFCHECK FAILED（见上方 FAIL 行）====="
fi
exit "$FAIL"
