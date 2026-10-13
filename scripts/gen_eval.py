import sys, os, subprocess
tier, gpu, port = sys.argv[1], sys.argv[2], sys.argv[3]
T="/data/remote/graspo-b11fix-20261012"
lower=tier.lower(); EV=f"/data/remote/eval-{lower}-recon"
HOSTMODELS="/home/user/models/"
def run(cmd):
    """远端执行（路径都在 228 上）"""
    import shlex
    return subprocess.run(["ssh","-p","22","user@192.0.2.10", cmd], capture_output=True, text=True).stdout.strip()
ck=run(f"ls -d {T}/runs/{tier}/{tier}/v*/checkpoint-* 2>/dev/null | tail -1")
if not ck:
    ck=run(f"ls -d {T}/runs/{tier}/{tier}/final 2>/dev/null | tail -1")   # native 侧终态目录
assert ck, "无 checkpoint"
form_out=run(f"PYTHONPATH={T}.hostdeps python3 {T}/scripts/run_guards.py form --checkpoint {ck}")
form=form_out.split("]")[0].strip("[")
mode=run(f"python3 -c \"import sys;sys.path.insert(0,'{T}/scripts');import launch_pipeline as lp;print(lp.decide_eval_mode('{form}')[0])\"")
mp=run(f"python3 -c \"import json;m=json.load(open('{T}/tests/e2e/matrix54_manifest_source.json'));t=[x for x in m['tiers'] if x['tier_id']=='{tier}'][0];print(t.get('model_path',''))\"")
hostmp=mp.replace("/models/", HOSTMODELS, 1) if mp.startswith("/models/") else mp
cp="/runs"+ck[len(T+"/runs"):]
merge=f"/evalout/merged-hf-{lower}"          # yaml 配置值（容器内）
mergehost=f"{EV}/merged-hf-{lower}"   # 宿主路径（§8 检查 / vLLM 挂载源）
md = merge if mode=="merge" else ck
mdhost = mergehost if mode=="merge" else ck
# ★ §10：27B/Mamba 档必须限制 max_num_seqs（9B 档不加）
maxseqs = "--max-num-seqs 192" if "27B" in str(mp).upper() or tier in ("T022","T024","T025","T026","T027","T040","T043") else ""
print(f"TIER={tier} form={form} mode={mode} MP={mp} HOSTMP={hostmp} CK={ck} CP={cp} MD={md}")
script=f"""#!/usr/bin/env bash
set -uo pipefail
# ★ 中途退出（INT/TERM）【不做清理】：拆掉 vLLM 会让失败原因不可见、也无法重试
trap 'say "★ 脚本中途退出 ⇒ 保留 vLLM 服务以便排查/重试"; exit 12' INT TERM
T={T}; EV={EV}; IMG=graspo:v0.28.11-cu130fix-v2-msgspec-20261012-run; BASEIMG=graspo:v0.28.11-cu130fix-v2-msgspec-20261012
G={gpu}; PORT={port}; TS=$(date -u +%s); MODE={mode}; TIER={tier}
CK={ck}; CP={cp}; MD={md}; MDHOST={mdhost}; MP={mp}; HOSTMP={hostmp}
MAXSEQS="{maxseqs}"   # ★ §10：27B(Mamba) 必须 ≤ 可用 cache blocks（实测 blocks=232 ⇒ 取 192）
EVLOG=$EV/evidence; LOG=$EV/eval.log
say(){{ printf '%s %s\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }}
say "TIER={tier} eval START GPU=$G port=$PORT form={form} mode=$MODE"
wait_ready(){{ for i in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q graspo-eval && return 0; sleep 15; done; return 1; }}
if [ "$MODE" = "merge" ]; then
  python3 $T/scripts/run_guards.py fresh --dir "$MDHOST" --phase pre | tee -a "$LOG" || {{ say "§8(pre) 失败"; echo stale > "$EV/EVAL-ABORTED"; exit 6; }}
  cat > $EV/export-{lower}.yaml <<YEOF
dataset_path: /work/elam-v5/data/test.jsonl
output_dir: /evalout/.local/reports/after
base_model_path: $MP
checkpoint_path: $CP
merged_output_dir: $MD
gpus: "$G"
role: after
YEOF
  docker run --rm --name {lower}-export-$TS --network=none --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
    -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \\
    -v $T:/workspace/graspo:ro -v /home/user/models:/models:ro -v $T/runs:/runs:ro -v $EV:/evalout \\
    -v /home/user/elam-v5-balanced:/work/elam-v5:ro -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
    graspo eval export --eval-config /evalout/export-{lower}.yaml > "$EVLOG/02-eval-export.txt" 2>&1
  say "export rc=$?"
  python3 $T/scripts/run_guards.py fresh --dir "$MDHOST" --phase post --since $TS | tee -a "$LOG" || {{ say "§8(post) 失败"; echo export-failed > "$EV/EVAL-ABORTED"; exit 6; }}
  SERVE=$MDHOST
else
  say "form={form} ⇒ 零 merge（vLLM 直指 checkpoint）"
  SERVE=$CK
fi
say "--- vLLM after ---"
nohup docker run --rm --name {lower}-vllm-after-$TS --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
  -v $SERVE:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g vllm-openai:v0.22.0 --model /model --served-model-name graspo-eval \\
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 $MAXSEQS --trust-remote-code \\
  --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$EVLOG/vllm-after.log" 2>&1 &
wait_ready || {{ say "★ vLLM after 未就绪"; docker stop {lower}-vllm-after-$TS >/dev/null 2>&1; echo vllm-after-not-ready > "$EV/EVAL-ABORTED"; exit 8; }}
say "vLLM after READY"; say "--- eval after ---"
bash "$EV/run-eval.sh" "$T" "$EV" {tier} after "$EV/eval-{lower}-after.yaml" > "$EVLOG/03-run-eval-after.out" 2>&1
say "eval after rc=$?"
# ★ 缺报告 ⇒ 不 stop（保留服务以便重试）；静默拆服务会让失败原因不可见
if [ -f "$EV/.local/reports/after/eval_report.json" ]; then
  docker stop {lower}-vllm-after-$TS >/dev/null 2>&1
else
  say "★ after 报告缺失 ⇒ **不 stop**（保留服务以便重试）"; exit 11
fi
say "--- vLLM base（宿主挂载源字面量 $HOSTMP）---"
nohup docker run --rm --name {lower}-vllm-base-$TS --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
  -v $HOSTMP:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g vllm-openai:v0.22.0 --model /model --served-model-name graspo-eval \\
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 $MAXSEQS --trust-remote-code \\
  --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$EVLOG/vllm-base.log" 2>&1 &
wait_ready || {{ say "★ vLLM base 未就绪"; docker stop {lower}-vllm-base-$TS >/dev/null 2>&1; echo vllm-base-not-ready > "$EV/EVAL-ABORTED"; exit 9; }}
say "vLLM base READY"; say "--- eval base + delta ---"
bash "$EV/run-eval.sh" "$T" "$EV" {tier} base "$EV/eval-{lower}-base.yaml" > "$EVLOG/04-run-eval-base.out" 2>&1
say "eval base rc=$?"
# ★ 缺报告 ⇒ 不 stop（保留服务以便重试）；静默拆服务会让失败原因不可见
if [ -f "$EV/.local/reports/base/eval_report.json" ]; then
  docker stop {lower}-vllm-base-$TS >/dev/null 2>&1
else
  say "★ base 报告缺失 ⇒ **不 stop**（保留服务以便重试）"; exit 11
fi
docker run --rm --network=none -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \\
  -v $T:/workspace/graspo:ro -v $EV:/evalout -v /home/user/elam-v5-balanced:/work/elam-v5:ro \\
  -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
  graspo eval delta --before /evalout/.local/reports/base/eval_report.json --after /evalout/.local/reports/after/eval_report.json \\
  --eval-config /evalout/eval-{lower}-after.yaml > "$EVLOG/05-eval-delta.txt" 2>&1
say "delta rc=$?"; say "=== {tier} EVAL DONE ==="; echo done > "$EV/EVAL-DONE"
"""
open(f"/tmp/run-{lower}-eval.sh","w").write(script)

# ★ 同时生成两侧评测 yaml（gen_eval 是第一出口：脚本 + 配置一起产）
import json as _json
_cp_after = (merge if mode=="merge" else cp)
for _role,_cpv in (("after", _cp_after), ("base", "null")):
    _cpx = "$CP" if False else _cpv
    y = f"""dataset_path: /work/elam-v5/data/test.jsonl
train_dataset_path: /work/elam-v5/data/train.jsonl
train_limit: null
dataset_split: test
base_model_path: {mp}
checkpoint_path: {_cpv}
role: {_role}
export_format: null
merged_output_dir: null
output_dir: /evalout/.local/reports/{_role}
endpoint: http://127.0.0.1:{port}
served_model_name: graspo-eval
gpus: "{gpu}"
max_workers: 8
seed: 42
graspo_threshold_pp: 20.0
"""
    open(f"/tmp/eval-{lower}-{_role}.yaml","w").write(y)
print("WROTE yamls:", f"/tmp/eval-{lower}-after.yaml", f"/tmp/eval-{lower}-base.yaml")

print("WROTE", f"/tmp/run-{lower}-eval.sh")
