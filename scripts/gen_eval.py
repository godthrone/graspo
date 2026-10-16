import sys, os, subprocess
tier, gpu, port = sys.argv[1], sys.argv[2], sys.argv[3]

# ── §15.1（机密）：真实账号 / 宿主路径 / 内网地址一律不得写成源码字面量 ────────────
# 本文件要进开源仓。以下取值**全部在运行期产生**，默认值不含任何真实信息：
#   · 部署树 T / 评测根 EV：默认由**本文件自身的 checkout 位置**推导（`__file__`，
#     §1.4 单一真相源：树在哪，`scripts/` 就在哪，⛔ 不另抄一份路径字面量）；
#     需要另指时用 GRASPO_EVAL_TREE / GRASPO_EVAL_DIR 覆盖。
#   · 宿主模型根 / elam 数据根：默认 `$HOME` 推导，用 GRASPO_MODEL_HOST_ROOT /
#     GRASPO_ELAM_V5_DIR 覆盖。
#   · 远端执行目标：**必须**由 GRASPO_EVAL_SSH_USER / GRASPO_EVAL_SSH_HOST 提供——
#     这两项没有"合理默认值"可言，缺失即 **fail-closed**（§2.3 边界校验），
#     ⛔ 不回落成假地址把命令静默发到错误机器。
_DEFAULT_TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.environ.get("GRASPO_EVAL_TREE") or _DEFAULT_TREE
lower = tier.lower()
EV = os.environ.get("GRASPO_EVAL_DIR") or os.path.join(
    os.path.dirname(T.rstrip("/")), f"eval-{lower}-recon"
)
HOSTMODELS = os.environ.get("GRASPO_MODEL_HOST_ROOT") or os.path.join(
    os.path.expanduser("~"), "models"
)
HOSTMODELS = HOSTMODELS.rstrip("/") + "/"        # 供 `/models/` → 宿主根 的替换用（带尾斜杠）
HOSTMODELS_MOUNT = HOSTMODELS.rstrip("/")        # 供 `-v <根>:/models` 挂载用（去尾斜杠）
ELAM_HOST = (os.environ.get("GRASPO_ELAM_V5_DIR") or os.path.join(
    os.path.expanduser("~"), "elam-v5-balanced"
)).rstrip("/")
SSH_USER = os.environ.get("GRASPO_EVAL_SSH_USER")
SSH_HOST = os.environ.get("GRASPO_EVAL_SSH_HOST")
#: 部署机 SSH 端口。它不是 §15.1 的机密类别（端口不指认资产），保留默认值以免改变既有用法；
#: 需要时用 GRASPO_EVAL_SSH_PORT 覆盖。
SSH_PORT = os.environ.get("GRASPO_EVAL_SSH_PORT", "22")
assert SSH_USER and SSH_HOST, (
    "§15.1：内网账号 / 内网地址不得写进源码字面量；请用 GRASPO_EVAL_SSH_USER / "
    "GRASPO_EVAL_SSH_HOST 提供远端执行目标 ⇒ fail-closed"
)
SSH_TARGET = f"{SSH_USER}@{SSH_HOST}"

# ── 评测集口径（V1 = fit 口径 · 2026-10-16 主席裁定）─────────────────────────────
# 主席原话：「我们只要同一个数据集，先训练一遍，然后测试的时候还是这个数据集」。
# ★ 落地：**100 行档**（T001–T027；判据 = 矩阵里该档 `subset_size == 100`，
#   见 `docs/capability-matrix.data.json`）的**评测集 = 该档自身训练用的同一份文件**，
#   并落 `dataset_split: train` ⇒ 这是 **fit 口径（训练集自评）**，
#   ⛔ 读数的含义是「训练是否生效」，⛔ **不是**训练分布之外的表现（矩阵里已同步标注）。
# ★ 该文件与 `subsets-v2/T0NN.jsonl` **逐字节相同**（27 档同内容，md5 2049ff04…）；
#   `"image": "../images/…"` 按**数据文件父目录**解析 ⇒ /work/elam-v5/images
#   （228 上实测：该 100 行引用的 200 张图像 200/200 在场，missing=0）。
# ⛔ **20 行档不适用**（T031–T033/T037–T039/T043–T045/T046–T051）：n=20 ⇒ 每样本 5.00 pp，
#   而 GRASPO 通过线正好 +5.0 pp ⇒ 判据退化为「1 个样本决定成败」；主席对该批的裁定尚未回来
#   ⇒ **本包不动它们**（保持 702 口径），其待定状态登记在矩阵与工位 report.md。
# ⛔ **9 个 formal 档不动**（T028–T030/T034–T036/T040–T042）：训练源 = 全量 train 前 N，
#   措辞分布与 702 评测集本就一致（48–288 行，非 100 行档）。
# ★ 重叠分析（`train_dataset_path` / `train_limit`）**随档取真值**：原先写死
#   `/data/.../train.jsonl` + `train_limit: null` ⇒ 拿**全量 6378 行**去剔除重叠，
#   而 mini 档实际只训 100 行 ⇒ 剔掉的样本与该档训练过的图像毫无关系（T011 实证
#   `train_subset.sample_count=6378` / `overlap_excluded_count=169`，而实测
#   mini100 × 702 图像交集 = 0）。现改为**该档自己的训练集**：100 行档 = 同一份文件 + 100。
FIT_CALIBER_TIERS = tuple(f"T{i:03d}" for i in range(1, 28))
FIT_CALIBER_SUBSET = 100
FIT_CALIBER_JSONL = "/work/elam-v5/mini-dataset/mini-short-mm-train.jsonl"
TEST_JSONL = "/work/elam-v5/data/test.jsonl"
FULL_TRAIN_JSONL = "/work/elam-v5/data/train.jsonl"
_is_fit = tier in FIT_CALIBER_TIERS
if _is_fit:
    # ★ 评测集 = 训练文件本身；重叠分析按定义恒为 100/100 ⇒ 见矩阵口径说明（⛔ 不作次级读数）
    ds_path, ds_split, tr_path, tr_limit = FIT_CALIBER_JSONL, "train", FIT_CALIBER_JSONL, str(FIT_CALIBER_SUBSET)
else:
    # 20 行档 / 9 formal 档：保持 702 口径（⛔ 本包不动；train_limit 的既有缺陷登记在案、未修）
    ds_path, ds_split, tr_path, tr_limit = TEST_JSONL, "test", FULL_TRAIN_JSONL, "null"
print(f"CALIBER tier={tier} fit={_is_fit} dataset_path={ds_path} dataset_split={ds_split} "
      f"train_dataset_path={tr_path} train_limit={tr_limit}")
def run(cmd):
    """远端执行（路径都在 228 上）"""
    import shlex
    return subprocess.run(["ssh","-p",SSH_PORT,SSH_TARGET, cmd], capture_output=True, text=True).stdout.strip()
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
dataset_path: {ds_path}
output_dir: /evalout/.local/reports/after
base_model_path: $MP
checkpoint_path: $CP
merged_output_dir: $MD
gpus: "$G"
role: after
YEOF
  docker run --rm --name {lower}-export-$TS --network=none --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
    -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \\
    -v $T:/workspace/graspo:ro -v {HOSTMODELS_MOUNT}:/models:ro -v $T/runs:/runs:ro -v $EV:/evalout \\
    -v {ELAM_HOST}:/work/elam-v5:ro -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
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
  -v $T:/workspace/graspo:ro -v $EV:/evalout -v {ELAM_HOST}:/work/elam-v5:ro \\
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
    y = f"""dataset_path: {ds_path}
train_dataset_path: {tr_path}
train_limit: {tr_limit}
dataset_split: {ds_split}
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
