#!/usr/bin/env python3
"""评测脚本 + 两侧评测 yaml 生成器 —— **按档位形态现算**（68 档 M001–M068）。

2026-10-22 改造（工作包 B / T3）——去掉两处 **T 档号硬编码**，改为"从清单 + 形态现算"：
  ① 原 `FIT_CALIBER_TIERS`（`T001–T027` + 18 个 `T03x/T04x/T05x` 字面量清单）删掉，
     改为**按清单该行 `data.subset_size` 现算**：`subset_size == 100` ⇒ fit 口径
     （评测集 = 该档自身训练用的**同一份**文件、`dataset_split: train`）；否则平台方 702 行口径。
     ⇒ M 档号可直接入参（⛔ 不再有"T/M 双轨"分支：本件**只认** `matrix68_manifest.json`）。
  ② 原 `tier in ("T022","T024","T025","T026","T027","T040","T043")` 硬编码名单删掉，
     改为**按档位形态现算** `--max-num-seqs`：`model == 27B`（Mamba 混合架构）⇒ 需要限流；
     取值 = `min(launch_pipeline.MAMBA_SAFE_MAX_NUM_SEQS, --blocks)`（§10 的 blocks 上限口径），
     并在生成前后各串一道护栏：
        · 起跑前 `scripts/run_guards.py mamba --max-num-seqs N --blocks B`（§10 fail-closed 探针）；
        · 产物级 `launch_pipeline.check_max_num_seqs_in_script()`（纪律必须落在**被执行的那份脚本**上）。
     9B 档不加该参数（旧行为不变）。

保留的既有语义（⛔ 改动即回归）：
  · §8 新鲜度（`run_guards fresh pre/post`）与"缺报告不 stop"分支、INT/TERM 不拆服务；
  · 判据② 逐算法（`judge_method` 取清单该行 `algorithm`，⛔ 不按 role 推断）；
  · §15.1 机密：账号/内网地址/宿主路径一律不写成源码字面量。

新增的可配置项（链路侦察 B5：原先写死 `/tmp` + 不可关的 ssh 断言）：
  · `--tmp-dir`（缺省 `$GRASPO_EVAL_TMP_DIR` → `/tmp/graspo-eval`）：脚本与 yaml 的落点；
  · `--no-ssh`：关闭远端探测（form / checkpoint / 清单读取），改由 `--checkpoint` / `--form`
    显式给出或按形态推定 —— 用于离线自检与干跑（⛔ 不用于真实评测）。

用法：
    python3 scripts/gen_eval.py --tier M001 --gpu 4 --port 18889
    python3 scripts/gen_eval.py --tier M041 --gpu 4,5 --port 18889 --blocks 232
    python3 scripts/gen_eval.py --tier M041 --gpu 4 --port 18889 --dry-run --no-ssh --tmp-dir /tmp/x

    # ★ OPD 教师侧（判据② 的 `eval.teacher_after_acc` 唯一来源；见 docs/run-discipline.md §20）
    python3 scripts/gen_eval.py --tier M068 --role teacher --gpu 4,5 --port 18889 --blocks 232
    #   ⇒ 生成 run-m068-teacher-eval.sh + eval-m068-teacher.yaml（`role: other`）
    #   ⇒ 报告落 <EVAL_DIR>/.local/reports/teacher/eval_report.json
    #   ★ 口径与学生侧**同一份**（compute_caliber 同一函数 + 温度 0 + 同一判分函数），⛔ 不另立参数；
    #   ★ 教师档按文档第 3 章口径（SFT / 27B / LoRA / dp=1）从清单现算，⛔ 不写死档号；
    #   ★ 顺序约束：教师档必须先于 OPD 档训练完成（教师产物是 OPD 档判据② 的前置证据）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "e2e" / "matrix68_manifest.json"

#: 唯一真相源：§10 的安全上限（远端机实测 27B(Mamba) 可用 cache blocks = 232 ⇒ 取 192）。
#: ⛔ 本文件不另写数值；需要时直接引用 `scripts/launch_pipeline.py` 的常量。
def _mamba_safe_default() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import launch_pipeline as lp                                    # noqa: PLC0415

    return int(lp.MAMBA_SAFE_MAX_NUM_SEQS)


#: 数据集口径里"平台方自有评测集"（正式档）的**容器内路径**。
#: ★ §15.1：内部数据集路径不写成源码字面量 ⇒ 缺省为空，**必须由现场用环境变量给出**。
#: 为什么这里是"必填"而非给占位缺省：它是**输入**——一个字符的偏差就是拿错数据集评；
#: 占位值只会把错误推迟到起跑时才以 `DatasetError` 暴露，缺省直接 fail-closed 更便宜。
TEST_JSONL = os.environ.get("GRASPO_EVAL_TEST_JSONL", "")
FULL_TRAIN_JSONL = os.environ.get("GRASPO_EVAL_TRAIN_JSONL", "")

#: 算法（清单 `algorithm`）⇒ 判据② 冻结口径键。只是名字映射，⛔ 不含阈值数值。
ALGO_METHOD = {"SFT": "sft", "GRASPO": "graspo", "CPT": "cpt", "OPD": "opd"}

#: ── OPD 教师档的形态键（★ 文档口径，⛔ 本文件不写死档号）────────────────────────────
#: 唯一真相源：定稿报告**第 3 章「模型说明 · OPD 教师」** = 「Qwen3.8-27B 在 **SFT / LoRA /
#: dp=1** 这一档训练完成后的产物（不是基座权重本身）」；数据件
#: `meta.criteria_caliber.threshold.by_method.opd.epoch_note` 是同口径的第二处逐字
#: （2026-10-22 判据变更）。
#: ⚠ 「dp=1」散文**单独不唯一**（M041/M044/M046/M047/M049/M050 的 dp 都是 1）⇒ 判别键取**单卡档**
#: 的完整并行形态 TP=PP=DP=1；本矩阵 68 档里唯一命中 M041。命中数 ≠ 1 即 fail-closed（⛔ 不猜）。
TEACHER_FORM = {"algorithm": "SFT", "model": "27B", "tuner_type": "lora",
                "tp": 1, "pp": 1, "dp": 1}

IMG = os.environ.get("GRASPO_EVAL_IMG", "graspo:v0.28.11-cu130fix-v2-msgspec-20261012-run")
BASEIMG = os.environ.get("GRASPO_EVAL_BASEIMG", "graspo:v0.28.11-cu130fix-v2-msgspec-20261012")
VLLM_IMG = os.environ.get("GRASPO_EVAL_VLLM_IMG", "vllm-openai:v0.22.0")


def die(msg: str, code: int = 2) -> "NoReturn":                     # noqa: F821
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        die(f"清单不存在：{path}（先跑 tests/e2e/generate_matrix68_manifest_source.py）")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("tiers"):
        die(f"清单没有 tiers：{path}")
    return doc


def tier_of(manifest: dict, tier: str) -> dict:
    hit = [t for t in manifest["tiers"] if t["tier_id"] == tier]
    if not hit:
        die(f"清单里没有档号 {tier!r}（本件只认 {MANIFEST.name} 的 M 档号）")
    return hit[0]


def find_teacher_tier(manifest: dict) -> dict:
    """按 `TEACHER_FORM`（= 文档第 3 章 OPD 教师口径）从清单现算教师档，⛔ 不写死档号。

    命中数必须**恰好 1**：0 ⇒ 清单里没有这一档（教师还不存在）；>1 ⇒ 口径有歧义（先修清单/文档，
    ⛔ 不取第一个凑数）。
    """
    hits = []
    for t in manifest["tiers"]:
        pc = t.get("parallel_components") or {}
        if (str(t.get("algorithm")).upper() != TEACHER_FORM["algorithm"]
                or str(t.get("model")) != TEACHER_FORM["model"]
                or str(t.get("tuner_type")) != TEACHER_FORM["tuner_type"]):
            continue
        try:
            if all(int(pc.get(k)) == TEACHER_FORM[k] for k in ("tp", "pp", "dp")):
                hits.append(t)
        except (TypeError, ValueError):
            continue
    if len(hits) != 1:
        die(f"OPD 教师档按文档口径 {TEACHER_FORM} 命中 {len(hits)} 个"
            f"{[h.get('tier_id') for h in hits]} ≠ 1 ⇒ 不生成（⛔ 不猜，先对齐清单与文档）", 1)
    return hits[0]


# ── ① 评测集口径：按清单该行的 subset_size 现算（替代 T 档号清单）──────────────────
def compute_caliber(mt: dict) -> dict:
    subset_size = int(mt["data"]["subset_size"])
    train_container = mt["data"]["train_path"]
    if subset_size == 100:
        return {
            "fit": True,
            "why": f"subset_size={subset_size} ⇒ fit 口径（训练集自评）",
            "dataset_path": train_container,
            "dataset_split": "train",
            "train_dataset_path": train_container,
            "train_limit": str(subset_size),
        }
    if not (TEST_JSONL and FULL_TRAIN_JSONL):
        die("subset_size ≠ 100 的档走「平台方自有评测集」口径，而该数据集的容器内路径"
            "不是源码字面量（§15.1）⇒ 必须显式给出："
            "GRASPO_EVAL_TEST_JSONL / GRASPO_EVAL_TRAIN_JSONL", 2)
    return {
        "fit": False,
        "why": f"subset_size={subset_size} ≠ 100 ⇒ 平台方 702 行评测集口径",
        "dataset_path": TEST_JSONL,
        "dataset_split": "test",
        "train_dataset_path": FULL_TRAIN_JSONL,
        "train_limit": "null",
    }


# ── ② §10：max-num-seqs 按形态现算（替代 T 档号名单）──────────────────────────────
def compute_max_num_seqs(mt: dict, blocks: int | None) -> dict:
    """返回 {needed, value, basis, text}。`needed` 恒由**形态**决定，⛔ 不看档号。"""
    safe = _mamba_safe_default()
    needed = "27B" in str(mt.get("model") or "") or "mamba" in json.dumps(mt).lower()
    if not needed:
        return {"needed": False, "value": None,
                "basis": f"model={mt.get('model')!r} 非 27B(Mamba) ⇒ 不限制 max-num-seqs（旧行为不变）",
                "text": ""}
    if blocks is None:
        return {"needed": True, "value": safe,
                "basis": f"27B(Mamba) ⇒ 取 §10 缺省安全值 {safe}"
                         f"（远端机实测可用 cache blocks=232；未给 --blocks ⇒ 不做 ≤blocks 断言，真跑前必须给）",
                "text": f"--max-num-seqs {safe}"}
    value = min(safe, int(blocks))
    return {"needed": True, "value": value,
            "basis": f"27B(Mamba) ⇒ min(§10 安全值 {safe}, 实测 blocks {blocks}) = {value} ≤ {blocks}",
            "text": f"--max-num-seqs {value}"}


def assert_mamba(needed: bool, value: int | None, blocks: int | None) -> None:
    """§10 起跑前探针（fail-closed）：串 `scripts/run_guards.py mamba`（⛔ 不用本文件的判断代替护栏）。"""
    if not needed or blocks is None:
        return
    cmd = [sys.executable, str(ROOT / "scripts" / "run_guards.py"), "mamba",
           "--max-num-seqs", str(value), "--blocks", str(blocks)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[§10] run_guards mamba → rc={p.returncode} {(p.stdout or p.stderr).strip()}")
    if p.returncode != 0:
        die(f"§10 护栏未过（max_num_seqs={value} vs blocks={blocks}）⇒ 拒绝生成", 1)


# ── 远端探测（可关）───────────────────────────────────────────────────────────
class Remote:
    """部署机探测（`--no-ssh` 时全部走本机/显式入参）。"""

    def __init__(self, enabled: bool, port: str, user: str | None, host: str | None):
        self.enabled = enabled
        self.port, self.user, self.host = port, user, host
        if self.enabled and not (user and host):
            die("§15.1：远端执行目标必须由 GRASPO_EVAL_SSH_USER / GRASPO_EVAL_SSH_HOST 提供"
                "（缺失即 fail-closed，⛔ 不回落成假地址）")

    def run(self, cmd: str) -> str:
        if not self.enabled:
            return ""
        p = subprocess.run(["ssh", "-p", self.port, f"{self.user}@{self.host}", cmd],
                           capture_output=True, text=True)
        return (p.stdout or "").strip()


# ── 渲染：评测脚本（保留既有回归约束的文本形态）──────────────────────────────────
def render_script(*, tier: str, gpu: str, port: str, tree: str, ev: str, lower: str,
                  mode: str, form: str, ck: str, cp: str, md: str, mdhost: str,
                  mp: str, hostmp: str, maxseqs: str, cal: dict, judge_method: str,
                  hostmodels_mount: str, elam_host: str, serving_basis: str,
                  needs_limit: bool) -> str:
    return f"""#!/usr/bin/env bash
set -uo pipefail
# ★ 中途退出（INT/TERM）【不做清理】：拆掉 vLLM 会让失败原因不可见、也无法重试
trap 'say "★ 脚本中途退出 ⇒ 保留 vLLM 服务以便排查/重试"; exit 12' INT TERM
T={tree}; EV={ev}; IMG={IMG}; BASEIMG={BASEIMG}
G={gpu}; PORT={port}; TS=$(date -u +%s); MODE={mode}; TIER={tier}
CK={ck}; CP={cp}; MD={md}; MDHOST={mdhost}; MP={mp}; HOSTMP={hostmp}
MAXSEQS="{maxseqs}"   # ★ §10 现算：{serving_basis}
SERVING_NOTE="§10 caliber: {serving_basis}（needs_limit={str(needs_limit).lower()}）"
EVLOG=$EV/evidence; LOG=$EV/eval.log
say(){{ printf '%s %s\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }}
say "TIER={tier} eval START GPU=$G port=$PORT form={form} mode=$MODE"
say "§10 $SERVING_NOTE"
wait_ready(){{ for i in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q graspo-eval && return 0; sleep 15; done; return 1; }}
if [ "$MODE" = "merge" ]; then
  python3 $T/scripts/run_guards.py fresh --dir "$MDHOST" --phase pre | tee -a "$LOG" || {{ say "§8(pre) 失败"; echo stale > "$EV/EVAL-ABORTED"; exit 6; }}
  cat > $EV/export-{lower}.yaml <<YEOF
dataset_path: {cal["dataset_path"]}
output_dir: /evalout/.local/reports/after
base_model_path: $MP
checkpoint_path: $CP
merged_output_dir: $MD
gpus: "$G"
role: after
YEOF
  docker run --rm --name {lower}-export-$TS --network=none --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
    -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \\
    -v $T:/workspace/graspo:ro -v {hostmodels_mount}:/models:ro -v $T/runs:/runs:ro -v $EV:/evalout \\
    -v {elam_host}:/work/elam-v5:ro -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
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
  -v $SERVE:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g {VLLM_IMG} --model /model --served-model-name graspo-eval \\
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
  -v $HOSTMP:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g {VLLM_IMG} --model /model --served-model-name graspo-eval \\
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
  -v $T:/workspace/graspo:ro -v $EV:/evalout -v {elam_host}:/work/elam-v5:ro \\
  -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
  graspo eval delta --before /evalout/.local/reports/base/eval_report.json --after /evalout/.local/reports/after/eval_report.json \\
  --eval-config /evalout/eval-{lower}-after.yaml > "$EVLOG/05-eval-delta.txt" 2>&1
say "delta rc=$?"; say "=== {tier} EVAL DONE ==="; echo done > "$EV/EVAL-DONE"
"""


def render_yaml(*, role: str, cal: dict, mp: str, ckpt: str, port: str, gpu: str,
                judge_method: str, out_dir_rel: str | None = None) -> str:
    """`out_dir_rel` = 评测产物在容器 `/evalout` 下的相对目录（缺省按 role 命名）。

    ★ 教师侧的落点**只能来自** `scripts/record_checks.py::TEACHER_REPORT_REL`（本文件从那里 import），
      ⛔ 不在这里另写一份路径字面量 —— 产出端与消费端必须同一份口径。
    """
    return f"""dataset_path: {cal["dataset_path"]}
train_dataset_path: {cal["train_dataset_path"]}
train_limit: {cal["train_limit"]}
dataset_split: {cal["dataset_split"]}
base_model_path: {mp}
checkpoint_path: {ckpt}
role: {role}
export_format: null
merged_output_dir: null
output_dir: /evalout/{out_dir_rel or f'.local/reports/{role}'}
endpoint: http://127.0.0.1:{port}
served_model_name: graspo-eval
gpus: "{gpu}"
max_workers: 8
seed: 42
judge_method: {judge_method}
"""


def render_teacher_script(*, opd_tier: str, teacher_tier: str, gpu: str, port: str, tree: str,
                          ev: str, lower: str, mode: str, form: str, ck: str, cp: str,
                          md: str, mdhost: str, mp: str, hostmp: str, maxseqs: str, cal: dict,
                          hostmodels_mount: str, elam_host: str, serving_basis: str,
                          needs_limit: bool, report_rel: str) -> str:
    """OPD **教师侧**评测脚本（判据② 的教师读数唯一产出件）。

    ★ 与两侧（after/base）评测**同一份口径**：`cal` 来自同一个 `compute_caliber(mt)`、
      同一份 100 条、同一个 vLLM 贪心温度 0（`EvalDecoding` 强制）、同一份逐样本判分函数
      （`src/graspo/eval/criteria.py`，与 rank_method 无关）⇒ ⛔ 本模板不含任何口径参数。
    ★ 只跑一侧：教师没有 base/after/delta 三件套（判据② 的 Δ 由回填器用学生报告现算）。
    """
    return f"""#!/usr/bin/env bash
set -uo pipefail
# ★ 中途退出（INT/TERM）【不做清理】：拆掉 vLLM 会让失败原因不可见、也无法重试
trap 'say "★ 脚本中途退出 ⇒ 保留 vLLM 服务以便排查/重试"; exit 12' INT TERM
T={tree}; EV={ev}; IMG={IMG}; BASEIMG={BASEIMG}
G={gpu}; PORT={port}; TS=$(date -u +%s); MODE={mode}; TIER={opd_tier}; TEACHER_TIER={teacher_tier}
CK={ck}; CP={cp}; MD={md}; MDHOST={mdhost}; MP={mp}; HOSTMP={hostmp}
MAXSEQS="{maxseqs}"   # ★ §10 现算（按**教师档**形态）：{serving_basis}
TREPORT="$EV/{report_rel}"
EVLOG=$EV/evidence; LOG=$EV/eval.log
say(){{ printf '%s %s\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }}
say "TEACHER eval START opd_tier={opd_tier} teacher_tier={teacher_tier} GPU=$G port=$PORT form={form} mode=$MODE"
say "§10 teacher caliber: {serving_basis}（needs_limit={str(needs_limit).lower()}）"
say "教师报告落点 = $TREPORT（判据② 的 eval.teacher_after_acc 只认这份）"
wait_ready(){{ for i in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q graspo-eval && return 0; sleep 15; done; return 1; }}
if [ "$MODE" = "merge" ]; then
  python3 $T/scripts/run_guards.py fresh --dir "$MDHOST" --phase pre | tee -a "$LOG" || {{ say "§8(pre) 失败 ⇒ fail-closed"; echo stale > "$EV/EVAL-ABORTED"; exit 6; }}
  cat > $EV/export-teacher.yaml <<YEOF
dataset_path: {cal["dataset_path"]}
output_dir: /evalout/{report_rel.rsplit('/', 1)[0]}
base_model_path: $MP
checkpoint_path: $CP
merged_output_dir: $MD
gpus: "$G"
role: other
YEOF
  docker run --rm --name teacher-export-$TS --network=none --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
    -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \\
    -v $T:/workspace/graspo:ro -v {hostmodels_mount}:/models:ro -v $T/runs:/runs:ro -v $EV:/evalout \\
    -v {elam_host}:/work/elam-v5:ro -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
    graspo eval export --eval-config /evalout/export-teacher.yaml > "$EVLOG/06-teacher-export.txt" 2>&1
  say "teacher export rc=$?"
  python3 $T/scripts/run_guards.py fresh --dir "$MDHOST" --phase post --since $TS | tee -a "$LOG" || {{ say "§8(post) 失败"; echo export-failed > "$EV/EVAL-ABORTED"; exit 6; }}
  SERVE=$MDHOST
else
  say "form={form} ⇒ 零 merge（vLLM 直指教师 checkpoint）"
  SERVE=$CK
fi
say "--- vLLM teacher ---"
nohup docker run --rm --name teacher-vllm-$TS --runtime=nvidia --gpus "\\"device=$G\\"" -e NVIDIA_VISIBLE_DEVICES=$G \\
  -v $SERVE:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g {VLLM_IMG} --model /model --served-model-name graspo-eval \\
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 $MAXSEQS --trust-remote-code \\
  --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$EVLOG/vllm-teacher.log" 2>&1 &
wait_ready || {{ say "★ vLLM teacher 未就绪"; docker stop teacher-vllm-$TS >/dev/null 2>&1; echo vllm-teacher-not-ready > "$EV/EVAL-ABORTED"; exit 8; }}
say "vLLM teacher READY"; say "--- eval teacher（与两侧同一份 100 条 / 温度 0 / 同一判分函数）---"
docker run --rm --network=host -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \\
  -v $T:/workspace/graspo:ro -v $EV:/evalout -v {elam_host}:/work/elam-v5:ro \\
  -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \\
  graspo eval run --eval-config /evalout/eval-{lower}-teacher.yaml > "$EVLOG/07-eval-teacher.txt" 2>&1
say "eval teacher rc=$?"
# ★ 缺报告 ⇒ 不 stop（保留服务以便重试）；静默拆服务会让失败原因不可见
if [ -f "$TREPORT" ]; then
  docker stop teacher-vllm-$TS >/dev/null 2>&1
  say "=== TEACHER EVAL DONE（$TREPORT）==="; echo done > "$EV/TEACHER-DONE"
else
  say "★ 教师报告缺失 ⇒ **不 stop**（保留服务以便重试）"; exit 11
fi
"""


def gen_teacher_eval(*, args, manifest: dict, mt: dict, cal: dict, offline: bool, remote,
                     tree: str, ev: str, lower: str, hostmodels: str,
                     hostmodels_mount: str, elam_host: str) -> int:
    """OPD **教师侧**评测的产出者（判据② 的 `eval.teacher_after_acc` 唯一来源）。

    口径（三条，⛔ 全部复用学生侧同一份，不另立）：
      ① 数据集 = `compute_caliber(mt)`（与该 OPD 档学生侧 after/base **同一个**函数、同一份 100 条、
         `dataset_split: train`）；
      ② 解码温度 = 0（`EvalDecoding.model_post_init` 强制，入口不提供覆盖参数）；
      ③ 判分函数 = `src/graspo/eval/criteria.py` 的逐字段完全一致比对（与算法/rank_method 无关）。
    教师制品 = 清单里按文档口径（`TEACHER_FORM`）现算出的那一档训练完成后的产物；
    LoRA 形态 ⇒ 必须 merge 成 merged-hf 才能被 vLLM 加载（走与学生档**同一条** `graspo eval export`）。
    """
    if str(mt.get("algorithm") or "").upper() != "OPD":
        die(f"--role teacher 只用于 OPD 档（判据② 的教师要求只存在于 OPD）；"
            f"清单里 {args.tier} 的 algorithm={mt.get('algorithm')!r} ⇒ 拒绝生成", 1)
    tm = find_teacher_tier(manifest)
    ttier = str(tm["tier_id"])

    # ── §10：按**教师档**形态现算（教师是 27B(Mamba)，与学生档是不是 9B 无关）──────────
    mn = compute_max_num_seqs(tm, args.blocks)
    assert_mamba(mn["needed"], mn["value"], args.blocks)

    # ── §7 形态 + 教师 checkpoint（离线按显式入参/形态推定，并显式标注）─────────────────
    ck, form, form_why = args.checkpoint, args.form, ""
    if not offline:
        ck = ck or remote.run(f"ls -d {tree}/runs/{ttier}/{ttier}/v*/checkpoint-* 2>/dev/null | tail -1")
        if not ck:
            ck = remote.run(f"ls -d {tree}/runs/{ttier}/{ttier}/final 2>/dev/null | tail -1")
        if not ck:
            die(f"教师档 {ttier} 无可加载 checkpoint（§7 前置：教师档必须先于 OPD 档跑完）", 1)
        out = remote.run(f"PYTHONPATH={tree}.hostdeps python3 {tree}/scripts/run_guards.py form "
                         f"--checkpoint {ck}")
        form = (out.split("]")[0].strip("[") if out else "")
        form_why = f"run_guards form ← {out}"
    if not ck:
        ck = "<未探测：dry-run/--no-ssh>"
    if not form:
        form = "full" if str(tm.get("tuner_type")) == "full" else "adapter"
        form_why = (f"未做 §7 实测探测（dry-run/--no-ssh）⇒ 按教师档形态推定 "
                    f"tuner_type={tm.get('tuner_type')!r} ⇒ {form}")

    sys.path.insert(0, str(ROOT / "scripts"))
    import launch_pipeline as lp                                    # noqa: PLC0415
    import record_checks as rc                                      # noqa: PLC0415

    if offline:
        mode = "zero-merge" if form == "full" else "merge"
        mode_why = f"按推定形态 {form} ⇒ {mode}（dry-run：⛔ 未对真实 checkpoint 做 §7 探测）"
    else:
        mode, mode_why = lp.decide_eval_mode(form)
        if mode == "unknown":
            die(f"形态未知（{form!r}）⇒ ⛔ 不得凭惯性选路径（§7）", 1)

    mp = str(tm.get("model_path") or "")
    if not mp:
        die(f"教师档 {ttier}: 清单缺 model_path")
    hostmp = mp.replace("/models/", hostmodels + "/", 1) if mp.startswith("/models/") else mp
    cp = "/runs" + ck[len(tree + "/runs"):] if ck.startswith(tree + "/runs") else ck
    # ★ D6（2026-10-27）：教师合并落点**带上教师档号**，否则报告里 `checkpoint_path` 看不出
    #   "评的是哪个教师" ⇒ `record_checks.py` 的身份门槛 fail-closed ⇒ 判据② 永远落不了结论。
    #   ★ 目录名只从消费端那一个函数取（`record_checks.teacher_merge_dirname`），⛔ 不另写字面量。
    md = f"/evalout/{rc.teacher_merge_dirname(ttier)}" if mode == "merge" else ck
    mdhost = f"{ev}/{rc.teacher_merge_dirname(ttier)}" if mode == "merge" else ck
    report_rel = rc.TEACHER_REPORT_REL.as_posix()

    tmp_dir = Path(args.tmp_dir)
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        die(f"--tmp-dir 不可用：{tmp_dir}（{exc}）—— 用 --tmp-dir 指一个可写目录", 2)

    print(f"TEACHER opd_tier={args.tier} teacher_tier={ttier}"
          f"（文档第 3 章口径：{TEACHER_FORM['model']} / {TEACHER_FORM['algorithm']} / LoRA / dp=1）")
    print(f"CALIBER 与学生侧同一份：fit={cal['fit']} dataset_path={cal['dataset_path']} "
          f"dataset_split={cal['dataset_split']} train_limit={cal['train_limit']}（{cal['why']}）")
    print(f"§10 teacher model={tm.get('model')} needs_limit={mn['needed']} "
          f"max_num_seqs={mn['value']}　（{mn['basis']}）")
    print(f"§7  teacher form={form} mode={mode}　（{form_why}；{mode_why}）")
    print(f"TEACHER_TIER={ttier} MP={mp} HOSTMP={hostmp} CK={ck} CP={cp} MD={md}")
    print(f"教师报告落点 = {ev}/{report_rel}"
          "　（判据② 的 eval.teacher_after_acc 只从这份报告现算）")

    script = render_teacher_script(opd_tier=args.tier, teacher_tier=ttier, gpu=args.gpu,
                                   port=args.port, tree=tree, ev=ev, lower=lower, mode=mode,
                                   form=form, ck=ck, cp=cp, md=md, mdhost=mdhost, mp=mp,
                                   hostmp=hostmp, maxseqs=mn["text"], cal=cal,
                                   hostmodels_mount=hostmodels_mount, elam_host=elam_host,
                                   serving_basis=mn["basis"], needs_limit=mn["needed"],
                                   report_rel=report_rel)

    # ── 产物级 §10 断言：纪律必须落在**将被执行的那份脚本**上（不是函数返回值上）────────
    probs = lp.check_max_num_seqs_in_script(script, mn["needed"],
                                            safe=mn["value"] or _mamba_safe_default())
    if probs:
        die("§10 产物级断言未过：" + "；".join(probs), 1)

    script_path = tmp_dir / f"run-{lower}-teacher-eval.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    yaml_path = tmp_dir / f"eval-{lower}-teacher.yaml"
    yaml_path.write_text(
        render_yaml(role="other", cal=cal, mp=mp,
                    ckpt=(md if mode == "merge" else cp), port=args.port, gpu=args.gpu,
                    judge_method="null", out_dir_rel=report_rel.rsplit("/", 1)[0]),
        encoding="utf-8")
    print("WROTE", str(yaml_path), str(script_path))

    if args.dry_run:
        print("[dry-run] 将执行的教师侧评测脚本（节选，完整见上面 WROTE 的路径）：")
        for line in script.splitlines():
            if any(k in line for k in ("vllm-teacher", "MAXSEQS=", "graspo eval export",
                                       "graspo eval run", "run_guards fresh", "TREPORT=")):
                print("  " + line.strip()[:200])
        print(f"[dry-run] 顺序约束：教师档 {ttier}（{TEACHER_FORM['algorithm']} / LoRA / dp=1）"
              "必须先训练完成并有可加载 checkpoint；⛔ 教师报告缺失时 OPD 档判据② 不可判定")
        print(f"[dry-run] §10：needs_limit={mn['needed']} → {mn['text'] or '<不加该参数>'}")
    return 0


def _ensure_eval_runner(ev: str) -> None:
    """评测前置：把 `<EVAL_DIR>/run-eval.sh` 幂等装到位（唯一产出者 = `scripts/eval_runner.py`）。

    为什么在这里：调用方（本文件 `render_script()` 的两处 `bash "$EV/run-eval.sh" …`、
    `scripts/launch_pipeline.py` 的同名两处）都假定该件已存在；而它此前**没有产出者**
    ⇒ 评测步 `rc=127`（No such file），判据④ 无从取证。故"生成评测脚本之前"先补齐入口。
    幂等：同内容 ⇒ 不动；异内容 ⇒ **拒写**（fail-closed，⛔ 不静默覆盖唯一评测入口）。
    """
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import eval_runner as er                                            # noqa: PLC0415

    try:
        res = er.install(ev, apply=True)
    except FileNotFoundError as exc:
        die(f"{exc} —— EVAL_DIR 必须先存在（mkdir -p {ev}，或用 GRASPO_EVAL_DIR 指定）", 2)
    print(f"[eval-prep] run-eval.sh {res.status} sha256={res.sha256[:16]}… → {res.path}")
    if res.status == "blocked":
        die(f"评测入口未就位：{res.message}", 1)
    print(f"[eval-prep] 自检：bash {res.path} --check-only --self ；"
          f"验 sha：python3 scripts/eval_runner.py check --eval-dir {ev}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M 档评测脚本 + 两侧 yaml 生成器（形态现算，无 T 档硬编码）")
    ap.add_argument("--tier", required=True, help="M 档号（M001–M068）")
    ap.add_argument("--gpu", required=True, help="评测卡集（与训练分卡）")
    ap.add_argument("--port", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--tmp-dir", default=os.environ.get("GRASPO_EVAL_TMP_DIR", "/tmp/graspo-eval"),
                    help="脚本与 yaml 的落点（缺省 $GRASPO_EVAL_TMP_DIR → /tmp/graspo-eval）")
    ap.add_argument("--blocks", type=int, default=None,
                    help="§10 实测可用 Mamba cache blocks（27B 档给则做 ≤blocks 断言并取 min）")
    ap.add_argument("--checkpoint", default=None,
                    help="最终检查点（`--no-ssh` 时必给；否则由 ssh 探测）；"
                         "--role teacher 时指**教师档**的检查点")
    ap.add_argument("--form", default=None, choices=["full", "adapter", "native_rank"],
                    help="§7 形态（`--no-ssh` 时可显式给；缺省按 tuner_type 推定并标注）；"
                         "--role teacher 时指**教师档**的形态")
    ap.add_argument("--no-ssh", action="store_true", help="关闭远端探测（离线自检/干跑；⛔ 不用于真实评测）")
    ap.add_argument("--dry-run", action="store_true", help="只生成命令与判据，不探测远端、不启动 vLLM")
    ap.add_argument("--role", default="student", choices=["student", "teacher"],
                    help="student（缺省）= 本档学生侧 after/base 两侧评测；"
                         "teacher = OPD 教师侧评测（只对 OPD 档；教师档按文档口径从清单现算，"
                         "产物落 <EVAL_DIR>/.local/reports/teacher/eval_report.json）")
    args = ap.parse_args(argv)

    tree = os.environ.get("GRASPO_EVAL_TREE") or str(ROOT)
    lower = args.tier.lower()
    ev = os.environ.get("GRASPO_EVAL_DIR") or os.path.join(
        os.path.dirname(tree.rstrip("/")), f"eval-{lower}-recon")
    # ── 评测前置（唯一真相源）：$EVAL_DIR/run-eval.sh 就位 ──────────────────────────
    #   本文件渲染出的 run-<tier>-eval.sh 在评测步以
    #     `bash "$EV/run-eval.sh" "$T" "$EV" <TIER> after|base "$EV/eval-<tier>-<role>.yaml"`
    #   调用它（`render_script()` 内两处）；该件缺 ⇒ 评测步 rc=127、判据④（vLLM 推理）无法取证。
    #   ⛔ 仓内不另存第二份实现：唯一产出者 = `scripts/eval_runner.py`（本处只调它，幂等 + sha 记录）。
    #   ★ 位置有意在**生成之前**：脚本一旦生成就默认"评测入口可用"，缺件不能等到真跑才暴露。
    _ensure_eval_runner(ev)
    hostmodels = (os.environ.get("GRASPO_MODEL_HOST_ROOT")
                  or os.path.join(os.path.expanduser("~"), "models")).rstrip("/")
    hostmodels_mount = hostmodels
    elam_host = (os.environ.get("GRASPO_ELAM_V5_DIR")
                 or os.path.join(os.path.expanduser("~"), "elam-v5-balanced")).rstrip("/")

    manifest = load_manifest(Path(args.manifest))
    mt = tier_of(manifest, args.tier)
    cal = compute_caliber(mt)
    judge_method = ALGO_METHOD.get(str(mt.get("algorithm") or "").upper())
    if not judge_method:
        die(f"清单里 {args.tier} 的 algorithm={mt.get('algorithm')!r} 不是 SFT/GRASPO/CPT/OPD 之一"
            "（判据② 通过线逐算法分列，⛔ 不得靠 role 推断）")

    mn = compute_max_num_seqs(mt, args.blocks)
    assert_mamba(mn["needed"], mn["value"], args.blocks)

    offline = args.no_ssh or args.dry_run
    remote = Remote(enabled=not offline, port=os.environ.get("GRASPO_EVAL_SSH_PORT", "22"),
                    user=os.environ.get("GRASPO_EVAL_SSH_USER"),
                    host=os.environ.get("GRASPO_EVAL_SSH_HOST"))

    # ── 教师侧：判据② 的 OPD 教师读数（与两侧同一份口径，⛔ 不另立参数）────────────────
    if args.role == "teacher":
        return gen_teacher_eval(args=args, manifest=manifest, mt=mt, cal=cal, offline=offline,
                                remote=remote, tree=tree, ev=ev, lower=lower,
                                hostmodels=hostmodels, hostmodels_mount=hostmodels_mount,
                                elam_host=elam_host)

    # ── §7 形态 + checkpoint（离线时按显式入参/形态推定，并**显式标注**）──────────────
    ck, form, form_why = args.checkpoint, args.form, ""
    if not offline:
        ck = ck or remote.run(f"ls -d {tree}/runs/{args.tier}/{args.tier}/v*/checkpoint-* 2>/dev/null | tail -1")
        if not ck:
            ck = remote.run(f"ls -d {tree}/runs/{args.tier}/{args.tier}/final 2>/dev/null | tail -1")
        if not ck:
            die("无 checkpoint（§7 前置：先跑完训练再接评测）", 1)
        out = remote.run(f"PYTHONPATH={tree}.hostdeps python3 {tree}/scripts/run_guards.py form --checkpoint {ck}")
        form = (out.split("]")[0].strip("[") if out else "")
        form_why = f"run_guards form ← {out}"
    if not ck:
        ck = "<未探测：dry-run/--no-ssh>"
    if not form:
        form = "full" if str(mt.get("tuner_type")) == "full" else "adapter"
        form_why = (f"未做 §7 实测探测（dry-run/--no-ssh）⇒ 按档位形态推定 "
                    f"tuner_type={mt.get('tuner_type')!r} ⇒ {form}")

    sys.path.insert(0, str(ROOT / "scripts"))
    import launch_pipeline as lp                                    # noqa: PLC0415

    if offline:
        mode = "zero-merge" if form == "full" else "merge"
        mode_why = f"按推定形态 {form} ⇒ {mode}（dry-run：⛔ 未对真实 checkpoint 做 §7 探测）"
    else:
        mode, mode_why = lp.decide_eval_mode(form)
        if mode == "unknown":
            die(f"形态未知（{form!r}）⇒ ⛔ 不得凭惯性选路径（§7）", 1)

    mp = str(mt.get("model_path") or "")
    if not mp:
        die(f"{args.tier}: 清单缺 model_path")
    hostmp = mp.replace("/models/", hostmodels + "/", 1) if mp.startswith("/models/") else mp
    cp = "/runs" + ck[len(tree + "/runs"):] if ck.startswith(tree + "/runs") else ck
    merge = f"/evalout/merged-hf-{lower}"
    mergehost = f"{ev}/merged-hf-{lower}"
    md = merge if mode == "merge" else ck
    mdhost = mergehost if mode == "merge" else ck

    tmp_dir = Path(args.tmp_dir)
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        die(f"--tmp-dir 不可用：{tmp_dir}（{exc}）—— 用 --tmp-dir 指一个可写目录", 2)

    print(f"CALIBER tier={args.tier} fit={cal['fit']} dataset_path={cal['dataset_path']} "
          f"dataset_split={cal['dataset_split']} train_dataset_path={cal['train_dataset_path']} "
          f"train_limit={cal['train_limit']}　（{cal['why']}）")
    print(f"§10 tier={args.tier} model={mt.get('model')} needs_limit={mn['needed']} "
          f"max_num_seqs={mn['value']}　（{mn['basis']}）")
    print(f"§7  tier={args.tier} form={form} mode={mode}　（{mode_why}）")
    print(f"TIER={args.tier} MP={mp} HOSTMP={hostmp} CK={ck} CP={cp} MD={md}")

    script = render_script(tier=args.tier, gpu=args.gpu, port=args.port, tree=tree, ev=ev,
                           lower=lower, mode=mode, form=form, ck=ck, cp=cp, md=md,
                           mdhost=mdhost, mp=mp, hostmp=hostmp, maxseqs=mn["text"],
                           cal=cal, judge_method=judge_method, hostmodels_mount=hostmodels_mount,
                           elam_host=elam_host, serving_basis=mn["basis"],
                           needs_limit=mn["needed"])

    # ── 产物级 §10 断言：纪律必须落在**将被执行的那份脚本**上（不是函数返回值上）────────
    probs = lp.check_max_num_seqs_in_script(script, mn["needed"],
                                            safe=mn["value"] or _mamba_safe_default())
    if probs:
        die("§10 产物级断言未过：" + "；".join(probs), 1)

    script_path = tmp_dir / f"run-{lower}-eval.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    after_cp = merge if mode == "merge" else cp
    for role, cpv in (("after", after_cp), ("base", "null")):
        (tmp_dir / f"eval-{lower}-{role}.yaml").write_text(
            render_yaml(role=role, cal=cal, mp=mp, ckpt=cpv, port=args.port, gpu=args.gpu,
                        judge_method=judge_method), encoding="utf-8")
    print("WROTE yamls:", str(tmp_dir / f"eval-{lower}-after.yaml"),
          str(tmp_dir / f"eval-{lower}-base.yaml"))
    print("WROTE", str(script_path))

    if args.dry_run:
        # ★ 只打印将要执行的命令与判据，⛔ 不起 vLLM / 不 ssh（零副作用，除上述产物）
        print("[dry-run] 将执行的评测脚本（节选，完整见上面 WROTE 的路径）：")
        for line in script.splitlines():
            if any(k in line for k in ("vllm-after", "vllm-base", "MAXSEQS=", "graspo eval export",
                                       "graspo eval delta", "run_guards fresh")):
                print("  " + line.strip()[:200])
        print(f"[dry-run] 判据② 口径：judge_method={judge_method}（清单 algorithm="
              f"{mt.get('algorithm')}）；阈值由 src/graspo/eval/thresholds.py 单源读取")
        print(f"[dry-run] §10：needs_limit={mn['needed']} → {mn['text'] or '<不加该参数>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
