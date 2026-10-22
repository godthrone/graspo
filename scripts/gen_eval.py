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
  · `--tmp-dir`（缺省 `$GRASPO_EVAL_TMP_DIR` → `/data/remote`）：脚本与 yaml 的落点；
  · `--no-ssh`：关闭远端探测（form / checkpoint / 清单读取），改由 `--checkpoint` / `--form`
    显式给出或按形态推定 —— 用于离线自检与干跑（⛔ 不用于真实评测）。

用法：
    python3 scripts/gen_eval.py --tier M001 --gpu 4 --port 18889
    python3 scripts/gen_eval.py --tier M041 --gpu 4,5 --port 18889 --blocks 232
    python3 scripts/gen_eval.py --tier M041 --gpu 4 --port 18889 --dry-run --no-ssh --tmp-dir /tmp/x
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

#: 唯一真相源：§10 的安全上限（228 实测 27B(Mamba) 可用 cache blocks = 232 ⇒ 取 192）。
#: ⛔ 本文件不另写数值；需要时直接引用 `scripts/launch_pipeline.py` 的常量。
def _mamba_safe_default() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import launch_pipeline as lp                                    # noqa: PLC0415

    return int(lp.MAMBA_SAFE_MAX_NUM_SEQS)


#: 数据集口径里"平台方自有评测集"（正式档）——环境信息，可用环境变量覆盖。
TEST_JSONL = "/work/elam-v5/data/test.jsonl"
FULL_TRAIN_JSONL = "/work/elam-v5/data/train.jsonl"

#: 算法（清单 `algorithm`）⇒ 判据② 冻结口径键。只是名字映射，⛔ 不含阈值数值。
ALGO_METHOD = {"SFT": "sft", "GRASPO": "graspo", "CPT": "cpt", "OPD": "opd"}

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
                         f"（228 实测可用 cache blocks=232；未给 --blocks ⇒ 不做 ≤blocks 断言，真跑前必须给）",
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
                judge_method: str) -> str:
    return f"""dataset_path: {cal["dataset_path"]}
train_dataset_path: {cal["train_dataset_path"]}
train_limit: {cal["train_limit"]}
dataset_split: {cal["dataset_split"]}
base_model_path: {mp}
checkpoint_path: {ckpt}
role: {role}
export_format: null
merged_output_dir: null
output_dir: /evalout/.local/reports/{role}
endpoint: http://127.0.0.1:{port}
served_model_name: graspo-eval
gpus: "{gpu}"
max_workers: 8
seed: 42
judge_method: {judge_method}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M 档评测脚本 + 两侧 yaml 生成器（形态现算，无 T 档硬编码）")
    ap.add_argument("--tier", required=True, help="M 档号（M001–M068）")
    ap.add_argument("--gpu", required=True, help="评测卡集（与训练分卡）")
    ap.add_argument("--port", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--tmp-dir", default=os.environ.get("GRASPO_EVAL_TMP_DIR", "/data/remote"),
                    help="脚本与 yaml 的落点（缺省 $GRASPO_EVAL_TMP_DIR → /data/remote）")
    ap.add_argument("--blocks", type=int, default=None,
                    help="§10 实测可用 Mamba cache blocks（27B 档给则做 ≤blocks 断言并取 min）")
    ap.add_argument("--checkpoint", default=None, help="最终检查点（`--no-ssh` 时必给；否则由 ssh 探测）")
    ap.add_argument("--form", default=None, choices=["full", "adapter", "native_rank"],
                    help="§7 形态（`--no-ssh` 时可显式给；缺省按 tuner_type 推定并标注）")
    ap.add_argument("--no-ssh", action="store_true", help="关闭远端探测（离线自检/干跑；⛔ 不用于真实评测）")
    ap.add_argument("--dry-run", action="store_true", help="只生成命令与判据，不探测远端、不启动 vLLM")
    args = ap.parse_args(argv)

    tree = os.environ.get("GRASPO_EVAL_TREE") or str(ROOT)
    lower = args.tier.lower()
    ev = os.environ.get("GRASPO_EVAL_DIR") or os.path.join(
        os.path.dirname(tree.rstrip("/")), f"eval-{lower}-recon")
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
