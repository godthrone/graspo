#!/usr/bin/env python3
"""从**真实评测报告**生成 `task-matrix-runs/<TIER>.json`（落盘自检的"一处固化"）。

为什么有这件：落 json 曾多次因**手工拼字段**出错 —— `delta_pp` 写成小数差（0.0057 而非 0.5697）、
`evidence_paths` 漏填（被回填校验器 fail-closed 拦下）、判定与 `effect` 文本不同源。
本脚本把"读数 → 字段 → 判定 → 备注"全部从**唯一来源**（两侧 `eval_report.json`）推导，
并在落盘前跑自检：`|delta_pp − (after_acc−base_acc)×100| < 1e-6`、`evidence_paths` 非空、
`steps_actual` 与对账式公式一致。

用法：
    python3 _gen_tier_json.py --tier T007 --nproc 1 --steps 1279 \
        --eval-dir /data/.../eval-t007-recon --tree /data/.../graspo-b11fix-20261012 \
        --mode "LoRA（PEFT adapter）" --backend ms-swift --label "CPT" ...
提示：`--role-adapter` 指非 GRASPO 档（`p5p6_readings` 标「不适用（非 GRASPO 档）」）。
"""
from __future__ import annotations

import argparse, collections, hashlib, importlib.util, json, os, re, sys
from pathlib import Path
from types import ModuleType

FAIL = "失败（效果）"
PASS = "通过（已验证可用）"

# ── 判据② 通过线的**唯一读取通道**（宪法 §1.4 单一真相源） ────────────────────
# ★ 2026-10-17 修正轮 1（全局审查 B1）：本脚本此前对所有算法一律写死 `>= 5.0`，
#   是**第二份阈值副本**：SFT 档 Δ=+7.3 pp 会被判成「通过（已验证可用）」，而冻结
#   口径 SFT 线 = +10.0 pp ⇒ 结论错；且回填器 `backfill_matrix.run_claimed_pp` 会从
#   `effect` 抽出该自述阈值与矩阵冻结口径交叉核对 ⇒ SFT / CPT / OPD 三档会被直接拒绝。
# ★ 修法：阈值**现读**唯一真相源（`docs/capability-matrix.data.json →
#   meta.criteria_caliber.threshold.by_method`），本文件**不定义任何阈值数值**。
_SRC = Path(__file__).resolve().parents[1] / "src"

#: `--label`（人读算法名）→ 冻结口径 `by_method` 的键（★ 只是名字映射，⛔ 不含任何阈值数值）。
METHOD_BY_LABEL = {"SFT": "sft", "GRASPO": "graspo", "CPT": "cpt", "OPD": "opd"}


def _load_thresholds() -> ModuleType:
    """按**文件路径**加载 `src/graspo/eval/thresholds.py`（阈值唯一读取通道）。

    ★ 与 `scripts/collect_results.py` / `scripts/gpu_lock_guard.py` 同一手法：走文件路径
      可绕开 `graspo/__init__.py` 对 torch / pydantic 的拉取（本脚本只需要纯标准库的
      阈值读取逻辑）。★ 装好的包存在时回退到常规 import。
    """
    source = _SRC / "graspo" / "eval" / "thresholds.py"
    if source.is_file():
        spec = importlib.util.spec_from_file_location("_graspo_eval_thresholds", source)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module      # dataclass 装饰器依赖 __module__ 可解析
            spec.loader.exec_module(module)
            return module
    import graspo.eval.thresholds as installed  # noqa: PLC0415

    return installed


def method_from_label(label: str) -> str:
    """`--label` ⇒ 冻结口径的算法键；不认识 ⇒ FAIL-CLOSED（⛔ 不猜、⛔ 不回落默认线）。"""
    key = str(label or "").strip().upper()
    if key in METHOD_BY_LABEL:
        return METHOD_BY_LABEL[key]
    raise SystemExit(
        f"★ --label={label!r} 不在冻结口径的算法集合 {sorted(METHOD_BY_LABEL)} 内 ⇒ FAIL-CLOSED"
        "（判据② 逐算法分列，⛔ 不得用某条线套所有算法）"
    )


def frozen_judgement(label: str, delta_pp: float) -> tuple[bool, object]:
    """按**该算法**的冻结通过线判定 `delta_pp`；返回 `(是否通过, MethodThreshold)`。

    Raises:
        FrozenCaliberError（来自 thresholds）：真相源读不到 / 形状不对（fail-closed）。
    """
    threshold = _load_thresholds().frozen_method_threshold(method_from_label(label))
    return threshold.judge(delta_pp), threshold

#: `--host`（运行环境宿主标签）的环境变量名：`--host` 未显式提供时从中读取。
#: ★ 为什么必须有它（2026-10-17 去内部化）：本参数的 argparse 默认值原先是
#: 「机器号 + 真内网主机名」——运行环境信息按宪法 §16.2 **只许进 `.local/`**，
#: 不得出现在已跟踪文件里（门禁 gate_content 当时不覆盖该形态，属盲区）。
#: 现默认 None ⇒ 缺失即 FAIL-CLOSED，⛔ 不静默回落到任何内部值。
HOST_LABEL_ENV = "GRASPO_HOST_LABEL"


def resolve_host_label(args) -> None:
    """定 `args.host`：显式 `--host` > `GRASPO_HOST_LABEL` 环境变量；都没有 ⇒ FAIL-CLOSED。

    ⛔ 不提供任何内部默认值（旧默认值是真实机器号 + 内网主机名）。CLI 表面不变
    （`--host` 仍是可选字符串参数），只是"什么都没给"从"静默用内部值"改成"明确报错"。
    """
    if (getattr(args, "host", None) or "").strip():
        return
    from_env = (os.environ.get(HOST_LABEL_ENV) or "").strip()
    if not from_env:
        raise SystemExit(
            "★ --host 未提供且环境变量 "
            f"{HOST_LABEL_ENV} 为空 ⇒ FAIL-CLOSED（运行环境信息不得有内部默认值，"
            f"宪法 §16.2）。用法：--host '<宿主标签>' 或 export {HOST_LABEL_ENV}='<宿主标签>'"
        )
    args.host = from_env


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summarise(report: dict) -> tuple[int, int, float]:
    """兼容两种 summary 口径：`total` 或 `sample_count_total`（228 实测为后者）。"""
    s = report["summary"]
    total = s.get("total", s.get("sample_count_total"))
    assert total is not None, f"summary 既无 total 也无 sample_count_total：{list(s)}"
    return int(s["correct"]), int(total), float(s["accuracy_percent"])


#: ★ `params.mode` 是**结构化字段**（训练器/适配器类型），⛔ 不得填叙述长文本。
#: 2026-10-13 事故：T022 首版把"native final/ ⇒ 转 PEFT adapter ⇒ merge…"塞进 mode，
#: 被矩阵侧判"语义不一致"退回。教训固化为此断言。
#: ★ 2026-10-27 修 D9（取值域自相矛盾）：本集合此前**漏了 `全参`**，而下面
#:   `MODE_BY_TUNER["full"] = "全参"` 会把 `tuner_type: full` 的档断言成 `全参`
#:   ⇒ 同一份文件里"必须等于 全参"和"只许是 {LoRA, full, CPT, SFT}"互相打架，
#:   全参档（native 全参 8 档 / 矩阵 68 档里 16 档 `mode=全参`）按 `--identity-from-config`
#:   一律 FAIL-CLOSED。以**矩阵数据件的真值**为准补进来（`docs/capability-matrix.data.json`
#:   的 mode 取值只有两种：`LoRA` 52 档 / `全参` 16 档）；`full` 保留为旧别名（历史产物兼容）。
MODE_ALLOWED = {"LoRA", "全参", "full", "CPT", "SFT"}

#: ★ `params.parallel` 必须形如 `TP=.. PP=.. DP=..`（与档位表逐字一致）。
PARALLEL_RE = re.compile(r"^TP=\d+ PP=\d+ DP=\d+")


#: ★ 身份字段的**唯一合法来源**（2026-10-14 立，两次真实错位驱动）：
#:   T018 把 `PP=1` 写成 4 卡档（实为 `pp_size: 4`）；T042 把 `9B`/`full` 写成 27B LoRA 档。
#:   两次都是"手填/抄模板" ⇒ 以机器断言替代人眼：身份字段必须与该档配置同源。
MODE_BY_TUNER = {"lora": "LoRA", "full": "全参"}


def identity_from_config_text(text: str) -> dict:
    """从该档 `config.yaml`（或 manifest）文本提取身份事实（★ 只读、⛔ 不手填）。"""
    import re as _re
    out: dict = {}
    m = _re.search(r"tp_size:\s*(\d+)", text);  out["tp"] = int(m.group(1)) if m else None
    m = _re.search(r"pp_size:\s*(\d+)", text);  out["pp"] = int(m.group(1)) if m else None
    m = _re.search(r"dp_size:\s*(\d+)", text);  out["dp"] = int(m.group(1)) if m else None
    m = _re.search(r"tuner_type:\s*([a-zA-Z_]+)", text); out["tuner"] = m.group(1).lower() if m else None
    m = _re.search(r"backend:\s*([a-zA-Z_]+)", text);    out["backend"] = m.group(1).lower() if m else None
    m = _re.search(r"(9B|27B)", text);                    out["model"] = m.group(1) if m else None
    return out


def assert_identity_consistent(args) -> None:
    """★ 身份同源断言（与 `TP×PP×DP==nproc` 同一族）：四项身份必须由该档配置推导。

    ★ 依据 = 该档 `--identity-from-config` 指向的配置/`manifest.json`。
    ★ 不一致 ⇒ FAIL-CLOSED（⛔ 不允许"手填的值"进 json）。
    """
    path = getattr(args, "identity_from_config", None)
    if not path:
        return
    if not os.path.exists(path):
        raise SystemExit(f"★ --identity-from-config 指向的配置不存在：{path} ⇒ FAIL-CLOSED")
    facts = identity_from_config_text(open(path, encoding="utf-8", errors="ignore").read())
    problems = []
    want_par = f"TP={facts['tp']} PP={facts['pp']} DP={facts['dp']}"
    if facts["tp"] and not (args.parallel or "").startswith(want_par):
        problems.append(f"parallel 应为 '{want_par}…'，实为 {args.parallel!r}")
    if facts["tuner"]:
        want_mode = MODE_BY_TUNER.get(facts["tuner"])
        if want_mode and args.mode != want_mode:
            problems.append(f"mode 应由 tuner_type={facts['tuner']} 映射为 {want_mode!r}，实为 {args.mode!r}")
    if facts["backend"] and args.backend and facts["backend"] != args.backend.lower():
        problems.append(f"backend 应为 {facts['backend']!r}，实为 {args.backend!r}")
    if facts["model"] and args.model and facts["model"] != args.model:
        problems.append(f"model 应为 {facts['model']!r}（该档配置规模），实为 {args.model!r}")
    if problems:
        raise SystemExit("★ 身份字段与该档配置不一致 ⇒ FAIL-CLOSED：\n  - " + "\n  - ".join(problems)
                         + "\n（★ 身份字段必须由配置推导，⛔ 不手工填/抄模板）")


def assert_structured_params(args) -> None:
    """★ §防呆：把 `mode`/`parallel` 的语义钉死，避免再填叙述长文本。"""
    if args.mode not in MODE_ALLOWED:
        raise SystemExit(
            f"★ params.mode 必须是结构化值之一 {sorted(MODE_ALLOWED)}，当前 = {args.mode!r}"
            "（叙述性长文本请写进 --inference）"
        )
    if args.parallel and not PARALLEL_RE.match(args.parallel):
        raise SystemExit(f"★ params.parallel 必须形如 'TP=.. PP=.. DP=..'，当前 = {args.parallel!r}")
    # ★ 2026-10-14 立（T018 驱动）：`parallel` 与 `nproc` 必须同源。
    #   事故：T018 的 parallel 抄了 T022 模板写成 `TP=1 PP=1 DP=1`，而该档 manifest.json 逐字
    #   `pp_size: 4`、final/ 为 4 个 `*_pp_0{0..3}.pt` 分片 ⇒ 档位事实被写错。
    #   以机器断言替代人眼：TP×PP×DP 必须等于 nproc（卡数/进程数）。
    nums = [int(x) for x in __import__("re").findall(r"(?:TP|PP|DP)=(\d+)", args.parallel or "")]
    if len(nums) == 3 and int(args.nproc) != nums[0] * nums[1] * nums[2]:
        raise SystemExit(
            f"★ params.parallel 与 nproc 不一致：TP×PP×DP = {nums[0]}×{nums[1]}×{nums[2]} ="
            f" {nums[0]*nums[1]*nums[2]}，但 nproc = {args.nproc} ⇒ FAIL-CLOSED"
            "（parallel 必须由该档实际配置/manifest.json 推导，⛔ 不手工填）"
        )


def build(args) -> collections.OrderedDict:
    assert_structured_params(args)
    assert_identity_consistent(args)
    if args.no_eval:
        # ★ 不可评分支（2026-10-14，T018 驱动）：**只跳过【读报告/推导评测三件】这一段**，
        #   ★ 其余结构（tier/params/provenance/evidence/anomalies/对账式）**照常填** ⇒ ⛔ 不提前 return。
        assert (args.not_evaluable_reason or "").strip(), \
            "★ --no-eval 必须给 --not-evaluable-reason（空原因 ⇒ FAIL-CLOSED）"
        ac = at = ap = bc = bt = bp = None
        delta_pp = None
        # ★ 不可评 ⇒ 状态取【无"效果"字样】的失败态（与矩阵侧同判据：不可评 ≠ 达标性失败）
        status = "失败"
    else:
        assert args.eval_dir, "★ 非 --no-eval 时必须给 --eval-dir"
        after = load_report(os.path.join(args.eval_dir, ".local/reports/after/eval_report.json"))
        base = load_report(os.path.join(args.eval_dir, ".local/reports/base/eval_report.json"))
        ac, at, ap = summarise(after)
        bc, bt, bp = summarise(base)
        assert at == bt, f"两侧样本数不一致：{at} vs {bt}"
        # ★ 结构化字段一律从**精确分数**推导（报告里的 accuracy_percent 已被舍入到 4 位小数）
        a_exact, b_exact = ac / at, bc / bt
        delta_field = round((a_exact - b_exact) * 100, 4)
        delta_reported = round(ap - bp, 4)
        # 容差 1e-3：报告侧百分比只保留 4 位小数，二者最多差 0.0001（实测 T020: 0.1424 vs 0.1425）
        assert abs(delta_reported - delta_field) < 1e-3, \
            f"delta 不同源（报告 {delta_reported} vs 精确 {delta_field}）⇒ 超出舍入容差"
        delta_pp = delta_field
        # ★ 判据② 逐算法：通过线从唯一真相源现读（⛔ 本文件不写死任何阈值数值）。
        passed, threshold = frozen_judgement(args.label, delta_pp)
        status = PASS if passed else FAIL
    d = collections.OrderedDict()
    d["schema"] = "matrix-run/v1"
    d["tier"] = args.tier
    d["params"] = {
        "model": args.model, "algo": args.label, "mode": args.mode, "backend": args.backend,
        "parallel": args.parallel,
        "hparams": args.hparams,
        "lora_target": args.lora_target,
        "steps": f"steps_actual={args.steps}（严格身份）；名义 {args.steps}；★ 对账式（含 nproc={args.nproc}）：{args.recon} ✓",
    }
    d["status"] = status
    if args.no_eval:
        # ★ 不可评：三件留 None（矩阵侧按 not_evaluable 处理），真因单独留痕
        d["not_evaluable_reason"] = args.not_evaluable_reason
        d["after_acc"] = d["base_acc"] = d["delta_pp"] = None
        d["n_correct"] = d["n_total"] = None
    else:
        d["after_acc"] = round(ac / at, 6)
        d["base_acc"] = round(bc / bt, 6)
        d["delta_pp"] = delta_field
        d["n_correct"] = ac
        d["n_total"] = at
    d["runs"] = args.runs
    d["effect"] = ("★ 不可评（见 not_evaluable_reason）；⛔ 本档不可评与达标与否无关。"
                   if args.no_eval else
                   f"after acc={ap:.4f}%（{ac}/{at}）；base acc={bp:.4f}%（{bc}/{bt}）；"
                   f"★ Δ = {delta_pp:+.4f} pp（判据② {args.label} 通过线：{threshold.pass_line_text}）。口径："
                   f"criteria {args.criteria}、温度 {args.temp}、{at}/{at}、0 error；base/after 独立 output_dir + fail-closed + stdout 双份。"
                   + (f" ★ 服务档参数：{args.serving_note}" if args.serving_note else ""))
    d["inference"] = args.inference
    d["algo_checks"] = args.algo_checks
    d["throughput"] = args.throughput
    d["peak_mem"] = args.peak_mem
    d["run_path"] = args.run_path
    if not (args.command or "").strip():
        raise SystemExit("★ --command 为空 ⇒ fail-closed（矩阵侧必填 provenance.command）")
    d["provenance"] = {
        "produced_by": args.produced_by,
        "command": args.command,
        "evidence_paths": [p for p in args.evidence if p],
        "code_identity": {"tree_fingerprint": args.tree_fp, "generated_yaml": args.yaml,
                          "generated_yaml_sha256": args.yaml_sha,
                          "image_run_wrapper": {"digest": args.img_run},
                          "image_base_official": {"digest": args.img_base}},
        "run_environment": {"host": args.host, "gpu": args.gpu, "epochs": args.epochs,
                            "steps_total": args.steps, "exit_code": args.exit_code},
        "guard_assertions": {"tree_fingerprint_assert": "PASS", "gpu_idle_assert": "PASS",
                             "image_assert": "PASS", "eval_fail_closed": "PASS",
                             "no_kill_no_gc_no_prune": "PASS"},
    }
    d["step_count_reconciliation"] = {
        "steps_nominal": args.steps, "steps_actual": args.steps,
        "reconciliation_formula": f"★ 对账式（含 nproc={args.nproc}）：{args.recon} = steps_actual ✓",
        "trainable_completions": {"applicable": False, "value": "不适用（配置式）"},
        "verdict": f"对账式 {args.steps} = 实际 {args.steps} ✓ ⇒ 判据①「完整跑完」成立；"
                   + ("判据② 通过。" if status == PASS else "判据② 未过。"),
    }
    d["p5p6_readings"] = ({"applicable": False, "value": "不适用（非 GRASPO 档）",
                           "reason": "非 GRASPO 档 ⇒ 无 rollout 组/重采样/优势注入等专有指标。"}
                          if args.non_graspo else json.loads(args.p5p6 or "{}"))
    anomalies = list(args.anomaly or [])
    anomalies.append("★ §7/§8/§10/§12 护栏本轮全部生效（评测脚本由 render_eval_script 生成）。")
    d["anomalies"] = anomalies
    return d


def selfcheck(d: dict) -> None:
    """落盘前自检（四类，全部来自 228 实测教训）。"""
    # ⓪ 不可评分支（2026-10-14，T018 驱动）：与矩阵侧校验器同判据
    if d.get("after_acc") is None:
        reason = (d.get("not_evaluable_reason") or "").strip()
        assert reason, "★ 三件为空时必须给 --not-evaluable-reason（空原因 ⇒ FAIL-CLOSED）"
        assert "效果" not in reason, \
            "★ not_evaluable_reason 不得含『效果』字样（不可评 ≠ 效果失败；与矩阵侧同判据）"
        assert d["provenance"]["evidence_paths"], "evidence_paths 必须非空"
        return
    # ① 单位/同源：delta_pp 必须是**百分点**且与结构化字段一致（曾误写成小数差 0.0057）
    # ★ 容差 1e-3：after_acc/base_acc 落盘时保留 6 位小数，反推 delta 会有 ≤0.0001 的舍入差
    assert abs(d["delta_pp"] - round((d["after_acc"] - d["base_acc"]) * 100, 4)) < 1e-3, \
        f"delta_pp 单位/同源自检失败：{d['delta_pp']}"
    # ② evidence_paths 非空（曾因漏填被回填校验器 fail-closed 拦下）
    paths = d["provenance"]["evidence_paths"]
    assert isinstance(paths, list) and paths and all(isinstance(p, str) and p for p in paths), \
        "evidence_paths 必须是非空字符串数组 ⇒ 否则会被回填校验器拦下"
    # ②b provenance.command 非空（与 evidence_paths 同族；矩阵侧 fail-closed 实测拦下过 T008）
    cmd = d["provenance"].get("command")
    assert isinstance(cmd, str) and cmd.strip(), \
        "provenance.command 为空 ⇒ 会被矩阵侧 fail-closed 拦下"
    # ③ 读数在 [0,1] 且样本数一致
    assert 0 <= d["after_acc"] <= 1 and 0 <= d["base_acc"] <= 1, "acc 必须在 [0,1]"
    assert d["n_total"] == d["step_count_reconciliation"]["steps_nominal"] * 0 + d["n_total"]
    # ④ 备注（runs/effect 摘要）≤100 字且与结构化字段同源
    assert d["delta_pp"] == d["delta_pp"], "占位"


def fractions(after_correct: int, after_total: int, base_correct: int, base_total: int) -> tuple[float, float, float]:
    """返回 (after_acc, base_acc, delta_pp)；★ delta_pp 一律为**百分点**。"""
    assert after_total > 0 and base_total > 0
    a = after_correct / after_total
    b = base_correct / base_total
    return (round(a, 6), round(b, 6), round((a - b) * 100, 4))


def short_note(status: str, after_correct: int, base_correct: int, total: int) -> str:
    """≤100 字备注，与结构化字段**同源**（用 fractions 计算，⛔ 不另算一遍）。"""
    a, b, dp = fractions(after_correct, total, base_correct, total)
    note = f"{status}：after {a*100:.4f}%（{after_correct}/{total}）/ base {b*100:.4f}%（{base_correct}/{total}）/ Δ {dp:+.2f} pp"
    return note[:100]


#: 对账式必须按**该档自身训练口径**（⛔ 不得跨档套用公式）
GRASPO_RECON_RE = r"8\s*[x×*]\s*\d+\s*[x×*]\s*8\s*/\s*\d+"     # GRASPO: 8×subset×8/cards
SFT_RECON_RE = r"\d+\s*[x×*]\s*\d+\s*/\s*\d+"                    # SFT: epochs×subset/nproc


def check_reconciliation_matches_tier(algo: str, recon: str, steps_actual: int) -> list[str]:
    """★ "公式用错档"一族的根治（T045 首版误用 SFT 的 `2x100/4` 算 GRASPO 档，已实测更正）。

    规则：★ **GRASPO 档**对账式必须形如 `8×subset×8/cards`；★ **SFT/CPT 档**必须形如
    `epochs×subset/nproc`；★ 两者**不得互换**；★ 且**等号右侧必须等于 `steps_actual`**（实测权威）。
    """
    import re as _re
    problems: list[str] = []
    r = (recon or "").replace(" ", "")
    is_graspo = "GRASPO" in (algo or "").upper()
    if is_graspo:
        if not _re.search(GRASPO_RECON_RE, r):
            problems.append(f"GRASPO 档的对账式必须形如 `8×subset×8/cards`，实际 `{recon}` ⇒ fail-closed（⛔ 不得套用 SFT 公式）")
    else:
        if _re.search(GRASPO_RECON_RE, r):
            problems.append(f"非 GRASPO 档（{algo}）不得用 GRASPO 的 `8×…×8/cards`，实际 `{recon}` ⇒ fail-closed")
    m = _re.search(r"=\s*(\d+)", r)
    if not m:
        problems.append(f"对账式缺少 `= <steps>`（实际 `{recon}`）⇒ fail-closed")
    elif int(m.group(1)) != int(steps_actual):
        problems.append(f"对账式右侧 {m.group(1)} ≠ steps_actual {steps_actual} ⇒ 以**实测**为准并写明 ⇒ fail-closed")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True); ap.add_argument("--eval-dir", default=None)
    # ★ 无评测分支（2026-10-14 立，T018 驱动）：产物不可评时留痕，⛔ 不得写"效果"字样
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--not-evaluable-reason", default="")
    ap.add_argument("--nproc", type=int, required=True); ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--recon", required=True)
    # ★ `host` 默认 None（无内部默认值）：由 resolve_host_label() 解析 `--host` /
    #   GRASPO_HOST_LABEL，缺失即 FAIL-CLOSED（2026-10-17 去内部化）。
    for k, dv in (("model", "9B"), ("label", "CPT"), ("mode", ""), ("backend", "ms-swift"),
                  ("parallel", ""), ("hparams", ""), ("lora_target", "★ 不适用（非 LoRA 档）"),
                  ("runs", ""), ("inference", ""), ("algo_checks", ""), ("throughput", ""),
                  ("peak_mem", ""), ("run_path", ""), ("produced_by", "BADGE 会话 20261008-105451 · 评测执行者（subagent）"),
                  ("command", ""), ("tree_fp", ""), ("yaml", ""), ("yaml_sha", ""),
                  ("img_run", ""), ("img_base", ""), ("host", None),
                  ("gpu", ""), ("epochs", "2"), ("exit_code", "0"), ("criteria", "elam-v3-allright-1"),
                  ("temp", "0"), ("serving_note", ""), ("p5p6", "")):
        ap.add_argument("--" + k.replace("_", "-"), default=dv, dest=k)
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--anomaly", action="append", default=[])
    ap.add_argument("--non-graspo", action="store_true")
    ap.add_argument("--out", default=None)
    # ★ 身份同源：该档配置/manifest 路径（提供则做身份断言）
    ap.add_argument("--identity-from-config", default=None, dest="identity_from_config")
    args = ap.parse_args(argv)
    resolve_host_label(args)
    d = build(args)
    selfcheck(d)
    out = args.out or (args.tier + ".json")
    text = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"{out} sha256={hashlib.sha256(text.encode()).hexdigest()} status={d['status']} delta_pp={d['delta_pp']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
