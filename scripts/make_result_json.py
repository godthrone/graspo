#!/usr/bin/env python3
"""跑次读数 ⇒ **回填口径**的结果 json（`matrix-run/v1`）—— 唯一产出者。

为什么有这件（工作包 B / 2026-10-22，链路侦察断点 B1+B3）：
  · `scripts/gen_tier_json.py` 的产物**结构上过不了回填器**：它写死 `status`（回填器见 `status`
    一律 rc=1 拒收）、缺 `run_result`/`steps_actual`/`elapsed`/`note`、评测三件在**顶层**
    （回填器要 `eval{}`）、`algo_checks`/`inference` 是**自由文本**（回填器要 `checks{bool,bool}`）。
  · `elapsed` / `throughput` 全仓**无产出脚本**，`peak_mem` 与台账里的 `peak_memory_gib` 名字也不对。
  ⇒ 没有一件脚本能产出"可回填"的 json。本文件是那个**薄适配器**：把
    ① runner 的原始实测读数（`<run-dir>/readings.json` / `exit_code` / `elapsed_sec` / `gpu/peak_mem.json`）、
    ② 清单身份（`tests/e2e/matrix68_manifest.json`）+ 数据件该行（**逐项对齐回填器的身份比对**）、
    ③ 可选的评测读数（`--eval-dir` 的 `eval_report.json`，或 `gen_tier_json.py` 的产物）
    组装成回填器入参契约的字段。

契约（逐字段对照 `tools/matrix/backfill_matrix.py`）：
  顶层必填 `schema/tier/params/run_result/steps_actual/elapsed/peak_mem/throughput/effect/note/run_path/provenance`；
  ⛔ **绝对不写 `status`**（是否通过由回填器按四条判据现算）；
  `run_result{exit_code:int, final_ckpt_loadable:bool, nan_or_inf_detected:bool}`；
  6 个数据列必须"写满"（⛔ 空/`—`/null 会被拒收）；
  `eval{after_acc,base_acc,delta_pp[,teacher_after_acc]}`（判据②）、
  `checks{algo_checks_passed:bool, inference_parseable:bool}`（判据③④）——
  **缺证据就不写这两个键**（⛔ 不用 false/null 冒充"判过了"），回填器会把对应判据判成
  `unknown` 并 fail-closed。
  ★ 判据③④ 的**唯一产出者与渠道**（审查第 2 轮 🟡-A 的落点）：评测跑完后先执行
  `scripts/record_checks.py`，它从 `<EVAL_DIR>/.local/reports/after/eval_report.json` 实测计算
  ④（样本数/请求失败数/逐样本可解析条数/温度 0），并按算法分级产出 ③（SFT/CPT/OPD 结构化置位 +
  依据串；GRASPO 必须有签署的逐条不变式证据）⇒ 写进 `<run-dir>/readings.json` 的
  `checks` 与 `checks_evidence`。本文件只**消费**那份结论（读 `readings.json`），⛔ 不自己判、
  ⛔ 不补 true。缺键时本文件判 rc=4（结构合法但证据不全），回填器保持「待测试」。
  成文渠道与字段格式见 `docs/run-discipline.md` 的「判据③④ 的判定与证据」一节。
  ★ 判据②（**仅 OPD 档**）的教师读数 `eval.teacher_after_acc`：唯一渠道 = `<run-dir>/readings.json`
  的顶层 `teacher_after_acc`（产出者同上，从 `<EVAL_DIR>/.local/reports/teacher/eval_report.json`
  按与学生侧同一口径实测）。本文件只**透传**，⛔ 不自己算、⛔ 缺就不写（OPD 档缺 ⇒ rc=4，
  回填器 `judge_effect` 判 unknown ⇒ fail-closed 保持「待测试」）。见 `docs/run-discipline.md` §20。

落盘前**必经回填器自己的校验器**（动态 import `backfill_matrix.validate_run`）自检：
字段/身份/口径任何一条不过 ⇒ **不宣告成功**，rc 非 0，并在 stdout 逐条列出问题。
缺读数（如没跑评测、日志缺失）⇒ 该字段写 `null` 并在 `note` 里写明**具体缺哪一项、为什么**
（⛔ 不估算、不填充数）；这种 json 会被回填器拒收 —— 这是**设计如此**（fail-closed）。

用法：
    # ① 评测跑完 ⇒ 先产出判据③④ 证据（唯一产出者；写 <run-root>/M001/readings.json）
    python3 scripts/record_checks.py --tier M001 --run-dir <run-root>/M001 \\
        --eval-dir <EV> [--algo-checks-dir <DIR>]
    # ② 再生成回填口径的结果 json（本文件只消费 readings.json 的 checks{}）
    python3 scripts/make_result_json.py --tier M001 --run-dir <run-root>/M001 \\
        --out <run-root>/M001/M001.json [--eval-dir <EV>] [--tier-json <gen_tier_json 产物>] \\
        [--manifest tests/e2e/matrix68_manifest.json] [--data docs/capability-matrix.data.json] \\
        --collect-dir <run-root>/collect
    # ⇒ 缺 ③④ 证据时 rc=4（结果 json 照落盘供审计，但回填器会 fail-closed 保持「待测试」）

★ **回填入口 = `<run-root>/collect`**（扁平 `<档号>.json`）：`--runs-dir` 只吃扁平目录，
  而 runner 的跑次归档在 `<run-root>/<档号>/<档号>.json`（嵌套，回填器不认）。
  ⇒ runner 每档都带 `--collect-dir <run-root>/collect`；回填命令：
        bash tools/matrix/backfill_dated.sh --runs-dir <run-root>/collect
  `--collect-dir` 的落盘时机：结构校验通过后（含"证据不全"的 rc=4 件）才写扁平副本，
  结构不合法（rc=1）不写 —— ⛔ 不让不合契约的件进回填目录。

★ **6 个实验数据列只放"读数/结论"，不放过程说明**：`peak_mem` 列只有 `<数值> GiB`（或
  `null`）；测量方式、证据来源一律写进 `note` 列，且用中性措辞 —— 该列会随回填进入人读件，
  ⛔ 不得出现内部路径、内部工位名、内部行话（人读件的脱敏/禁词由机核现验）。

退出码：0 = 已通过回填器校验器（可回填）；4 = 结构合法但**证据不全**（已落盘、供审计，
        回填器会拒收）；2 = 用法/输入错误；1 = 校验器判不通过（字段/身份口径错）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "matrix-run/v1"

#: 检查点权重文件名（与 `scripts/collect_results.py::FINAL_CKPT_WEIGHT_FILENAMES` 同一张表：
#: 覆盖 native 分片 / HF 分片 / ms-swift peft adapter 三种布局）。
CKPT_WEIGHT_NAMES = (
    "adapter_model.safetensors", "adapter_model.bin",
    "model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
)
CKPT_WEIGHT_GLOBS = ("model-*.safetensors", "rank_*.pt", "*.safetensors")

#: 「实跑步数」的唯一读取口径（与 `scripts/collect_results.py::_GLOBAL_STEP` 同一正则语义）。
GLOBAL_STEP_RE = re.compile(r"['\"]?global_step['\"]?\s*[:=]\s*(\d+)")
GLOBAL_STEP_SLASH_RE = re.compile(r"global_step/max_steps['\"]?\s*[:=]\s*(\d+)")
NANINF_RE = re.compile(r"(?i)['\"]?(loss|grad_norm|reward|logits)['\"]?\s*[:=]\s*"
                       r"(nan|-?inf(?:inity)?)\b")

DATA_COLUMNS = ("peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note")


# ── 小工具 ───────────────────────────────────────────────────────────────────
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:                      # pragma: no cover
        raise SystemExit(f"FATAL: 无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_float_file(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


# ── 证据抽取（全部来自**实测**产物；抽不到 ⇒ None + 原因）───────────────────────
def find_checkpoint_dirs(run_dir: Path) -> list[Path]:
    """发现本档的检查点目录（`checkpoint-*` / `final`），按步号降序。"""
    found = [p for p in run_dir.rglob("checkpoint-*") if p.is_dir()]
    found += [p for p in run_dir.rglob("final") if p.is_dir()]

    def step_of(p: Path) -> int:
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 10 ** 9

    return sorted(set(found), key=step_of, reverse=True)


def ckpt_has_weights(path: Path) -> bool:
    for name in CKPT_WEIGHT_NAMES:
        p = path / name
        if p.is_file() and p.stat().st_size > 0:
            return True
    for pattern in CKPT_WEIGHT_GLOBS:
        for p in path.glob(pattern):
            if p.is_file() and p.stat().st_size > 0:
                return True
    return False


def ckpt_loadable(run_dir: Path) -> tuple[bool | None, str, str | None]:
    """最终检查点是否**完整可加载**（此处口径：找到最终目录 ∧ 有权重文件且非空）。

    ⛔ 这是"形态/完整性"判据，不是"真的 reload 一遍"—— 真 reload 由 `run_guards form` 与
    评测导出阶段承担；本函数只保证**不许在没有任何权重的情况下宣告可加载**。
    """
    dirs = find_checkpoint_dirs(run_dir)
    if not dirs:
        return None, "未找到任何 checkpoint-*/ 或 final/ 目录（证据缺失）", None
    latest = dirs[0]
    if not ckpt_has_weights(latest):
        return False, f"最新检查点 {latest.name} 内没有非空权重文件", str(latest)
    return True, f"最新检查点 {latest.name} 内有非空权重", str(latest)


def parse_steps(stdout_log: Path) -> tuple[int | None, str]:
    """实跑优化步数：日志里 `global_step` 的**最大值**（与台账同一口径）。"""
    if not stdout_log.is_file():
        return None, "训练日志缺失 ⇒ 无法取实跑步数"
    best = None
    text = stdout_log.read_text(encoding="utf-8", errors="replace")
    for m in GLOBAL_STEP_RE.finditer(text):
        v = int(m.group(1))
        best = v if best is None else max(best, v)
    for m in GLOBAL_STEP_SLASH_RE.finditer(text):
        v = int(m.group(1))
        best = v if best is None else max(best, v)
    if best is None:
        return None, "训练日志里没有步数读数（后端未打印/训练未进入优化循环）"
    return best, f"训练日志的最大步数读数 = {best}"


def detect_nan_inf(stdout_log: Path) -> tuple[bool | None, str]:
    if not stdout_log.is_file():
        return None, "训练日志缺失 ⇒ 无法判 NaN/Inf"
    hit = NANINF_RE.search(stdout_log.read_text(encoding="utf-8", errors="replace"))
    if hit:
        return True, f"日志出现异常数值：{hit.group(0)[:60]!r}"
    return False, "日志（loss/grad_norm/reward/logits 字段）未出现 NaN/Inf"


def ledger_record(ledger: Path | None, tier: str) -> dict | None:
    """从 `scripts/collect_results.py` 的台账（`ledger.jsonl`）里取该档记录（若有）。"""
    if ledger is None or not ledger.is_file():
        return None
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("tier_id")) == tier:
            return rec
    return None


def peak_mem_text(run_dir: Path, ledger: Path | None = None, tier: str = "") -> tuple[str | None, str]:
    """峰值显存列。**列值只写数值读数**（`<数值> GiB`；无读数 ⇒ None），⛔ 不写任何解释性文字。

    为什么（修正包 R1-1，审查 🔴-1）：本列会随回填落进人读件的「实验数据」表，机核会扫正文
    —— 解释性文字（测量方式/来源/内部路径）一旦写进本列就会被人读件闸门拒收。
    ⇒ 测量方式与来源作为**第二个返回值**由调用方写进 `note` 列（中性措辞）。

    取值优先级（§1.4 单一真相源）：
    ① `collect_results.py` 台账的 `peak_memory_gib`（该列测量方式由
       `graspo.core.result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND` 按后端唯一决定，
       ⛔ 本文件不另立口径）；
    ② 否则退回 runner 的**宿主侧采样**（`<run-dir>/gpu/peak_mem.json`）。
    """
    rec = ledger_record(ledger, tier) if ledger else None
    if rec:
        gib = rec.get("peak_memory_gib")
        if isinstance(gib, (int, float)):
            return f"{float(gib):.2f} GiB", "矩阵台账的峰值显存读数（测量方式按后端唯一确定）"
    doc = read_json(run_dir / "gpu" / "peak_mem.json", {}) or {}
    used = doc.get("peak_used_mib")
    if used is None:
        return None, "宿主侧显存采样无读数"
    gpus = ",".join(str(g) for g in doc.get("gpus") or [])
    return f"{float(used) / 1024.0:.2f} GiB", f"宿主侧采样的显存占用峰值 = {used} MiB（GPU {gpus}）"


# ── 评测读数（判据②）────────────────────────────────────────────────────────
def eval_block(eval_dir: Path | None, tier_json: Path | None) -> tuple[dict | None, str | None, str]:
    """返回 (eval{}, effect 文本, 说明)。⛔ 两处都取不到 ⇒ (None, None, 原因)。"""
    if eval_dir:
        after = read_json(eval_dir / ".local" / "reports" / "after" / "eval_report.json")
        base = read_json(eval_dir / ".local" / "reports" / "base" / "eval_report.json")
        if isinstance(after, dict) and isinstance(base, dict):
            a = _acc(after)
            b = _acc(base)
            if a is not None and b is not None:
                ev = {"after_acc": round(a, 6), "base_acc": round(b, 6),
                      "delta_pp": round((a - b) * 100, 4)}
                return (ev,
                        (f"after acc={a * 100:.4f}%；base acc={b * 100:.4f}%；"
                         f"Δ = {(a - b) * 100:+.4f} pp"),
                        "取自评测报告（eval_report.json）的 after / base 准确率")
    if tier_json:
        doc = read_json(tier_json, {}) or {}
        a, b = doc.get("after_acc"), doc.get("base_acc")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ev = {"after_acc": float(a), "base_acc": float(b),
                  "delta_pp": round((float(a) - float(b)) * 100, 4)}
            return ev, str(doc.get("effect") or ""), "取自评测汇总产物的 after / base 准确率"
    return None, None, ("本档没有评测读数（未运行评测，或 after / base 两侧评测报告不齐；"
                        "判据② 不可判定，回填器会 fail-closed）")


def _acc(report: dict) -> float | None:
    """取准确率比例。★ 兼容评测产物契约：`eval_report.json` 的聚合结论在 `summary{}` 下
    （`src/graspo/eval/schema.py::EvalSummary`），不是顶层；顶层形态留给 `gen_tier_json` 那类
    扁平产物。⛔ 缺这两种形态之一 ⇒ 返回 None（判据② 由回填器 fail-closed，⛔ 不估算）。"""
    if isinstance(report.get("summary"), dict):
        report = report["summary"]
    for key in ("accuracy", "accuracy_percent", "acc"):
        v = report.get(key)
        if isinstance(v, (int, float)):
            v = float(v)
            return v / 100.0 if key == "accuracy_percent" or v > 1.0 else v
    n_correct, n_total = report.get("n_correct"), report.get("n_total")
    if isinstance(n_correct, int) and isinstance(n_total, int) and n_total > 0:
        return n_correct / n_total
    return None


# ── 组装 ─────────────────────────────────────────────────────────────────────
def build_doc(*, tier: str, run_dir: Path, manifest: dict, data_rows: dict,
              eval_dir: Path | None, tier_json: Path | None, note_extra: str,
              ledger: Path | None = None) -> tuple[dict, dict]:
    """返回 (结果 json, 诊断信息{schema_errors, evidence_gaps})。"""
    mt = next((t for t in manifest["tiers"] if t["tier_id"] == tier), None)
    if mt is None:
        raise SystemExit(f"FATAL: 清单里没有档号 {tier}")
    row = data_rows.get(tier)
    if row is None:
        raise SystemExit(f"FATAL: 数据件里没有档号 {tier}")

    # ── 身份：清单与数据件必须同源（⛔ 不同源就别写 json，先修数据）──────────────
    ident_problems = []
    for mk, dk in (("algorithm", "algo"), ("mode", "mode"), ("backend", "backend"),
                   ("model", "model"), ("lr", "lr"), ("tuner_type", "tuner_type")):
        if str(mt.get(mk)).strip() != str(row.get(dk)).strip():
            try:
                same = abs(float(str(mt.get(mk))) - float(str(row.get(dk)))) < 1e-15
            except (TypeError, ValueError):
                same = False
            if not same:
                ident_problems.append(f"清单 {mk}={mt.get(mk)!r} ≠ 数据件 {dk}={row.get(dk)!r}")
    if ident_problems:
        raise SystemExit("FATAL: 清单与数据件身份不同源 ⇒ ⛔ 不产出结果 json：" + "；".join(ident_problems))

    gaps: list[str] = []
    readings = read_json(run_dir / "readings.json", {}) or {}

    # ── 判据① 的结构化事实 ────────────────────────────────────────────────
    exit_code = readings.get("exit_code")
    if exit_code is None:
        exit_code = read_int_file(run_dir / "exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        gaps.append("缺跑次退出码（跑次目录下无退出码读数）")
        exit_code = None

    steps_actual = readings.get("steps_actual")
    steps_why = "读自跑次读数文件"
    if not isinstance(steps_actual, int) or isinstance(steps_actual, bool):
        steps_actual, steps_why = parse_steps(run_dir / "stdout.log")
        if steps_actual is None:
            gaps.append(f"缺实跑步数（{steps_why}）")

    loadable, loadable_why, ckpt_path = ckpt_loadable(run_dir)
    if loadable is None:
        gaps.append(f"缺最终检查点可加载性结论（{loadable_why}）")

    nan_inf, naninf_why = detect_nan_inf(run_dir / "stdout.log")
    if nan_inf is None:
        gaps.append(f"缺 NaN/Inf 检测结论（{naninf_why}）")

    # ── 三个数值/文本列 ───────────────────────────────────────────────────
    elapsed = readings.get("elapsed_sec")
    if elapsed is None:
        elapsed = read_float_file(run_dir / "elapsed_sec")
    if elapsed is None:
        gaps.append("缺训练墙钟秒数（跑次未落墙钟读数）")
    peak_mem, peak_why = peak_mem_text(run_dir, ledger, tier)
    if peak_mem is None:
        gaps.append(f"缺 peak_mem（{peak_why}）")

    throughput = None
    if isinstance(steps_actual, int) and elapsed:
        throughput = (f"{steps_actual / float(elapsed):.3f} 步/s"
                      f"（{steps_actual} 步 / {float(elapsed):.1f} s；样本吞吐 "
                      f"{mt['data']['subset_size'] * mt['max_epochs'] / float(elapsed):.3f} 样本/s）")
    else:
        gaps.append("缺 throughput（步数与墙钟不齐）")

    ev, ev_effect, ev_why = eval_block(eval_dir, tier_json)
    if ev is None:
        gaps.append(f"effect/判据②：{ev_why}")

    # ── 判据②（OPD）：教师读数 `eval.teacher_after_acc`（★ 唯一渠道 = readings.json）─────────
    #   ★ 产出者 = `scripts/record_checks.py`（从 `<EVAL_DIR>/.local/reports/teacher/eval_report.json`
    #     按与学生侧同一口径实测：同一份 100 条 / 温度 0 / 同一判分函数；见
    #     `docs/run-discipline.md` 的「OPD 教师侧评测」一节）。
    #   ⛔ 缺读数就**不写**这个键（回填器 judge_effect 判 unknown ⇒ fail-closed 保持「待测试」），
    #     绝不填 0、绝不拿学生读数冒充教师读数、绝不因为"非 OPD 档"给默认值。
    algo = str(row.get("algo") or "").upper()
    teacher_acc = readings.get("teacher_after_acc")
    if not isinstance(teacher_acc, (int, float)) or not (0.0 <= float(teacher_acc) <= 1.0):
        teacher_acc = None
    if algo == "OPD":
        if teacher_acc is None:
            gaps.append("判据② 缺证据：OPD 需教师在训练数据集上的完全正确率"
                        "（eval.teacher_after_acc；先跑教师侧评测，见 run-discipline「OPD 教师侧评测」）"
                        " ⇒ 不作通过判定")
        elif ev is not None:
            ev["teacher_after_acc"] = round(float(teacher_acc), 6)

    # ── 判据③④ 的结构化结论（★ 唯一渠道：`<run-dir>/readings.json` 的 `checks`）──────────
    #   ★ 产出者 = `scripts/record_checks.py`（评测跑完后一条命令，从评测报告与算法检查证据现算；
    #     见 `docs/run-discipline.md` 的「判据③④ 的判定与证据」一节）。
    #   ⛔ 缺键就**不写**这两个键（回填器据此判 unknown ⇒ fail-closed 保持「待测试」），
    #     绝不写 false/null 冒充"判过了"，更不硬编 true。
    checks: dict[str, bool] = {}
    checks_ev = readings.get("checks_evidence") if isinstance(readings.get("checks_evidence"), dict) else {}
    raw_checks = readings.get("checks")
    if isinstance(raw_checks, dict):
        checks = {k: raw_checks[k] for k in ("algo_checks_passed", "inference_parseable")
                  if isinstance(raw_checks.get(k), bool)}
    for key, cid, why in (
        ("algo_checks_passed", "判据③", "算法正确性检查缺证据（GRASPO 档须人工/脚本签署逐条不变式）"),
        ("inference_parseable", "判据④", "评测报告缺证据（未跑评测或报告不齐）"),
    ):
        if key not in checks:
            gaps.append(f"{cid} 缺证据：{why} ⇒ 不作通过判定")

    # ── ★ 以下 6 列会随回填进入人读件的「实验数据」表 ⇒ 只放读数/结论，措辞必须中性 ────
    #   （⛔ 内部路径 / 内部工位名 / 内部行话一律不进这 6 列；证据路径留在 provenance 里供审计，
    #    provenance 不进人读件。）
    # ★ `effect` 特例：回填器要求"6 列写满"（空/null 一律拒收），而"本档未跑评测"是一个
    #   **有据的结论**（不是缺数据）⇒ 按既有 `gen_tier_json --no-eval` 的同一写法如实声明
    #   "不可评"，⛔ 不填任何效果数字。数值类列（elapsed/throughput/peak_mem/steps_actual）
    #   缺读数时仍写 null —— 那些是**实测值**，编不出来。
    effect = ev_effect if ev_effect else ("不可评：本档没有评测读数（未运行评测，或 after / base "
                                         "两侧评测报告不齐）")
    note_parts = []
    if note_extra:
        note_parts.append(note_extra)
    note_parts.append(f"退出码 {exit_code}" if exit_code is not None else "退出码缺（未取到跑次退出码）")
    note_parts.append(f"实跑步数 {steps_actual}（{steps_why}；计划 {mt['expected_optimizer_steps']} 步）"
                      if isinstance(steps_actual, int) else
                      f"计划 {mt['expected_optimizer_steps']} 步，实跑未知")
    note_parts.append(f"检查点：{loadable_why}")
    note_parts.append(f"数值异常：{naninf_why}")
    note_parts.append(f"峰值显存：{peak_why}")
    note_parts.append(f"效果读数：{ev_why}")
    note_parts.extend(_teacher_note(algo, ev))
    note_parts.extend(_checks_note(checks, checks_ev, row))
    comparable = readings.get("comparable")
    if comparable is False:
        note_parts.append("环境快照冻结维度发生变化 ⇒ 本档读数不可比（不得悄悄重跑）")
    if gaps:
        note_parts.append("证据缺口（不满足回填条件，不补数）：" + "；".join(gaps))

    doc = {
        "schema": SCHEMA,
        "tier": tier,
        "params": {
            "model": str(row.get("model") or ""),
            "algo": str(row.get("algo") or ""),
            "mode": str(row.get("mode") or ""),
            "backend": str(row.get("backend") or ""),
            "parallel": str(row.get("parallel") or ""),
            "lr": str(row.get("lr") or ""),
            # ★ 身份/语义键一律**逐字取数据件**（回填器按语义归一化后逐项比对；自己拼串会错位）
            "lora_target": str(row.get("lora_target") or ""),
            "data_subset": str(row.get("data_subset") or ""),
            "switches": str(row.get("switches") or ""),
            # 审计用（回填器不比对未知键）：
            "data_subset_detail": f"{mt['data']['subset_size']} 条（{mt['data']['host_relative_path']}）",
            # 审计用（回填器不比对未知键）：
            "tuner_type": mt["tuner_type"],
            "nproc_per_node": mt["parallel_components"]["nproc_per_node"],
            "tp": mt["parallel_components"]["tp"],
            "pp": mt["parallel_components"]["pp"],
            "dp": mt["parallel_components"]["dp"],
            "max_epochs": mt["max_epochs"],
        },
        "run_result": {
            "exit_code": exit_code,
            "final_ckpt_loadable": loadable,
            "nan_or_inf_detected": nan_inf,
        },
        "steps_actual": steps_actual,
        "elapsed": elapsed,
        "peak_mem": peak_mem,
        "throughput": throughput,
        "effect": effect,
        "note": "；".join(note_parts),
        "run_path": str(ckpt_path or run_dir),
        "provenance": {
            "produced_by": "scripts/make_result_json.py（工作包 B / task-manifest）",
            "command": readings.get("command") or f"bash tests/e2e/run_matrix68.sh --only {tier}",
            "evidence_paths": [str(p) for p in _evidence_paths(run_dir, ckpt_path, checks_ev)],
            # 判据③④ 的证据链（有据才落；审计用，不进人读件）
            "checks_evidence": checks_ev,
        },
    }
    if str(row.get("target_modules") or "").strip():
        doc["params"]["target_modules"] = str(row["target_modules"])
    if ev is not None:
        doc["eval"] = ev
    # 判据③④：只有真的有结构化结论时才写（⛔ 不写 false/null 冒充"判过了"）
    if checks:
        doc["checks"] = checks
    return doc, {"evidence_gaps": gaps}


def _teacher_note(algo: str, ev: dict | None) -> list[str]:
    """判据② 的 OPD 教师读数**备注列**文字（中性措辞：⛔ 无内部路径/工位名/过程行话）。

    只写"规则 + 可核数"：教师 = 27B / SFT / LoRA / dp=1 档产物，教师线由回填器按数据件现算
    （⛔ 本文件不写阈值数值）。
    """
    if algo != "OPD":
        return []
    t = (ev or {}).get("teacher_after_acc")
    b = (ev or {}).get("base_acc")
    if not isinstance(t, (int, float)):
        return ["判据②：教师侧读数缺（未跑教师侧评测）⇒ 判据② 不可判定"]
    txt = f"判据②：教师在训练数据集上的完全正确率 {float(t) * 100:.2f}%"
    if isinstance(b, (int, float)):
        txt += f"；教师 − 学生未训练模型 = {(float(t) - float(b)) * 100:+.2f} pp"
    return [txt]


def _checks_note(checks: dict, checks_ev: dict, row: dict) -> list[str]:
    """判据③④ 的**备注列**文字（中性措辞：⛔ 无内部路径/工位名/过程行话）。

    只写"结论 + 可核数"，不写证据路径（路径在 `provenance.checks_evidence` 里供审计）。
    """
    out: list[str] = []
    if checks.get("algo_checks_passed") is True:
        if str(row.get("algo") or "").upper() == "GRASPO":
            out.append("判据③：按《GRASPO 算法说明》第 9 节逐条算法检查，全部不变式通过")
        else:
            out.append(f"判据③：{row.get('algo') or '本算法'} 按报告第 2 章判据③ 的公开标准实现判定通过")
    elif checks.get("algo_checks_passed") is False:
        out.append("判据③：算法正确性检查未通过")
    inf_ev = (checks_ev.get("inference_parseable")
              if isinstance(checks_ev.get("inference_parseable"), dict) else {})
    inf = inf_ev.get("detail") if isinstance(inf_ev.get("detail"), dict) else {}
    if checks.get("inference_parseable") is True:
        out.append(f"判据④：训练数据集 {inf.get('parse_ok')}/{inf.get('expected')} 条推理输出可解析"
                   f"（vLLM 温度 {inf.get('temperature')}）")
    elif checks.get("inference_parseable") is False:
        out.append(f"判据④：推理结果不满足可解析要求"
                   f"（可解析 {inf.get('parse_ok')}/{inf.get('expected')} 条）")
    return out


def _evidence_paths(run_dir: Path, ckpt_path: str | None,
                    checks_ev: dict | None = None) -> list[Path]:
    out = []
    for rel in ("stdout.log", "exit_code", "gpu/peak_mem.json", "tier.json",
                "env-snapshot-before.json", "env-snapshot-after.json", "env-diff.txt"):
        p = run_dir / rel
        if p.exists():
            out.append(p)
    if ckpt_path:
        out.append(Path(ckpt_path))
    # 判据③④ 的证据原件（评测报告 / 算法检查签署件）—— 存在于盘才登记
    for key in ("algo_checks_passed", "inference_parseable", "teacher_after_acc"):
        src = ((checks_ev or {}).get(key) or {}).get("detail", {}).get("source")
        if isinstance(src, str) and Path(src).exists() and Path(src) not in out:
            out.append(Path(src))
    if not out:
        out = [run_dir]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="跑次读数 ⇒ 回填口径结果 json（matrix-run/v1）")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--run-dir", required=True,
                    help="跑次目录（内含 readings.json / stdout.log / gpu/…；判据③④ 结论取自 "
                         "readings.json 的 checks{} —— 由 scripts/record_checks.py 产出）")
    ap.add_argument("--out", default=None, help="默认 <run-dir>/<tier>.json")
    ap.add_argument("--manifest", default=str(ROOT / "tests" / "e2e" / "matrix68_manifest.json"))
    ap.add_argument("--data", default=str(ROOT / "docs" / "capability-matrix.data.json"))
    ap.add_argument("--eval-dir", default=None, help="评测根（含 .local/reports/{after,base}/eval_report.json）")
    ap.add_argument("--tier-json", default=None, help="scripts/gen_tier_json.py 的产物（含 after/base/delta）")
    ap.add_argument("--ledger", default=None,
                    help="collect_results.py 的 ledger.jsonl（提供则峰值显存用矩阵冻结口径 "
                         "peak_memory_gib，否则用宿主侧采样并显式标注口径）")
    ap.add_argument("--note", default="", help="附加备注（写进 note 列）")
    ap.add_argument("--collect-dir", default=None,
                    help="回填入口：额外落一份扁平副本 <dir>/<tier>.json（runner 传 <run-root>/collect；"
                         "回填器 `--runs-dir` 只吃扁平目录、且拒 stray）")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: --run-dir 不存在：{run_dir}")
        return 2
    manifest = read_json(Path(args.manifest))
    if not isinstance(manifest, dict) or not manifest.get("tiers"):
        print(f"FAIL: 清单读不出 tiers：{args.manifest}")
        return 2
    matrix = read_json(Path(args.data))
    if not isinstance(matrix, dict) or not matrix.get("rows"):
        print(f"FAIL: 数据件读不出 rows：{args.data}")
        return 2
    data_rows = {r["tier"]: r for r in matrix["rows"]}

    doc, diag = build_doc(tier=args.tier, run_dir=run_dir, manifest=manifest,
                          data_rows=data_rows, eval_dir=Path(args.eval_dir) if args.eval_dir else None,
                          tier_json=Path(args.tier_json) if args.tier_json else None,
                          note_extra=args.note,
                          ledger=Path(args.ledger) if args.ledger else None)

    out = Path(args.out) if args.out else run_dir / f"{args.tier}.json"
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  bytes={len(text.encode())}")

    # ── 用**回填器自己的校验器**自检（★ 契约不靠本文件自述）────────────────────
    bf = load_module(ROOT / "tools" / "matrix" / "backfill_matrix.py", "_backfill_contract")
    errs = bf.validate_run(doc, args.tier, data_rows)
    if errs:
        print("FAIL: 回填器校验器判不通过（这些字段/身份口径不对）：")
        for e in errs:
            print(f"  - {e}")
        print("  ⇒ 结果 json 已落盘供审计，但**不得**当作可回填件（rc=1）")
        return 1

    if args.collect_dir:
        cdir = Path(args.collect_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{args.tier}.json").write_text(text, encoding="utf-8")
        print(f"collected {cdir / (args.tier + '.json')}（扁平目录，供 backfill_matrix --runs-dir）")

    if diag["evidence_gaps"]:
        print("WARN: 结构合法但证据不全（回填器会在判据层 fail-closed 拒收；⛔ 不补数）:")
        for g in diag["evidence_gaps"]:
            print(f"  - {g}")
        return 4

    print(f"OK: 通过回填器校验器（tier={args.tier} steps_actual={doc['steps_actual']} "
          f"elapsed={doc['elapsed']} peak_mem={str(doc['peak_mem'])[:40]}…）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
