#!/usr/bin/env python3
"""判据②③④ 的证据产出者（**唯一**）—— 把实测证据折算成 `checks{}` 与 `teacher_after_acc`
写进 `readings.json`。

★ 判据②（**仅 OPD 档**）的教师读数 `teacher_after_acc`：
  来源 = `<EVAL_DIR>/.local/reports/teacher/eval_report.json`（`scripts/gen_eval.py --role teacher`
  生成脚本跑出来的报告）；口径与学生侧**同一份**（同一份 100 条 / 温度 0 / 同一判分函数），
  本文件另用 `dataset.sha256 ==` 清单该档 `data.subset_sha256` 当场证一遍。
  ⇒ 缺报告 / 不合口径（温度≠0、条数≠100、数据 sha 不同、role≠other、准确率读不到）⇒ **不写该键**
  ⇒ 回填器 `judge_effect` 判 `unknown` ⇒ fail-closed（该档保持「待测试」）。
  ⛔ 不填默认值、不拿学生读数冒充教师读数。成文口径见 `docs/run-discipline.md` §20。

为什么需要本文件（审查报告第 2 轮 🟡-A）：
  `tools/matrix/backfill_matrix.py::judge_algo_checks / judge_inference` 只认结果 json 的
  `checks{algo_checks_passed, inference_parseable}` 两个布尔；而 `scripts/make_result_json.py`
  只从 `<run-dir>/readings.json` 的 `checks` 键取，`tests/e2e/run_matrix68.sh` 第 12 步写的
  readings.json 里**没有这个键**，`--tier-json` 也不提供它 ⇒ 证据齐但缺 `checks{}` 时回填器
  `FAIL-CLOSED`（整批不写、该档保持「待测试」）⇒ 第一个「通过档」会被卡住。
  ⇒ 本文件补上**产出者与成文渠道**：评测跑完后一条命令，从实测产物现算，写进 readings.json。

判据③④ 的语义与数据来源（定稿文档为唯一真相源，⛔ 本文件不另立口径）：

  ④ **能推理**（`docs/GRASPO 训练框架功能测试报告.html` 第 2 章判据④）：
     「LoRA 档合并后、全参档直接，用 vLLM 启动（温度=0），在训练数据集上完整推理一遍，能完整产出可解析的
     推理结果」。⇒ 数据源 = 评测产物 `<EVAL_DIR>/.local/reports/after/eval_report.json`
     （`src/graspo/eval/schema.py` 的 `graspo-eval-report-1` 契约）。逐条判：
       ① 报告可读 ∧ 有逐样本明细 `samples[]`（报告即"最小充分信息"，缺它无从判"可解析"）；
       ② `decoding.temperature == 0.0`（算法说明 §9 第 4 条：评测温度恒 0，入口不提供覆盖参数）；
       ③ `summary.sample_count_total == 清单该档 subset_size`（= 完整推理一遍，本矩阵 100 条）；
       ④ `summary.sample_count_error == 0`（无请求级失败：缺条 = 没跑完整）；
       ⑤ 逐样本「可解析」条数 == 期望条数：无 `error` ∧ 无 `parse_error` ∧ `pred_name` 非空
          （`vllm_client.SampleRecord.to_dict()` 的字段；`parse_error` = 有输出但 arguments 串
          解析不出，`pred_name` 空 = 没产出结构化工具调用）。
     ①②③④ 任一不成立 ⇒ 判 `false`（**有据的不通过**，照实写键）；
     报告缺失 / 不可解析 / 无 samples ⇒ **不写该键**（fail-closed，回填器不判通过）。

  ③ **算法正确性检查通过**（同一章判据③），按算法分级：
     · **SFT / CPT / OPD**：文档口径 = 「按各自公开的标准实现逻辑判断」＋**另按公开标准核对
       损失函数与标签掩码的处理规则**（OPD 另含教师信号的处理规则）—— 后半句是**动作**，
       因此本文件只认「真核过并留痕」的**签署件**
       `samples/configs/matrix68-checks/public-algos.json`（一套规则覆盖该算法的**全部**档，与档号无关；
       可用 `--public-algos` 或 `$GRASPO_PUBLIC_ALGOS` 指别处）。每条规则必须带实现位置 `file:line`。
       缺件 / 方法不匹配 / `verdict != pass` / 缺 `checker` 或 `checked_at` / 缺 loss 或 mask 规则
       （OPD 另缺 teacher_signal 规则）/ 规则里没有代码坐标 ⇒ **不写该键**（fail-closed）。
       ⛔ 不因「公开算法不作逐条清单要求」就置位（审查第 3 轮 🔴-1 的根因正是这个）。
       成文格式见 `docs/run-discipline.md` §19.2。
     · **GRASPO**：⛔ **不许自动置位**。必须有一份**显式签署**的算法检查证据
       `<algo-checks-dir>/<tier>.json`，逐条覆盖《docs/graspo-algorithm.html》**第 9 节**
       的 8 条不变式（本节是"逐条判断"的唯一清单），每条 `{id, pass, evidence}`，并含
       `checker`（检查人）与 `checked_at`（检查时间）。缺件/不合契约 ⇒ **不写该键**；
       8 条里有 `pass=false` ⇒ 写 `false`（有据的不通过）。

⛔ 红线：不得为凑「通过」手填 `true`；不得放宽本文件的判定；不得让回填器在缺③④时判通过。
   本文件只**写实测结论**，`status` 仍由回填器按四条判据现算。

用法：
    # 评测跑完之后（一次即可；重跑幂等：先清旧结论再按本次证据重写，⛔ 不留陈旧通过）
    python3 scripts/record_checks.py --tier M001 --run-dir <RUN_DIR> \\
        --eval-dir <EVAL_DIR> [--algo-checks-dir <DIR>] [--public-algos <FILE>] \\
        [--manifest tests/e2e/matrix68_manifest.json]

    # ⚠ SFT/CPT/OPD 档的判据③ 认的是**公开算法签署件**（缺省
    #   samples/configs/matrix68-checks/public-algos.json）；GRASPO 档认 <algo-checks-dir>/<tier>.json。

    # OPD 档另需教师侧报告（缺省 <EVAL_DIR>/.local/reports/teacher/eval_report.json；
    # 8 个 OPD 档共用一份时用 --teacher-report 指同一份，或 $GRASPO_TEACHER_REPORT）
    python3 scripts/record_checks.py --tier M068 --run-dir <RUN_DIR> \\
        --eval-dir <EVAL_DIR> --teacher-report <教师报告路径>

    # 再复跑适配器 ⇒ 结果 json 带上 checks{} ⇒ 回填
    python3 scripts/make_result_json.py --tier M001 --run-dir <RUN_DIR> \\
        --eval-dir <EVAL_DIR> --out <RUN_DIR>/M001.json --collect-dir <RUN_ROOT>/collect

退出码：0 = 必需键都已写入（非 OPD 档 = ③④；OPD 档 = ③④ + 教师读数）；
        4 = 至少一条缺证据（缺的那条不写键，回填器会 fail-closed 保持「待测试」）；
        2 = 用法/输入错误（run-dir / manifest / readings 不可读）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "e2e" / "matrix68_manifest.json"

#: 评测产物在 `<EVAL_DIR>` 下的相对路径（与 `scripts/gen_eval.py` 的 `output_dir` 同源）。
AFTER_REPORT_REL = Path(".local") / "reports" / "after" / "eval_report.json"

#: ★ OPD **教师侧**评测报告的相对路径 = 判据② 的 `eval.teacher_after_acc` 唯一来源。
#:   ★★ 本常量是**唯一真相源**：`scripts/gen_eval.py` 渲染教师侧 yaml 时直接 import 本值当
#:   `output_dir`（⛔ 不在产出端另写一份路径）⇒ 产出端与消费端不可能错位。
#:   口径 = 定稿报告第 2 章判据②「教师在训练数据集上的完全正确率」+ 第 3 章「OPD 教师」。
TEACHER_REPORT_REL = Path(".local") / "reports" / "teacher" / "eval_report.json"

#: 非 GRASPO 算法（SFT/CPT/OPD）判据③ 的依据标识 = **签署件**（不是"文档说无需逐条"）。
PUBLIC_ALGO_BASIS = "public_algorithm_signed_public_algos_json"
PUBLIC_ALGOS = ("SFT", "CPT", "OPD")

#: ★ 公开算法签署件在仓内的**唯一**位置（与档号无关）。放在 `samples/configs/matrix68-checks/` 的理由：
#:   ① 损失/掩码规则是**算法级**的（一套规则覆盖该算法全部档），不属于某个跑次 ⇒ 不进 `<run-dir>`；
#:   ② ⛔ 不放进 `samples/configs/matrix68/`：该目录被就绪闸门约束为"只放 M###.yaml 档号配置"，
#:      混入非 yaml 会让 `tools/matrix/check_68_readiness.py` 判 FAIL（实测）；
#:   ③ GRASPO 的逐档签署件仍走 `<algo-checks-dir>/<tier>.json`（与档号绑定），两套不混。
PUBLIC_ALGOS_SIG = ROOT / "samples" / "configs" / "matrix68-checks" / "public-algos.json"

#: GRASPO 签署件的依据标识（让 `make_result_json.py` 的备注列能分辨"签署类证据"）。
GRASPO_ALGO_BASIS = "graspo_algorithm_section9_signed"

#: 规则串里必须有实现坐标（`path/to/file.py:123`）：判据③ 要求的是「核对处理规则」，
#: 一句没有代码坐标的结论不可核 ⇒ 缺坐标按"缺规则"处理（不写键）。
RE_CODE_POINTER = re.compile(r"[\w./\\-]+\.py:\d+")

#: GRASPO 判据③ 的逐条清单 =《docs/graspo-algorithm.html》**第 9 节 不变式清单**（逐字对齐，8 条）。
#: ⛔ 不在此另立清单：文档改条数/改措辞时这里必须同步改（本条即"单一真相源"的落地）。
GRASPO_INVARIANTS: tuple[tuple[str, str], ...] = (
    ("group_integrity",
     "组完整性：同一提示的 N 条 rollout 同批同版产出，缺一条或跨版混组不可训练"),
    ("reward_sample_alignment",
     "奖励与样本一一对齐：每条 completion 一奖励/一字符标注/一组 advantage，"
     "advantage 长度等于该条生成 token 数"),
    ("algo_logic_unchanged",
     "算法逻辑严禁改动：标注/奖励/advantage/组决策/裁剪 loss 的算式以算法说明为准"),
    ("temperature_not_overridden",
     "温度互不覆盖：rollout 采样温度取 training.temperature 且不被隐式改写，"
     "评测温度恒 0（评测入口不提供覆盖参数）"),
    ("label_length_invariant",
     "标注长度不变式：字符标记长度 = completion 字符数，token 标记长度 = 生成 token 数"),
    ("sampling_train_same_policy",
     "采样与训练同源：rollout 由当前策略在线采样，旧对数概率即采样时的策略输出"),
    ("ratio_interval_constant",
     "裁剪区间恒定：ratio 有效区间 [1-eps, 1+eps]（eps=0.2），不被隐式改写或逐样本缩放"),
    ("value_error_not_truncated",
     "值错误不截断：值错误只影响相似度打分，只有结构/类型错误才置 E 并标 D；"
     "末尾 EOS 是唯一例外（只把末尾非空白字符置 E，其后的尾随空白仍 W）"),
)
GRASPO_ITEM_IDS = tuple(i for i, _ in GRASPO_INVARIANTS)


# ── 小工具 ───────────────────────────────────────────────────────────────────
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


def resolve_eval_dir(arg: str | None) -> Path | None:
    raw = arg or os.environ.get("GRASPO_EVAL_DIR") or ""
    return Path(raw).expanduser() if raw.strip() else None


def resolve_algo_checks_dir(arg: str | None, run_dir: Path) -> Path:
    raw = (arg or os.environ.get("GRASPO_ALGO_CHECKS_DIR") or "").strip()
    return Path(raw).expanduser() if raw else run_dir / "algo-checks"


def resolve_teacher_report(arg: str | None, eval_dir: Path | None) -> Path | None:
    """教师侧报告路径：显式 `--teacher-report` > `$GRASPO_TEACHER_REPORT` > `<eval-dir>/TEACHER_REPORT_REL`。"""
    raw = (arg or os.environ.get("GRASPO_TEACHER_REPORT") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return (eval_dir / TEACHER_REPORT_REL) if eval_dir else None


def resolve_public_algos_sig(arg: str | None) -> Path:
    """公开算法签署件路径：显式 `--public-algos` > `$GRASPO_PUBLIC_ALGOS` > 仓内唯一位置。"""
    raw = (arg or os.environ.get("GRASPO_PUBLIC_ALGOS") or "").strip()
    return Path(raw).expanduser() if raw else PUBLIC_ALGOS_SIG


def is_opd(algo: str) -> bool:
    """判据② 是否要求教师读数（文档第 2 章判据② 只有 OPD 一条带教师要求）。"""
    return str(algo or "").strip().upper() == "OPD"


# ── 判据④：能推理（自动，判据来自评测报告实测）──────────────────────────────────
def judge_inference(report, expected: int, report_path: Path) -> dict:
    """返回 `{ok: bool|None, why: str, evidence: {...}}`。`ok=None` = 缺证据（不写键）。"""
    ev: dict = {"criterion": "④ 能推理", "source": str(report_path), "expected": expected}
    if not isinstance(report, dict):
        return {"ok": None, "why": f"评测报告缺失或不可解析：{report_path} ⇒ 判据④ 无实测证据",
                "evidence": ev}

    decoding = report.get("decoding") if isinstance(report.get("decoding"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    samples = report.get("samples")

    temp = decoding.get("temperature")
    total = summary.get("sample_count_total")
    err = summary.get("sample_count_error")
    requested = dataset.get("sample_count_requested")
    ev.update({"temperature": temp, "total": total, "error": err,
               "valid": summary.get("sample_count_valid"), "requested": requested,
               "report_schema_version": report.get("schema_version")})

    if not isinstance(samples, list) or not samples:
        return {"ok": None,
                "why": "评测报告没有逐样本明细 samples[]（无逐条输出可核）⇒ 判据④ 无实测证据",
                "evidence": ev}

    parse_ok = 0
    for rec in samples:
        if not isinstance(rec, dict):
            continue
        if rec.get("error") or rec.get("parse_error"):
            continue
        if str(rec.get("pred_name") or "").strip():
            parse_ok += 1
    ev["parse_ok"] = parse_ok
    ev["samples_len"] = len(samples)

    problems = []
    if temp != 0.0:
        problems.append(f"评测温度 decoding.temperature={temp!r} ≠ 0（算法说明 §9 第 4 条）")
    if total != expected:
        problems.append(f"评测样本数 {total!r} ≠ 期望 {expected}（未在训练数据集上完整推理一遍）")
    if total is not None and isinstance(samples, list) and total != len(samples):
        problems.append(f"汇总样本数 {total} ≠ 逐样本明细条数 {len(samples)}（报告自相矛盾）")
    if err != 0:
        problems.append(f"请求级失败 {err!r} 条（缺条/请求失败 ⇒ 未完整产出）")
    if requested is not None and requested != expected:
        problems.append(f"报告登记的请求条数 {requested!r} ≠ 期望 {expected}")
    if parse_ok != expected:
        problems.append(f"可解析输出 {parse_ok}/{expected} 条"
                        "（无 error ∧ 无 parse_error ∧ 有结构化工具名）")

    ev["problems"] = problems
    if problems:
        return {"ok": False, "why": "判据④ 未过：" + "；".join(problems), "evidence": ev}
    return {"ok": True,
            "why": (f"判据④ 通过：训练数据集 {parse_ok}/{expected} 条输出可解析，"
                    f"评测温度 {temp}、请求级失败 0 条"),
            "evidence": ev}


# ── 判据②（OPD）：教师侧读数 `teacher_after_acc` ─────────────────────────────────
def read_accuracy(report: dict) -> float | None:
    """取聚合准确率**比例**（0–1）。★ 与 `scripts/make_result_json.py::_acc` 同一读法：
    `graspo-eval-report-1` 的聚合结论在 `summary{}` 下（`EvalSummary.accuracy`），
    顶层形态只留给 `gen_tier_json` 那类扁平产物。⛔ 缺形态 ⇒ None（不估算）。"""
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


def teacher_tier_of(manifest) -> tuple[str, str] | None:
    """从清单**现算**教师档（口径唯一真相源 = `scripts/gen_eval.py::TEACHER_FORM`）。

    返回 `(档号, 该档 model_path)`；算不出（清单不可用 / 命中 ≠ 1）⇒ `None`（⛔ 不猜、不写死档号）。
    复用 `gen_eval.find_teacher_tier`：两处各写一份口径就会分叉（§1.4）。它用 `die()` 退出进程
    ⇒ 这里捕获 `SystemExit` 转成 None（判据② 会因此 fail-closed，而不是带着猜出来的身份放行）。
    """
    if not isinstance(manifest, dict) or not manifest.get("tiers"):
        return None
    try:
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        import gen_eval  # noqa: PLC0415

        tm = gen_eval.find_teacher_tier(manifest)
    except SystemExit:
        return None
    except Exception:  # noqa: BLE001 - 任何导入/计算失败都按"算不出"处理（⛔ 不猜）
        return None
    tier_id = str(tm.get("tier_id") or "").strip()
    if not tier_id:
        return None
    return tier_id, str(tm.get("model_path") or "").strip()


def judge_teacher(report, expected: int, subset_sha: str | None, report_path: Path,
                  after_report=None, manifest=None) -> dict:
    """OPD 教师侧读数：`teacher_after_acc`（教师在**训练数据集**上的完全正确率，比例）。

    口径（与学生侧同一份，缺一条 ⇒ 不写键 ⇒ 回填器 fail-closed）：
      ① 报告可读 ∧ `model.role == "other"`（教师侧通道的标识，与 after/base 报告区分）；
      ② `decoding.temperature == 0.0`（`EvalDecoding` 强制；教师/学生必须同温）；
      ③ `summary.sample_count_total == 清单该档 data.subset_size` ∧ `sample_count_error == 0`
         （同一份 100 条、无请求失败）；
      ④ `dataset.sha256 == 清单该档 data.subset_sha256`（★ 同一份文件才算同一口径；若学生 after
         报告在，还须与它的 `dataset.sha256` 一致）；
      ⑤ 聚合准确率可读（`summary.accuracy`，逐条 = 1 pp）；
      ⑥ ★ **教师身份**：`checkpoint_path` 必须落在清单现算的教师档（本矩阵 = M041）产物的
         目录下，且 `base_model_path` == 该档 `model_path`（文档第 3 章：教师 = 27B 在
         SFT / LoRA / dp=1 这一档训练完成后的**产物**，不是基座权重本身）。`role=other`
         只证明"这是教师通道产出的报告"，不证明"评的是那个教师"——审查第 3 轮 🟡-D。
    ★ 判分函数本身不是参数：两侧都走 `graspo eval run` → `src/graspo/eval/criteria.py`
      的逐字段完全一致比对，教师与学生同一条码。
    """
    ev: dict = {"criterion": "② 效果（OPD 教师读数 teacher_after_acc）", "source": str(report_path),
                "expected": expected, "expected_subset_sha256": subset_sha}
    if not isinstance(report, dict):
        return {"ok": None,
                "why": (f"缺教师评测报告（{report_path} 不存在或不可解析）⇒ 判据② 的 "
                        "teacher_after_acc 无实测证据；⛔ 不得填默认值、不得跳过"),
                "evidence": ev}

    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    decoding = report.get("decoding") if isinstance(report.get("decoding"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    acc = read_accuracy(report)
    ev.update({"role": model.get("role"), "temperature": decoding.get("temperature"),
               "total": summary.get("sample_count_total"), "error": summary.get("sample_count_error"),
               "correct": summary.get("correct"), "valid": summary.get("sample_count_valid"),
               "dataset_sha256": dataset.get("sha256"), "accuracy": acc,
               "report_schema_version": report.get("schema_version")})

    problems = []
    if str(model.get("role") or "") != "other":
        problems.append(f"报告 role={model.get('role')!r} ≠ 'other'（不是教师侧通道产出的报告）")
    # ★ ⑥ 教师身份（审查第 3 轮 🟡-D）：role=other 只说明"走的是教师通道"，不说明"评的是哪个模型"。
    teacher = teacher_tier_of(manifest)
    ck_path = str(model.get("checkpoint_path") or "").strip()
    base_path = str(model.get("base_model_path") or "").strip()
    ev.update({"checkpoint_path": ck_path or None, "base_model_path": base_path or None,
               "teacher_tier": (teacher[0] if teacher else None),
               "teacher_tier_model_path": (teacher[1] if teacher else None)})
    if teacher is None:
        problems.append("清单算不出教师档（TEACHER_FORM 命中 ≠ 1 或清单不可用）"
                        "⇒ 无法核教师身份（⛔ 不猜档号、不写 teacher_after_acc）")
    else:
        t_tier, t_model = teacher
        if not ck_path or t_tier not in Path(ck_path).parts:
            problems.append(f"教师身份不符：报告 checkpoint_path={ck_path or '<空>'!r} 不在 {t_tier} "
                            f"档产物目录下（判据② 的教师必须是 {t_tier} 档训练完成的产物）")
        if t_model and base_path != t_model:
            problems.append(f"教师身份不符：报告 base_model_path={base_path or '<空>'!r} ≠ 清单 "
                            f"{t_tier} 档 model_path={t_model!r}（不是该教师的底座）")
    if decoding.get("temperature") != 0.0:
        problems.append(f"评测温度 decoding.temperature={decoding.get('temperature')!r} ≠ 0 "
                        "（教师与学生必须同温；算法说明 §9 第 4 条）")
    if summary.get("sample_count_total") != expected:
        problems.append(f"教师评测样本数 {summary.get('sample_count_total')!r} ≠ 期望 {expected}"
                        "（未在与学生同一份训练数据集上完整评测）")
    if summary.get("sample_count_error") != 0:
        problems.append(f"教师评测请求级失败 {summary.get('sample_count_error')!r} 条（读数不可用）")
    sha = str(dataset.get("sha256") or "")
    if not subset_sha:
        problems.append("清单该档缺 data.subset_sha256 ⇒ 无法证明与学生同一份数据（不改用猜测）")
    elif sha != subset_sha:
        problems.append(f"教师评测数据 sha256={sha[:12] or '<空>'}… ≠ 清单该档 subset_sha256="
                        f"{subset_sha[:12]}…（不是同一份数据 ⇒ 与学生不同口径）")
    a_sha = (""
             if not isinstance(after_report, dict)
             else str((after_report.get("dataset") or {}).get("sha256") or ""))
    if a_sha and subset_sha and a_sha != subset_sha:
        problems.append(f"学生 after 报告的数据 sha256={a_sha[:12]}… 与清单该档不一致 ⇒ 学生侧读数本身不可比")
    if acc is None:
        problems.append("报告读不到聚合准确率（summary.accuracy 缺失/非数值）")

    ev["problems"] = problems
    if problems:
        return {"ok": False, "why": "教师侧读数不合口径 ⇒ 不写 teacher_after_acc：" + "；".join(problems),
                "evidence": ev}
    return {"ok": True,
            "why": (f"教师侧评测读数可用：训练数据集 {summary.get('sample_count_total')}/{expected} 条、"
                    f"温度 {decoding.get('temperature')}、请求失败 0、数据 sha256 与学生侧同一份；"
                    f"教师完全正确率 = {acc * 100:.4f}%"),
            "evidence": ev, "acc": round(float(acc), 6)}


# ── 判据③：算法正确性检查（分级：GRASPO / 公开算法都要签署件，清单不同）──────────────
def judge_graspo_algo(tier: str, algo_dir: Path) -> dict:
    """GRASPO：必须有一份显式签署的算法检查证据；缺件/不合契约 ⇒ ok=None（不写键）。"""
    path = algo_dir / f"{tier}.json"
    ev: dict = {"criterion": "③ 算法正确性检查（GRASPO 逐条不变式）", "source": str(path),
                "basis_id": GRASPO_ALGO_BASIS,
                "required_items": list(GRASPO_ITEM_IDS)}
    doc = read_json(path)
    if not isinstance(doc, dict):
        return {"ok": None,
                "why": (f"缺 GRASPO 算法检查证据（{path} 不存在或不是 JSON 对象）"
                        "⇒ 判据③ 无签署证据；⛔ 不得自动置位、不得手填 true"),
                "evidence": ev}

    errs = []
    if str(doc.get("algorithm") or "").upper() != "GRASPO":
        errs.append(f"algorithm={doc.get('algorithm')!r} ≠ 'GRASPO'")
    if str(doc.get("tier") or "") != tier:
        errs.append(f"tier={doc.get('tier')!r} ≠ {tier!r}")
    checker = str(doc.get("checker") or "").strip()
    checked_at = str(doc.get("checked_at") or "").strip()
    if not checker:
        errs.append("缺签名：checker（检查人）为空")
    if not checked_at:
        errs.append("缺签名：checked_at（检查时间）为空")
    items = doc.get("items")
    if not isinstance(items, list):
        errs.append("items 不是数组（须逐条列出第 9 节 8 条不变式）")
        items = []
    got = {}
    for it in items:
        if not isinstance(it, dict):
            errs.append(f"items 含非对象项：{it!r}")
            continue
        iid = str(it.get("id") or "").strip()
        if iid not in GRASPO_ITEM_IDS:
            errs.append(f"未知不变式 id：{iid!r}（只认第 9 节 8 条：{list(GRASPO_ITEM_IDS)}）")
            continue
        if iid in got:
            errs.append(f"不变式 id 重复：{iid!r}")
            continue
        if not isinstance(it.get("pass"), bool):
            errs.append(f"{iid}: pass 必须是布尔值，实际 {it.get('pass')!r}")
        if not str(it.get("evidence") or "").strip():
            errs.append(f"{iid}: evidence 为空（每条须写明判定依据）")
        got[iid] = it
    missing = [i for i in GRASPO_ITEM_IDS if i not in got]
    if missing:
        errs.append(f"未覆盖的不变式 {len(missing)} 条：{missing}")

    ev.update({"checker": checker or None, "checked_at": checked_at or None,
               "items_covered": sorted(got), "samples": {k: v.get("pass") for k, v in got.items()}})
    if errs:
        return {"ok": None, "why": "GRASPO 算法检查证据不合契约 ⇒ 判据③ 不置位：" + "；".join(errs),
                "evidence": ev}
    failing = [i for i in GRASPO_ITEM_IDS if got[i].get("pass") is not True]
    if failing:
        return {"ok": False,
                "why": f"判据③ 未过：GRASPO 算法检查 {len(failing)} 条不变式未通过 {failing}"
                       f"（检查人 {checker}，{checked_at}）",
                "evidence": ev}
    return {"ok": True,
            "why": (f"判据③ 通过：GRASPO《算法说明》第 9 节 {len(GRASPO_ITEM_IDS)} 条不变式逐条检查"
                    f"全部通过（检查人 {checker}，{checked_at}）"),
            "evidence": ev}


def _rule_ok(entry: dict, key: str, label: str) -> str:
    """规则字段是否"可核"：非空 ∧ 带实现坐标 `file:line`。返回空串 = 通过，否则是错误描述。"""
    text = str(entry.get(key) or "").strip()
    if not text:
        return f"缺 {label}（{key} 为空）"
    if not RE_CODE_POINTER.search(text):
        return f"{label}（{key}）没有实现位置 file:line ⇒ 结论不可核"
    return ""


def judge_public_algo(algo: str, sig_doc, sig_path: Path) -> dict:
    """SFT / CPT / OPD：必须有**签署件**（损失/掩码规则的核对留痕）。缺件 ⇒ `ok=None`。

    判据③ 原文（定稿报告第 2 章）＝「按各自公开的标准实现逻辑判断；…**另按公开标准核对损失函数
    与标签掩码的处理规则**（OPD 另含教师信号的处理规则）」。后半句是一个**动作**，所以本函数
    ⛔ 不因"公开算法不作逐条清单要求"就置位，只认签过字的核对结论：
      文件：`samples/configs/matrix68-checks/public-algos.json`（`methods[]` 里 `method` = sft/cpt/opd）
      必需：`verdict == "pass"` ∧ `checker`/`checked_at` 非空 ∧ `loss_rule`/`label_mask_rule`
            非空且带 `file:line`（OPD 另需 `teacher_signal_rule`）∧ `evidence` 非空。
    任一不满足 ⇒ **不写 `algo_checks_passed` 键**（回填器 fail-closed 保持「待测试」）。
    ⚠ `verdict != "pass"`（含 fail）⇒ 同样不写键：签署件说的是"这套规则不成立"，
      那是"不能判"，不是"这一档算法错"（该档要等规则修好再签）。
    """
    method = str(algo or "").strip().lower()
    ev: dict = {"criterion": f"③ 算法正确性检查（{algo} 签署件）", "source": str(sig_path),
                "basis_id": PUBLIC_ALGO_BASIS, "method": method}
    if not isinstance(sig_doc, dict):
        return {"ok": None,
                "why": (f"缺公开算法签署件（{sig_path} 不存在或不是 JSON 对象）⇒ 判据③ 的"
                        "「按公开标准核对损失函数与标签掩码处理规则」无核对留痕；"
                        "⛔ 不得自动置位、不得手填 true"),
                "evidence": ev}

    methods = sig_doc.get("methods")
    if not isinstance(methods, list) or not methods:
        return {"ok": None, "why": f"签署件 {sig_path} 缺 methods[] 数组 ⇒ 不合契约",
                "evidence": ev}
    entry = next((m for m in methods if isinstance(m, dict)
                  and str(m.get("method") or "").strip().lower() == method), None)
    if entry is None:
        return {"ok": None,
                "why": (f"签署件里没有 method={method!r} 的条目"
                        f"（现有 {[str(m.get('method')) for m in methods if isinstance(m, dict)]}）"
                        "⇒ 方法不匹配，判据③ 不置位"),
                "evidence": ev}

    checker = str(entry.get("checker") or "").strip()
    checked_at = str(entry.get("checked_at") or "").strip()
    verdict = str(entry.get("verdict") or "").strip().lower()
    evidence_items = entry.get("evidence")
    ev.update({"checker": checker or None, "checked_at": checked_at or None,
               "verdict": verdict or None,
               "signed_loss_rule": str(entry.get("loss_rule") or ""),
               "signed_label_mask_rule": str(entry.get("label_mask_rule") or ""),
               "signed_teacher_signal_rule": str(entry.get("teacher_signal_rule") or "") or None,
               "evidence_items": (len(evidence_items) if isinstance(evidence_items, list)
                                  else None)})

    errs: list[str] = []
    if verdict != "pass":
        return {"ok": None,
                "why": (f"签署件 verdict={verdict or '<缺失>'!r} ≠ 'pass' ⇒ 判据③ 不置位"
                        "（⛔ 不写 algo_checks_passed 键；规则不成立时该档保持「待测试」）"),
                "evidence": ev}
    if not checker:
        errs.append("缺签名：checker（核对人）为空")
    if not checked_at:
        errs.append("缺签名：checked_at（核对时间）为空")
    for key, label in (("loss_rule", "损失函数处理规则"),
                       ("label_mask_rule", "标签掩码处理规则")):
        bad = _rule_ok(entry, key, label)
        if bad:
            errs.append(bad)
    if method == "opd":
        bad = _rule_ok(entry, "teacher_signal_rule", "教师信号处理规则")
        if bad:
            errs.append(bad)
    if not isinstance(evidence_items, list) or not evidence_items:
        errs.append("缺 evidence（逐条核对项为空）")

    if errs:
        return {"ok": None,
                "why": f"公开算法签署件不合契约 ⇒ 判据③ 不置位：{'；'.join(errs)}",
                "evidence": ev}
    return {"ok": True,
            "why": (f"判据③ 通过：{algo} 损失函数与标签掩码规则已按公开标准逐条核对并签署"
                    f"（签署人 {checker}，{checked_at}"
                    f"{'；含教师信号规则' if method == 'opd' else ''}）"),
            "evidence": ev}


def judge_algo(algo: str, tier: str, algo_dir: Path,
               sig_doc=None, sig_path: Path | None = None) -> dict:
    a = str(algo or "").strip().upper()
    if a in PUBLIC_ALGOS:
        return judge_public_algo(a, sig_doc, sig_path or PUBLIC_ALGOS_SIG)
    if a == "GRASPO":
        return judge_graspo_algo(tier, algo_dir)
    return {"ok": None,
            "why": f"清单该档 algorithm={algo!r} 不是 SFT/GRASPO/CPT/OPD 之一 ⇒ 判据③ 无判定渠道",
            "evidence": {"criterion": "③ 算法正确性检查", "algorithm": algo}}


# ── 写回 readings.json（唯一渠道）──────────────────────────────────────────────
def write_back(run_dir: Path, results: dict) -> list[str]:
    """把本次结论写进 `<run-dir>/readings.json`（先清旧键：⛔ 不留陈旧通过）。

    写两类键：
      · `checks{algo_checks_passed, inference_parseable}`（判据③④，布尔）
      · `teacher_after_acc`（判据② 的 OPD 教师读数，比例；只在**证据合规**时才写）
    两类都进 `checks_evidence`（判定条件 + 实测数 + 证据原件路径，审计用）。
    """
    path = run_dir / "readings.json"
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise SystemExit(f"FATAL: readings.json 缺失或不可解析：{path}（先跑 runner 落读数）")
    doc.pop("checks", None)
    doc.pop("checks_evidence", None)
    doc.pop("teacher_after_acc", None)          # ★ 先清旧读数：⛔ 不留上一次的教师读数
    checks, evidence, wrote = {}, {}, []
    for key, name in (("algo_checks_passed", "algo"), ("inference_parseable", "inference")):
        r = results.get(name) or {}
        if isinstance(r.get("ok"), bool):
            checks[key] = r["ok"]
            wrote.append(key)
        evidence[key] = {"ok": r.get("ok"), "why": r.get("why"), **({"detail": r["evidence"]}
                                                                   if "evidence" in r else {})}
    # 判据② 的教师读数：★ 只有「证据合规且算出数值」才写（缺/不合口径 ⇒ 不写键）
    t = results.get("teacher")
    if isinstance(t, dict):
        acc = t.get("acc")
        if t.get("ok") is True and isinstance(acc, (int, float)):
            doc["teacher_after_acc"] = round(float(acc), 6)
            wrote.append("teacher_after_acc")
        evidence["teacher_after_acc"] = {"ok": t.get("ok"), "why": t.get("why"),
                                         **({"detail": t["evidence"]} if "evidence" in t else {})}
    if checks:
        doc["checks"] = checks
    doc["checks_evidence"] = evidence
    doc["checks_recorded"] = {"by": "scripts/record_checks.py", "recorded_at": utc_now()}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return wrote


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="判据②③④ 证据产出者：从评测报告/算法检查证据现算 check 结论与 OPD 教师读数，"
                    "写进 readings.json",
        epilog=("⛔ 不得为凑通过手填 true / 填默认读数；GRASPO 档必须有人工/脚本签署的算法检查证据；"
                "SFT/CPT/OPD 档必须有公开算法签署件（损失/掩码规则核对留痕）；"
                "OPD 档必须有教师侧评测报告（判据② 的 teacher_after_acc）；"
                "缺证据 ⇒ 不写键 ⇒ 回填器 fail-closed（该档保持「待测试」）"))
    ap.add_argument("--tier", required=True, help="档号（M0nn）")
    ap.add_argument("--run-dir", required=True, help="跑次目录（内含 readings.json）")
    ap.add_argument("--eval-dir", default=None,
                    help=f"评测根（含 {AFTER_REPORT_REL}；"
                         "OPD 档还须含教师侧报告 " + str(TEACHER_REPORT_REL) + "；"
                         "也可用环境变量 GRASPO_EVAL_DIR）")
    ap.add_argument("--teacher-report", default=None,
                    help="教师侧评测报告路径（缺省 <eval-dir>/" + str(TEACHER_REPORT_REL) +
                         "；也可用 GRASPO_TEACHER_REPORT）")
    ap.add_argument("--algo-checks-dir", default=None,
                    help="GRASPO 算法检查证据目录（内含 <tier>.json；"
                         "也可用 GRASPO_ALGO_CHECKS_DIR；缺省 <run-dir>/algo-checks）")
    ap.add_argument("--public-algos", default=None,
                    help="SFT/CPT/OPD 判据③ 的公开算法签署件（损失/掩码规则核对留痕）；"
                         "也可用 GRASPO_PUBLIC_ALGOS；缺省 " + str(PUBLIC_ALGOS_SIG.relative_to(ROOT)))
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="跑次侧清单（取该档 algorithm / subset_size / subset_sha256；"
                         "判据④ 的期望条数与判据② 的同口径校验由此现读）")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: --run-dir 不存在：{run_dir}")
        return 2
    # ── ★ 档号护栏（审查第 3 轮 🔴-2）──────────────────────────────────────────
    #   为什么必须有：68 档是人工分档补跑的场景，`--tier` 与 `--run-dir` 数量多、格式相近
    #   （`M001` / `run/M018`），错配一次就会把**别的算法**的结论写进本档的跑次
    #   （实测：`--tier M001` + M018 的跑次 ⇒ SFT 的 algo_checks_passed 写进 GRASPO 档）。
    #   ⇒ `readings.json.tier` 必须存在且与 `--tier` 逐字相同；⛔ 不自动纠正、不覆盖。
    readings_doc = read_json(run_dir / "readings.json")
    if not isinstance(readings_doc, dict):
        print(f"FAIL: readings.json 缺失或不可解析：{run_dir / 'readings.json'}"
              f"（--tier={args.tier}；先跑 runner 落读数）")
        return 2
    rtier = readings_doc.get("tier")
    if not isinstance(rtier, str) or not rtier.strip():
        print(f"FAIL: readings.json 缺 tier 字段（--tier={args.tier}，readings.tier={rtier!r}）"
              "⇒ 无法证明该跑次目录属于本档；⛔ 不自动纠正、不覆盖")
        return 2
    if rtier.strip() != args.tier.strip():
        print(f"FAIL: 档号错配：readings.json tier={rtier.strip()} ≠ --tier={args.tier.strip()}"
              "（⛔ 不自动纠正/不覆盖；请改用与该跑次一致的 --tier，或换 --run-dir）")
        return 2
    manifest = read_json(Path(args.manifest))
    if not isinstance(manifest, dict) or not manifest.get("tiers"):
        print(f"FAIL: 清单读不出 tiers：{args.manifest}")
        return 2
    mt = next((t for t in manifest["tiers"] if str(t.get("tier_id")) == args.tier), None)
    if mt is None:
        print(f"FAIL: 清单里没有档号 {args.tier}")
        return 2
    algo = str(mt.get("algorithm") or "")
    data = mt.get("data") or {}
    expected = int(data.get("subset_size") or 0)
    subset_sha = str(data.get("subset_sha256") or "")
    if expected <= 0:
        print(f"FAIL: 清单该档 data.subset_size 不可用：{mt.get('data')!r}")
        return 2

    eval_dir = resolve_eval_dir(args.eval_dir)
    algo_dir = resolve_algo_checks_dir(args.algo_checks_dir, run_dir)
    sig_path = resolve_public_algos_sig(args.public_algos)
    report_path = (eval_dir / AFTER_REPORT_REL) if eval_dir else None
    teacher_path = resolve_teacher_report(args.teacher_report, eval_dir)

    print(f"tier={args.tier} algorithm={algo} 期望评测条数={expected}")
    print(f"评测报告 = {report_path if report_path else '<未指定 --eval-dir / GRASPO_EVAL_DIR>'}")
    print(f"算法检查证据目录 = {algo_dir}")
    print(f"公开算法签署件 = {sig_path}"
          f"{'' if str(algo).upper() in PUBLIC_ALGOS else '（本档不是公开算法档 ⇒ 不用它）'}")
    print(f"教师评测报告 = {teacher_path if is_opd(algo) else '<非 OPD 档：判据② 不要求教师读数>'}")

    after_report = read_json(report_path) if report_path else None
    inf = judge_inference(after_report, expected,
                          report_path or Path("<未指定>"))
    sig_doc = read_json(sig_path) if str(algo).upper() in PUBLIC_ALGOS else None
    alg = judge_algo(algo, args.tier, algo_dir, sig_doc, sig_path)
    results = {"algo": alg, "inference": inf}
    for tag, r in (("③ algo_checks_passed", alg), ("④ inference_parseable", inf)):
        state = "缺证据（不写键）" if r.get("ok") is None else ("true" if r["ok"] else "false")
        print(f"[{tag}] {state} — {r.get('why')}")

    required = ["algo_checks_passed", "inference_parseable"]
    if is_opd(algo):
        tch = judge_teacher(read_json(teacher_path) if teacher_path else None, expected,
                            subset_sha or None, teacher_path or Path("<未指定>"), after_report,
                            manifest)
        results["teacher"] = tch
        state = "缺证据（不写键）" if tch.get("ok") is None else (
            f"teacher_after_acc={tch.get('acc')}" if tch["ok"] else "不合口径（不写键）")
        print(f"[② teacher_after_acc] {state} — {tch.get('why')}")
        required.append("teacher_after_acc")

    wrote = write_back(run_dir, results)
    print(f"wrote {run_dir / 'readings.json'}：keys={wrote}")
    if not all(k in wrote for k in required):
        missing = [k for k in required if k not in wrote]
        print("WARN: 以下判据缺证据 ⇒ 未写键 ⇒ 回填器会 fail-closed（该档保持「待测试」）："
              + "、".join(missing))
        print("  ⇒ 补齐证据后重跑本文件；⛔ 不得手填 true / 不得填默认读数让流程变绿。")
        return 4
    print(f"OK: {'、'.join(required)} 均有实测证据并已落进 readings.json；"
          "下一步复跑 scripts/make_result_json.py 生成可回填的结果 json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
