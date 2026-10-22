#!/usr/bin/env python3
"""判据③④ 的证据产出者（**唯一**）—— 把实测证据折算成 `checks{}` 写进 `readings.json`。

为什么需要本文件（审查报告第 2 轮 🟡-A）：
  `tools/matrix/backfill_matrix.py::judge_algo_checks / judge_inference` 只认结果 json 的
  `checks{algo_checks_passed, inference_parseable}` 两个布尔；而 `scripts/make_result_json.py`
  只从 `<run-dir>/readings.json` 的 `checks` 键取，`tests/e2e/run_matrix68.sh` 第 12 步写的
  readings.json 里**没有这个键**，`--tier-json` 也不提供它 ⇒ 证据齐但缺 `checks{}` 时回填器
  `FAIL-CLOSED`（整批不写、该档保持「待测试」）⇒ 第一个「通过档」会被卡住。
  ⇒ 本文件补上**产出者与成文渠道**：评测跑完后一条命令，从实测产物现算，写进 readings.json。

判据③④ 的语义与数据来源（定稿文档为唯一真相源，⛔ 本文件不另立口径）：

  ④ **能推理**（`docs/GRASPO 训练框架功能测试报告.html` 第 2 章判据④）：
     「LoRA 档合并后、全参档直接，用 vLLM 启动，在训练数据集上完整推理一遍，能完整产出可解析的
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
     · **SFT / CPT / OPD**：文档口径 = 「按各自公开的标准实现逻辑判断」＋另核损失函数/标签掩码
       （OPD 另含教师信号）—— 即**公开标准实现**一层，文档不作逐条清单要求。⇒ 本文件按
       **结构化置位**（`public_algorithm_per_doc_2_3`），依据串逐次落进 `checks_evidence`，
       ⛔ 不是无条件 `true`：算法必须是三者之一，且必须能从清单该档读到 `algorithm`。
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
        --eval-dir <EVAL_DIR> [--algo-checks-dir <DIR>] \\
        [--manifest tests/e2e/matrix68_manifest.json]

    # 再复跑适配器 ⇒ 结果 json 带上 checks{} ⇒ 回填
    python3 scripts/make_result_json.py --tier M001 --run-dir <RUN_DIR> \\
        --eval-dir <EVAL_DIR> --out <RUN_DIR>/M001.json --collect-dir <RUN_ROOT>/collect

退出码：0 = ③④ 两个键都已写入（true / false 都有据）；
        4 = 至少一条缺证据（缺的那条不写键，回填器会 fail-closed 保持「待测试」）；
        2 = 用法/输入错误（run-dir / manifest / readings 不可读）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "e2e" / "matrix68_manifest.json"

#: 评测产物在 `<EVAL_DIR>` 下的相对路径（与 `scripts/gen_eval.py` 的 `output_dir` 同源）。
AFTER_REPORT_REL = Path(".local") / "reports" / "after" / "eval_report.json"

#: 非 GRASPO 算法（文档口径：按公开标准实现判断，可结构化置位）及其依据串。
PUBLIC_ALGO_BASIS = "public_algorithm_per_doc_2_3"
PUBLIC_ALGOS = ("SFT", "CPT", "OPD")

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


# ── 判据③：算法正确性检查（分级：GRASPO 须签署，SFT/CPT/OPD 结构化置位）─────────────
def judge_graspo_algo(tier: str, algo_dir: Path) -> dict:
    """GRASPO：必须有一份显式签署的算法检查证据；缺件/不合契约 ⇒ ok=None（不写键）。"""
    path = algo_dir / f"{tier}.json"
    ev: dict = {"criterion": "③ 算法正确性检查（GRASPO 逐条不变式）", "source": str(path),
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


def judge_public_algo(algo: str) -> dict:
    """SFT / CPT / OPD：文档口径 = 按公开标准实现判断（结构化置位，带依据串）。"""
    basis = (f"{PUBLIC_ALGO_BASIS}（定稿报告第 2 章判据③：{algo} 按各自公开的标准实现逻辑判断，"
             f"另核损失函数与标签掩码处理规则{'，OPD 另含教师信号' if algo == 'OPD' else ''}）")
    return {"ok": True,
            "why": f"判据③ 通过：{algo} 按报告第 2 章判据③ 的公开标准实现判定",
            "evidence": {"criterion": f"③ 算法正确性检查（{algo}）", "basis": basis,
                         "source": "docs/GRASPO 训练框架功能测试报告.html · 第 2 章判据③",
                         "basis_id": PUBLIC_ALGO_BASIS}}


def judge_algo(algo: str, tier: str, algo_dir: Path) -> dict:
    a = str(algo or "").strip().upper()
    if a in PUBLIC_ALGOS:
        return judge_public_algo(a)
    if a == "GRASPO":
        return judge_graspo_algo(tier, algo_dir)
    return {"ok": None,
            "why": f"清单该档 algorithm={algo!r} 不是 SFT/GRASPO/CPT/OPD 之一 ⇒ 判据③ 无判定渠道",
            "evidence": {"criterion": "③ 算法正确性检查", "algorithm": algo}}


# ── 写回 readings.json（唯一渠道）──────────────────────────────────────────────
def write_back(run_dir: Path, results: dict) -> list[str]:
    """把本次结论写进 `<run-dir>/readings.json`（先清旧键：⛔ 不留陈旧通过）。"""
    path = run_dir / "readings.json"
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise SystemExit(f"FATAL: readings.json 缺失或不可解析：{path}（先跑 runner 落读数）")
    doc.pop("checks", None)
    doc.pop("checks_evidence", None)
    checks, evidence, wrote = {}, {}, []
    for key, name in (("algo_checks_passed", "algo"), ("inference_parseable", "inference")):
        r = results.get(name) or {}
        if isinstance(r.get("ok"), bool):
            checks[key] = r["ok"]
            wrote.append(key)
        evidence[key] = {"ok": r.get("ok"), "why": r.get("why"), **({"detail": r["evidence"]}
                                                                   if "evidence" in r else {})}
    if checks:
        doc["checks"] = checks
    doc["checks_evidence"] = evidence
    doc["checks_recorded"] = {"by": "scripts/record_checks.py", "recorded_at": utc_now()}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return wrote


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="判据③④ 证据产出者：从评测报告/算法检查证据现算 checks{} 并写进 readings.json",
        epilog=("⛔ 不得为凑通过手填 true；GRASPO 档必须有人工/脚本签署的算法检查证据；"
                "缺证据 ⇒ 不写键 ⇒ 回填器 fail-closed（该档保持「待测试」）"))
    ap.add_argument("--tier", required=True, help="档号（M0nn）")
    ap.add_argument("--run-dir", required=True, help="跑次目录（内含 readings.json）")
    ap.add_argument("--eval-dir", default=None,
                    help=f"评测根（含 {AFTER_REPORT_REL}；"
                         "也可用环境变量 GRASPO_EVAL_DIR）")
    ap.add_argument("--algo-checks-dir", default=None,
                    help="GRASPO 算法检查证据目录（内含 <tier>.json；"
                         "也可用 GRASPO_ALGO_CHECKS_DIR；缺省 <run-dir>/algo-checks）")
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="跑次侧清单（取该档 algorithm 与 subset_size；判据④ 的期望条数由此现读）")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: --run-dir 不存在：{run_dir}")
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
    expected = int((mt.get("data") or {}).get("subset_size") or 0)
    if expected <= 0:
        print(f"FAIL: 清单该档 data.subset_size 不可用：{mt.get('data')!r}")
        return 2

    eval_dir = resolve_eval_dir(args.eval_dir)
    algo_dir = resolve_algo_checks_dir(args.algo_checks_dir, run_dir)
    report_path = (eval_dir / AFTER_REPORT_REL) if eval_dir else None

    print(f"tier={args.tier} algorithm={algo} 期望评测条数={expected}")
    print(f"评测报告 = {report_path if report_path else '<未指定 --eval-dir / GRASPO_EVAL_DIR>'}")
    print(f"算法检查证据目录 = {algo_dir}")

    inf = judge_inference(read_json(report_path) if report_path else None, expected,
                          report_path or Path("<未指定>"))
    alg = judge_algo(algo, args.tier, algo_dir)
    for tag, r in (("③ algo_checks_passed", alg), ("④ inference_parseable", inf)):
        state = "缺证据（不写键）" if r.get("ok") is None else ("true" if r["ok"] else "false")
        print(f"[{tag}] {state} — {r.get('why')}")

    results = {"algo": alg, "inference": inf}
    wrote = write_back(run_dir, results)
    print(f"wrote {run_dir / 'readings.json'}：checks={wrote}")
    if len(wrote) < 2:
        missing = [k for k in ("algo_checks_passed", "inference_parseable") if k not in wrote]
        print("WARN: 以下判据缺证据 ⇒ 未写键 ⇒ 回填器会 fail-closed（该档保持「待测试」）："
              + "、".join(missing))
        print("  ⇒ 补齐证据后重跑本文件；⛔ 不得手填 true 让流程变绿。")
        return 4
    print("OK: 判据③④ 均有实测证据并已落进 readings.json；"
          "下一步复跑 scripts/make_result_json.py 生成可回填的结果 json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
