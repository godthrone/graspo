#!/usr/bin/env python3
"""A4 档族容差**标定 rig** —— 由 n≥5 读数池逐字产出可粘贴的 `A4TierCalibration(...)` 元组。

## 为什么有这个脚本（§18 消债）

在它之前，"读数池 → 容差 → 标定表条目"这条链**全靠手工转录**：`build_tier_calibration`
/`calibrated_tolerance` 只在单测里被调用，生产入表是把数字手打进 `result_judge.py`。
手工转录的债有三条：① 数字可能抄错（无机器校验）；② 公式的代入过程不可复算；
③ 元数据（代表档 / 卡集 / 采样时间 / 小样本核对）容易被漏。本脚本把这条链闭合：
读数池进 → 推导过程与元组原文出，且**当场过与模块导入同一把尺子**的校验。

## 用法

```bash
python3 scripts/a4_calibrate.py --input pool.json
python3 scripts/a4_calibrate.py --input pool.json --out entry.py
```

`pool.json`（唯一输入，纯数据；不含任何"想要的容差"——防止"看结果调容差"）：

```json
{
  "backend": "ms-swift",
  "cards": 2,
  "algorithm": "GRASPO",
  "model": "27B",
  "mode": "LoRA",
  "representative_tier": "T044",
  "card_set": [4, 5],
  "sampled_at": "228 时钟 2026-09-22 1x:xx–1x:xx",
  "split_check": "留一：max(1.5×worst_pair(n-1)) = …",
  "provenance": "T044 · … 逐跑读数 / 清单 sha / 卡集 / 出处工位路径",
  "run_roots": ["/…/p1", "/…/p2", "/…/p3", "/…/p4", "/…/p5"],
  "readings": [0.1234, 0.1240, 0.1232, 0.1262, 0.1230],
  "judged_pair": {"run_roots": ["/…/r1", "/…/r2"], "delta": 8.1e-5}
}
```

## 三条硬约束（脚本**拒绝**违反的输入）

1. **M1 池与被判对互斥**：`judged_pair.run_roots` 与 `run_roots` 不得有交集
   ——否则"判定的差 ≤ 池内最坏对"由构造恒成立，容差变成自证（循环论证）。
2. **独立性**：`run_roots` 条数必须等于 `readings` 条数，且**逐根唯一**
   （AO1 立下的"5 个各自完整 runner 进程 + 各自 RUN_ROOT"语义）。
3. **同卡集**：`card_set` 必须显式给出（空列表直接拒绝）——AO1 实测跨卡集把方差抬 2.74×。

脚本**不**接受"期望容差"字段；容差一律由 `calibrated_tolerance` 按
`min(1.5 × worst_pair, 硬上界)`、下界兜底推导（公式与判定器同一实现，单一真相源）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import Any

# `scripts/` 不是包（无 `__init__.py`），但本脚本既要能"直接执行"，
# 又要能被 `importlib.util.spec_from_file_location` 按路径加载
# （见 tests/core/test_a4_calibrate.py）。自插脚本目录是同时满足两种
# 加载方式的唯一稳妥写法（§2.2 显式依赖，不靠运行目录碰运气）。
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from errors import CalibrationInputError  # noqa: E402  （须在 sys.path 注入之后）

# ── 纯逻辑模块按文件路径加载，避免经 graspo/__init__ 拉入 torch/pydantic ──────
_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_result_judge() -> ModuleType:
    source = _SRC / "graspo" / "core" / "result_judge.py"
    if source.is_file():
        spec = importlib.util.spec_from_file_location("_graspo_result_judge", source)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            # 必须先登记到 sys.modules：dataclass 装饰器在 exec 时依赖
            # cls.__module__ 能在 sys.modules 中找到对应模块（与 collect_results 同源）。
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    import graspo.core.result_judge as installed  # noqa: PLC0415

    return installed


_judge: Any = _load_result_judge()

#: 池规格里**必须**出现的字段（缺一即拒：§2.1 契约即防呆）。
REQUIRED_POOL_FIELDS: tuple[str, ...] = (
    "backend",
    "cards",
    "algorithm",
    "model",
    "mode",
    "representative_tier",
    "card_set",
    "sampled_at",
    "split_check",
    "provenance",
    "run_roots",
    "readings",
)


def check_independence(pool_roots: list[str], judged_roots: list[str]) -> None:
    """M1：读数池与被判双跑**必须互斥**（否则容差自证，循环论证）。

    为什么这是硬错误而不是警告：若被判的两次运行也在池里，那么"两跑之差 ≤ 池内最坏对"
    由构造恒成立 ⇒ 判据**必然通过**，与训练是否真的可复现无关。这正是本 rig 存在的意义。
    """
    overlap = sorted(set(pool_roots) & set(judged_roots))
    if overlap:
        raise CalibrationInputError(
            f"M1 违反：读数池与被判双跑共用 RUN_ROOT {overlap}。"
            "定容差的池必须与被它判定的那两跑**互斥**，否则容差变成自证（循环论证）。"
        )


def validate_pool(spec: dict[str, Any]) -> None:
    """入表前的边界校验（§2.3）：字段齐全、读数与运行根一一对应、卡集非空、M1 成立。"""
    missing = [name for name in REQUIRED_POOL_FIELDS if name not in spec]
    if missing:
        raise CalibrationInputError(f"池规格缺字段：{missing}（§2.1 契约：拒绝不完整输入）")
    readings = spec["readings"]
    if not isinstance(readings, list) or not readings:
        raise CalibrationInputError("readings 必须是非空列表（终态 loss 读数池）")
    for value in readings:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise CalibrationInputError(f"readings 含非数值：{value!r}")
    run_roots = spec["run_roots"]
    if len(run_roots) != len(readings):
        raise CalibrationInputError(
            f"独立性违反：run_roots={len(run_roots)} 条 ≠ readings={len(readings)} 条"
            "（每个读数必须来自一个独立完整 runner 进程）"
        )
    if len(set(run_roots)) != len(run_roots):
        raise CalibrationInputError(f"独立性违反：run_roots 有重复：{run_roots}")
    card_set = spec["card_set"]
    if not isinstance(card_set, list) or not card_set:
        raise CalibrationInputError(
            "card_set 不得为空（AO1「同一卡集合 mandatory」：跨卡集把方差抬高 ≈2.74×）"
        )
    judged = spec.get("judged_pair") or {}
    if judged:
        check_independence(list(run_roots), list(judged.get("run_roots") or []))
        if "delta" not in judged:
            raise CalibrationInputError("judged_pair 给了 run_roots 就必须给 delta")


def derive(spec: dict[str, Any]) -> tuple[Any, str, str]:
    """由读数池推导标定记录，返回 ``(A4TierCalibration, 人读推导过程, 元组原文)``。"""
    validate_pool(spec)
    readings = [float(value) for value in spec["readings"]]
    tol, outcome = _judge.calibrated_tolerance(readings)
    measured = _judge.worst_pair(readings)
    calibration = _judge.build_tier_calibration(
        readings,
        backend=str(spec["backend"]),
        cards=int(spec["cards"]),
        algorithm=str(spec["algorithm"]),
        provenance=str(spec["provenance"]),
        model=str(spec["model"]),
        mode=str(spec["mode"]),
        representative_tier=str(spec["representative_tier"]),
        card_set=tuple(int(card) for card in spec["card_set"]),
        model_scope=f"{spec['model']}/{spec['mode']}",
        sampled_at=str(spec["sampled_at"]),
        split_check=str(spec["split_check"]),
    )
    hard_cap = _judge.A4_TOLERANCE_HARD_UPPER_BOUND
    base = _judge.A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    factor = _judge.A4_WORST_PAIR_SAFETY_FACTOR
    signature = _judge.A4_MIN_TRUE_BUG_SIGNATURE
    raw = factor * measured
    derivation = [
        f"档族键 = {calibration.backend} × {calibration.cards} 卡 × "
        f"{calibration.algorithm} × {calibration.model} × {calibration.mode}",
        f"n = {calibration.n}（run_roots 逐根唯一 = "
        f"{len(set(spec['run_roots'])) == len(spec['run_roots'])}）",
        f"worst_pair = {measured!r}",
        f"1.5 × worst_pair = {raw!r}",
        f"硬上界 = {hard_cap!r}（= 最小真 bug 签名 {signature!r} / 2）",
        f"基线下界 = {base!r}",
        f"⇒ tol = min(1.5×worst_pair, 硬上界) 再取下界 = {calibration.tol!r}"
        f"（outcome={calibration.outcome}）",
        (
            f"隔离带 = {signature!r} / {calibration.tol!r} = "
            f"{signature / calibration.tol:.3f}×（必须 > 2.0×）"
        ),
        f"note = {calibration.note or '（无）'}",
    ]
    judged = spec.get("judged_pair") or {}
    if judged:
        delta = float(judged["delta"])
        verdict = "通过" if delta <= calibration.tol else "不通过"
        derivation.append(
            f"被判双跑：delta={delta!r} vs tol={calibration.tol!r} ⇒ 终态子检查 {verdict}"
            f"（run_roots={judged.get('run_roots')}，M1 已核：与池互斥）"
        )
    return calibration, "\n".join(derivation), render_entry(calibration)


def _py_str(value: str) -> str:
    """把人读文本渲染成 Python 字符串字面量（保换行、逐字可粘贴）。"""
    return json.dumps(value, ensure_ascii=False)


def _py_str_wrapped(value: str, indent: str, width: int = 96) -> str:
    """把长文本渲染成**隐式拼接的多行字面量**（每行 ≤ width），供 ruff E501 友好入表。

    为什么需要它：`provenance` 逐字写全读数+公式+卡集后会到 800+ 字符；单行字面量会让
    lint 报 E501，而现有标定表全部用隐式拼接。这里按字符切块（文本里不含反斜杠/引号，
    由 :func:`_py_str` 保证转义），拼出的**字符串值逐字相同**。
    """
    literal = _py_str(value)  # 含首尾引号
    if _text_width(literal) + len(indent) <= width:
        return literal
    inner = literal[1:-1]
    budget = max(20, width - len(indent) - 2)
    chunks: list[str] = []
    start = 0
    while start < len(inner):
        # ★ ruff 的 E501 按**显示宽度**计（CJK 字 = 2 列）⇒ 预算必须按显示宽度累加，
        #   否则一段中文 provenance 会在 lint 里"超长"（本仓既有注释行同理）。
        used = 0
        end = start
        while end < len(inner) and used + _display_width(inner[end]) <= budget:
            used += _display_width(inner[end])
            end += 1
        if end == start:  # 单字符就超预算（病态输入）⇒ 至少取一个，保证推进
            end = start + 1
        # 不在转义序列中间切（避免把 `\"` 拆成两半）。
        backslashes = 0
        probe = end - 1
        while probe >= start and inner[probe] == "\\":
            backslashes += 1
            probe -= 1
        if backslashes % 2 == 1 and end < len(inner):
            end += 1
        chunks.append(inner[start:end])
        start = end
    return "\n".join(f'{indent}"{chunk}"' for chunk in chunks)


def _display_width(char: str) -> int:
    """单字符显示宽度：CJK 全角 = 2，其余 = 1（与 ruff E501 的计宽口径一致）。"""
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _text_width(text: str) -> int:
    """整串显示宽度 = 逐字符 :func:`_display_width` 之和。"""
    return sum(_display_width(char) for char in text)


def render_entry(calibration: Any) -> str:
    """把一个标定记录渲染成**可直接粘贴进 `A4_TIER_CALIBRATIONS`** 的元组原文。"""
    lines = [
        "    A4TierCalibration(",
        f"        backend={_py_str(calibration.backend)},",
        f"        cards={calibration.cards},",
        f"        algorithm={_py_str(calibration.algorithm)},",
        f"        tol={calibration.tol!r},",
        f"        worst_pair={calibration.worst_pair!r},",
        f"        n={calibration.n},",
        f"        outcome={_py_str(calibration.outcome)},",
        "        provenance=(",
        _py_str_wrapped(calibration.provenance, "            "),
        "        ),",
        f"        model={_py_str(calibration.model) if calibration.model else 'None'},",
        f"        mode={_py_str(calibration.mode) if calibration.mode else 'None'},",
        f"        representative_tier={_py_str(calibration.representative_tier)},",
        f"        card_set={tuple(calibration.card_set)!r},",
        f"        model_scope={_py_str(calibration.model_scope)},",
        f"        sampled_at={_py_str(calibration.sampled_at)},",
    ]
    # ★ 长字段一律用**隐式拼接**渲染（ruff E501 按显示宽度计，CJK = 2 列）。
    for field, value in (
        ("split_check", calibration.split_check),
        ("note", calibration.note),
    ):
        if not value:
            continue
        wrapped = _py_str_wrapped(value, "            ")
        if "\n" in wrapped:
            lines.append(f"        {field}=(")
            lines.append(wrapped)
            lines.append("        ),")
        else:
            lines.append(f"        {field}={wrapped},")
    lines.append("    ),")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="池规格 JSON 路径")
    parser.add_argument("--out", default=None, help="把元组原文写到该文件（默认只打印）")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    try:
        _calibration, derivation, entry = derive(spec)
    except CalibrationInputError as exc:
        print(f"❌ 池规格非法：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:  # 来自 validate_tier_calibration_entry / validate_tier_tolerance
        print(f"❌ 入表校验拒绝：{exc}", file=sys.stderr)
        return 3

    print("── 推导过程 ──")
    print(derivation)
    print("── 可粘贴元组（逐字）──")
    print(entry)
    if args.out:
        Path(args.out).write_text(entry + "\n", encoding="utf-8")
        print(f"（已写入 {args.out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
