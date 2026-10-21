"""判据② 通过线的**唯一读取通道**（冻结口径 → 评测侧）。

**为什么有这个模块**（宪法 §1.4 单一真相源 ＋ §2 防呆）

判据② 的通过线**只在一处定义**：``docs/capability-matrix.data.json`` 的
``meta.criteria_caliber.threshold.by_method``（逐算法分列，已冻结；该段自带
``by_method_single_source`` 字段，逐字声明「本 by_method 是跑次阈值判定的唯一真相源」）。

评测侧此前另存了一份**旧值的数值副本**（``graspo_threshold_pp=20.0`` /
``sft_threshold_percent=50.0``），且 SFT 那支比的是**绝对准确率**而口径要求比
**Δ** ⇒ 两份数值、两种量纲，结论必错。本模块把数值副本**删干净**：
需要阈值就现读真相源，读不到 / 形状不对即 fail-closed，⛔ 绝不回落到某个"缺省数"。

⛔ 本模块**不定义任何阈值常量**；⛔ 不得在别处新增第二份数值副本。

**冻结口径的数值不在此复述**（⛔ 文档里另存一份数值副本会漂移）——唯一真相源：

``docs/capability-matrix.data.json`` → ``meta.criteria_caliber.threshold.by_method``
（即上文 :data:`CALIBER_SOURCE_RELPATH` ＋ :data:`CALIBER_JSON_POINTER` 两常量所指的落点）。
逐算法的通过线一律由 :func:`load_frozen_method_thresholds` / :func:`frozen_method_threshold`
**现读**，随真相源同步；本模块 ⛔ 不复述任何数值、⛔ 不定义任何阈值常量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 真相源相对**仓库根**的位置（唯一可编辑的源头；人读件由它确定性生成）。
CALIBER_SOURCE_RELPATH = "docs/capability-matrix.data.json"

#: 真相源内的唯一落点（报错时直接指到这里，省得排查者翻文件）。
CALIBER_JSON_POINTER = "meta.criteria_caliber.threshold.by_method"

#: 合法算法名 = ``by_method`` 的键集合（与 ``core.schema.TrainMethod`` 同一集合）。
KNOWN_METHODS: tuple[str, ...] = ("sft", "graspo", "cpt", "opd")

#: 认得的判据类型 → 判定式。两者都归结为「``Δ ≥ 下限``」（点估计）：
#: ``delta_pp`` 的下限 = 该算法的 Δ 通过线；``non_inferiority`` 的下限 = 0.0 pp
#: （即 ``after ≥ base``，真相源就是按 0.0 登记的）。
#: ★ 2026-10-22：Δ 判据的类型名由 `delta_pp` 改为 **`gain_pp`**（数据件不再把内部字段名
#:   当作取值发布）；旧名 `delta_pp` 仍接受，兼容历史数据件/跑次 json。两者判定式相同。
KNOWN_CRITERIA: tuple[str, ...] = ("gain_pp", "delta_pp", "non_inferiority")

#: 唯一支持的比较算子。真相源登记为 ``">="``；出现别的算子即 fail-closed
#: （不猜、不静默放宽）。比较用 ``>=`` 且判定一律**点估计**，不做假设检验。
SUPPORTED_OP = ">="


class FrozenCaliberError(RuntimeError):
    """冻结口径读不到 / 形状不对（fail-closed；⛔ 绝不回落旧值或猜一个数）。"""


@dataclass(frozen=True, slots=True)
class MethodThreshold:
    """某算法的通过线（逐字段取自真相源 ``by_method.<method>``，未做任何改写）。"""

    method: str
    criterion: str
    op: str
    value_pp: float
    unit: str
    pass_line_text: str
    epoch: int | None
    metric: str

    def judge(self, delta_pp: float) -> bool:
        """按 ``criterion`` 判定点估计 ``Δ``（百分点）。

        - ``gain_pp``（旧名 ``delta_pp``）⇒ ``Δ ≥ value_pp``（该算法的 Δ 通过线）；
        - ``non_inferiority`` ⇒ ``Δ ≥ value_pp``（真相源登记 ``value_pp = 0.0``，
          等价于 ``after ≥ base``）。

        Raises:
            FrozenCaliberError: 算子/判据不是本模块认得的那一种（fail-closed）。
        """
        if self.op != SUPPORTED_OP:
            raise FrozenCaliberError(
                f"冻结口径 {CALIBER_JSON_POINTER}.{self.method}.op={self.op!r} 不是 "
                f"{SUPPORTED_OP!r}；本模块不实现别的算子（不静默放宽）"
            )
        if self.criterion not in KNOWN_CRITERIA:
            raise FrozenCaliberError(
                f"冻结口径 {CALIBER_JSON_POINTER}.{self.method}.criterion="
                f"{self.criterion!r} 不认识（只认 {KNOWN_CRITERIA}）"
            )
        return delta_pp >= self.value_pp


def caliber_source_path() -> Path:
    """真相源在**本 checkout** 里的绝对路径（由 ``__file__`` 推导，⛔ 不抄字面量）。"""
    # src/graspo/eval/thresholds.py → parents[3] = 仓库根。
    return Path(__file__).resolve().parents[3] / CALIBER_SOURCE_RELPATH


def _read_caliber_payload() -> dict[str, Any]:
    """读真相源 JSON 并定位到 ``threshold`` 段。任何一步不对 ⇒ fail-closed。"""
    path = caliber_source_path()
    if not path.is_file():
        raise FrozenCaliberError(
            f"冻结口径真相源不存在：{path}（评测侧阈值只能从它读；⛔ 不回落旧值）"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenCaliberError(f"冻结口径真相源不可解析：{path}：{exc}") from None
    node: Any = payload
    for key in CALIBER_JSON_POINTER.split("."):
        if not isinstance(node, dict) or key not in node:
            raise FrozenCaliberError(
                f"冻结口径真相源缺键：{path} 的 {CALIBER_JSON_POINTER}（在 {key!r} 处断）"
            )
        node = node[key]
    if not isinstance(node, dict):
        raise FrozenCaliberError(f"冻结口径 {CALIBER_JSON_POINTER} 不是对象：{type(node).__name__}")
    return node


def _method_threshold(method: str, raw: Any) -> MethodThreshold:
    """把真相源里的一条逐字段搬进 :class:`MethodThreshold`（搬运，不改写）。"""
    where = f"{CALIBER_JSON_POINTER}.{method}"
    if not isinstance(raw, dict):
        raise FrozenCaliberError(f"冻结口径 {where} 不是对象：{type(raw).__name__}")
    missing = [k for k in ("criterion", "op", "value", "pass_line_text") if k not in raw]
    if missing:
        raise FrozenCaliberError(f"冻结口径 {where} 缺字段：{missing}")
    value = raw["value"]
    # 防呆：bool 是 int 的子类，不得被当成阈值（§2.4）。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenCaliberError(f"冻结口径 {where}.value 不是数：{value!r}")
    epoch = raw.get("epoch")
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int)):
        raise FrozenCaliberError(f"冻结口径 {where}.epoch 不是整数：{epoch!r}")
    return MethodThreshold(
        method=method,
        criterion=str(raw["criterion"]),
        op=str(raw["op"]),
        value_pp=float(value),
        unit=str(raw.get("unit", "pp")),
        pass_line_text=str(raw["pass_line_text"]),
        epoch=epoch,
        metric=str(raw.get("metric", "")),
    )


def load_frozen_method_thresholds() -> dict[str, MethodThreshold]:
    """读出四个算法的通过线（每次现读真相源；⛔ 无缓存副本）。

    Raises:
        FrozenCaliberError: 真相源缺失 / 不可解析 / 四算法不全 / 形状不对。
    """
    by_method = _read_caliber_payload()
    return {
        method: _method_threshold(method, by_method.get(method)) for method in KNOWN_METHODS
    }


def frozen_method_threshold(method: str) -> MethodThreshold:
    """取单个算法的通过线。

    Raises:
        FrozenCaliberError: 算法名不认识，或真相源读不到。
    """
    if method not in KNOWN_METHODS:
        raise FrozenCaliberError(
            f"未知算法 {method!r}：冻结口径只认 {KNOWN_METHODS}（判据② 逐算法分列）"
        )
    return load_frozen_method_thresholds()[method]
