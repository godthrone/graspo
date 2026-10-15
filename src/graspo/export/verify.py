"""分片 / 产物比对的**纯函数**（零 IO、零重依赖、零 graspo 内部 import）。

**职责**：把"两载荷是否逐键一致""产物是否有限"这类判定写成可单测的纯逻辑。
编排在 :class:`graspo.export.base.Exporter`；本模块不产生权重、不读文件。

**为什么不用 ``import torch``**：这些函数只调用传入张量对象自身的**方法**
（``.shape`` / ``.equal`` / ``.isclose`` / ``.isfinite`` / ``.numpy``），因此
模块级与函数体内都不需要 torch —— 既满足 C1/C3，也让它可被按文件路径加载。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _as_cpu_float(tensor: Any) -> Any:
    """把张量搬到 CPU 并转 float32（bfloat16/半精度不支持直接 ``.numpy()``）。"""
    return tensor.detach().cpu().float()


def tensor_digest(tensor: Any) -> str:
    """张量内容的 sha256（逐字节，CPU/float32 归一后计算，仅用于证据与比对）。"""
    return hashlib.sha256(_as_cpu_float(tensor).contiguous().numpy().tobytes()).hexdigest()


def state_digest(state: Mapping[str, Any]) -> str:
    """一个 state-dict 的摘要：键名 + 逐键内容摘要（顺序无关，用于副本比对证据）。"""
    payload = {str(key): tensor_digest(value) for key, value in state.items()}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _max_abs_diff(left: Any, right: Any) -> float:
    try:
        if int(left.numel()) == 0:
            return 0.0
        return float((_as_cpu_float(left) - _as_cpu_float(right)).abs().max().item())
    except Exception:  # noqa: BLE001 —— 差异展示是尽力而为，不掩盖上面的判定
        return float("nan")


def states_identical(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> tuple[bool, str, int]:
    """逐键比对两个 state-dict。

    ``rtol == atol == 0``（默认）⇒ 用 ``Tensor.equal`` 比较（"位模式等价"）；
    DP 副本唯一性判定用它（设计 §0.4/T2：差 1 ulp 即 fail-closed）。传入非零容差时
    改用 ``Tensor.isclose``（TP 复制片比对沿用既有 ``1e-5/1e-6`` 语义）。

    ★ **`Tensor.equal` 的语义边界（如实声明，⛔ 不笼统称"逐 bit 相同"）**：
    ``equal`` 是"位模式等价"，**不是** IEEE 意义下的逐 bit 同一 —— 具体地：

    - ``-0.0`` 与 ``+0.0`` 的位模式不同，但 ``equal`` 判**相等**；
    - ``NaN`` 与 ``NaN``（含同一位模式）``equal`` 判**不相等**。

    ⇒ 对本模块的用途是可接受的：``NaN`` 分歧会被判 fail-closed，即使漏放，
      V14 的 ``non_finite_keys`` 也会在落盘回读时拦下；``-0.0/+0.0`` 数值等价，
      放行无行为差异。若将来需要"真正的逐 bit 同一"（例如跨实现字节级复现），
      须改用位模式比较（``tensor.view(torch.int32)`` / 序列化摘要），不能复用本谓词。

    Returns:
        ``(是否完全一致, 首个分歧的可读描述, 已比对的键数)``。
    """
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))[:3]
        extra = sorted(set(right) - set(left))[:3]
        return False, f"key sets differ (missing={missing}, extra={extra})", 0

    compared = 0
    for key in sorted(left, key=str):
        a, b = left[key], right[key]
        if tuple(a.shape) != tuple(b.shape):
            return False, f"{key}: shape {tuple(a.shape)} != {tuple(b.shape)}", compared
        if rtol == 0.0 and atol == 0.0:
            same = bool(a.equal(b))
        else:
            same = bool(a.isclose(b, rtol=rtol, atol=atol).all())
        if not same:
            return False, f"{key}: values differ (max|delta|={_max_abs_diff(a, b)})", compared
        compared += 1
    return True, "", compared


def non_finite_keys(state: Mapping[str, Any]) -> list[str]:
    """返回含 NaN/Inf 的键（非浮点张量跳过 —— 它们没有非有限值语义）。"""
    bad: list[str] = []
    for key, value in state.items():
        try:
            finite = bool(value.isfinite().all())
        except Exception:  # noqa: BLE001 —— int/bool 张量没有 isfinite
            continue
        if not finite:
            bad.append(str(key))
    return sorted(bad)


def canonical_json(value: Any) -> str:
    """稳定的 JSON 序列化（用于比对非张量元数据，忽略字典顺序）。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


__all__ = [
    "canonical_json",
    "non_finite_keys",
    "state_digest",
    "states_identical",
    "tensor_digest",
]
