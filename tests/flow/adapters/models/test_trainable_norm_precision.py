"""全参训练进度指标的**分辨率**缺陷 —— 可证伪的回归测试。

**它修的是什么（实测，不是推断）**

T034（9B 全参 GRASPO native 1 卡）的 5 个训练步全部记到
``trainable_norm_before == trainable_norm_after == 1796.796176531996``（逐位相同）、
``trainable_norm_delta = 0.0``，台账 A2/③ 据此判「可训练参数未被更新」。

一次只读的逐张量比对（终态 checkpoint ↔ 基座权重）否掉了这个结论：
808 个张量里 **641 个不同**、**905,496,665 个元素（9.62 %）逐元素变化**，
精确 fp64 的全模型 ΔL2 = **−7.32e-06**。权重确实动了，是指标测不出来。

**为什么测不出来**：``_chunked_parameter_norm`` 在 **fp32** 里累加 Σp²。
L2² 量级 3.228e6 的 fp32 ULP 是 **0.25**，而真实的 Δ(Σp²) ≈ 5.25e-03
（= 2·L2·ΔL2）比它还小约 48 倍 ⇒ 块内求和与跨块累加都把变化吃掉，``sqrt`` 逐位相同。
等价地：L2 ≈ 1796.8 处的 fp32 ULP 是 **1.2207e-04**，而真实每步 ΔL2 ≈ 1.46e-06。

**本文件的可证伪性（它的全部价值）**

构造一个"精确 ΔL2 非零、但严格小于 fp32 ULP"的权重变更，然后断言两件事：

1. **旧实现**（``_legacy_fp32_norm``，逐字复刻修复前的代码）**测不出**——逐位相同；
   这一条同时**自证测试没有空转**：如果构造出来的变更旧实现也能测出，
   说明它没落在缺陷区间，本测试就失去意义（会当场失败）。
2. **生产实现**（``_chunked_parameter_norm``）**测得出**——非零，且与 fp64 精确参考一致。

任何人把生产实现改回 fp32 累加，第 2 条立刻失败。
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="权重范数精度测试需要 torch")

from graspo.flow.adapters.models.common.base import (  # noqa: E402
    _WEIGHT_NORM_CHUNK_ELEMENTS,
    _chunked_parameter_norm,
)

#: T034 的 run 自产读数（逐位相同的那个 L2）——用来把合成模型放到**同一个 fp32 阶**
#: （L2 ∈ [1024, 2048) ⇒ fp32 ULP = 1.2207e-04，与真实跑次同档），
#: 而不是随手造一个分辨率不同的玩具。
_T034_REPORTED_L2 = 1796.796176531996

#: 合成"模型"的规模：16 × 1 Mi = 16.78M 个 bf16 元素（≈32 MiB，单测秒级）。
#: 比真机 9.4B 小三个数量级，但 **L2² 落在同一个 fp32 二进制阶**，
#: 因此 fp32 累加的分辨率与真实跑次**同档**——这正是本测试要复现的那把尺子。
_ELEMENTS_PER_TENSOR = 1 << 20
_NORMAL_TENSORS = 15
_ZERO_TENSORS = 1
#: bf16 精确可表示（= 450 × 2⁻¹⁰），且落在 [0.25, 0.5) 这个阶。
_NORMAL_WEIGHT = 0.439453125


def _legacy_fp32_norm(parameters) -> float:
    """**逐字复刻**修复前的实现（fp32 分块累加）——只用于证明测试落在缺陷区间。

    不要把它当成"另一种正确实现"：它就是被证伪的那把尺子。
    """
    total = None
    for param in parameters:
        flat = param.detach().reshape(-1)
        for start in range(0, flat.numel(), _WEIGHT_NORM_CHUNK_ELEMENTS):
            chunk = flat[start : start + _WEIGHT_NORM_CHUNK_ELEMENTS].float()
            partial = chunk.pow(2).sum()
            total = partial if total is None else total + partial
    if total is None:
        return 0.0
    return math.sqrt(float(total.cpu()))


def _exact_fp64_norm(parameters) -> float:
    """参考实现：块内与跨块都在 fp64 里算（判据只用它做交叉校验）。"""
    total = 0.0
    for param in parameters:
        flat = param.detach().reshape(-1)
        for start in range(0, flat.numel(), _WEIGHT_NORM_CHUNK_ELEMENTS):
            chunk = flat[start : start + _WEIGHT_NORM_CHUNK_ELEMENTS]
            total += float(chunk.double().pow(2).sum())
    return math.sqrt(total)


def _fp32_ulp(value: float) -> float:
    """``value`` 在 fp32 下的相邻可表示值间距（本测试的"分辨率"标尺）。"""
    here = torch.tensor(value, dtype=torch.float32)
    nxt = torch.nextafter(here, torch.tensor(float("inf"), dtype=torch.float32))
    return float(nxt) - float(here)


def _t034_shaped_model() -> list[torch.Tensor]:
    """与 T034 同阶的合成"模型"：15 个大权重张量 + 末尾 1 个全零张量。

    为什么留一个全零张量：它让"落在零张量里的单元素值"处在一个 partial == 0 的块，
    于是跨块累加时被总量（≈3.0e6）的 fp32 舍入吸收——这正是真机上发生的事。
    """
    return [
        torch.full((_ELEMENTS_PER_TENSOR,), _NORMAL_WEIGHT, dtype=torch.bfloat16)
        for _ in range(_NORMAL_TENSORS)
    ] + [torch.zeros((_ELEMENTS_PER_TENSOR,), dtype=torch.bfloat16) for _ in range(_ZERO_TENSORS)]


def _with_one_bf16_ulp_change(
    model: list[torch.Tensor], tensor_index: int = 0, element_index: int = 0
) -> tuple[list[torch.Tensor], float]:
    """返回一个**新模型**：把一个元素抬到 bf16 的下一个可表示值，外加该步长。

    单元素一个 bf16 ULP——与真机 bf16 权重上观测到的变化同一量级
    （T034 实测单元素最大变化 3.81e-05，即若干 bf16 ULP）。

    ⚠ 返回的是**整个模型列表**（克隆后改一个元素）。早期版本返回单个张量，
    结果被当成"张量序列"迭代（1-D 张量逐元素迭代）而算出一个完全不同的范数——
    那种测试是坏测试，不是失败测试。
    """
    changed = [tensor.clone() for tensor in model]
    before = changed[tensor_index][element_index].clone()
    after = torch.nextafter(before, torch.tensor(float("inf"), dtype=torch.bfloat16))
    changed[tensor_index][element_index] = after
    return changed, float(after) - float(before)


def test_the_construction_really_lands_in_the_defect_region() -> None:
    """自证：构造出的变更必须**小于 fp32 ULP**，且旧实现确实测不出。

    这一条如果不成立，下面的测试就是空转——所以它先失败。
    """
    model = _t034_shaped_model()
    changed, step = _with_one_bf16_ulp_change(model)

    assert step > 0.0, "bf16 步长必须为正"
    assert _fp32_ulp(_legacy_fp32_norm(model)) == _fp32_ulp(_T034_REPORTED_L2), (
        "合成模型的 L2 必须与 T034 读数落在同一个 fp32 阶（否则分辨率不同档，测试不成立）"
    )

    legacy_before = _legacy_fp32_norm(model)
    legacy_after = _legacy_fp32_norm(changed)
    assert legacy_before == legacy_after, (
        "构造的变更被旧 fp32 实现测出来了 ⇒ 它没落在缺陷区间，本测试失去意义"
    )

    exact_delta = _exact_fp64_norm(changed) - _exact_fp64_norm(model)
    assert exact_delta > 0.0, "精确 fp64 必须看得见这个变更（否则构造本身是零变更）"
    assert exact_delta < _fp32_ulp(legacy_before), (
        f"真实 ΔL2={exact_delta:.6e} 必须小于 fp32 ULP={_fp32_ulp(legacy_before):.6e}"
    )


def test_single_bf16_ulp_weight_change_is_visible_to_the_shipped_norm() -> None:
    """**核心回归**：生产实现必须对"小于 fp32 ULP"的真实变更给出非零 ΔL2。

    把 ``_chunked_parameter_norm`` 改回 fp32 累加，本测试立刻失败。
    """
    model = _t034_shaped_model()
    changed, _ = _with_one_bf16_ulp_change(model)

    before = _chunked_parameter_norm(model)
    after = _chunked_parameter_norm(changed)

    exact_delta = _exact_fp64_norm(changed) - _exact_fp64_norm(model)
    delta = after - before
    assert delta != 0.0, "生产实现没测出单元素 bf16 ULP 变更 ⇒ 精度缺陷复现"
    assert delta == pytest.approx(exact_delta, rel=1e-9, abs=1e-15)
    assert 0.0 < abs(delta) < _fp32_ulp(before), (
        f"ΔL2={delta:.6e} 应严格小于 fp32 ULP={_fp32_ulp(before):.6e}（这正是旧口径盲区）"
    )


def test_tiny_weight_appearing_in_a_zero_block_is_visible_to_the_shipped_norm() -> None:
    """第二个形状：变化落在"partial 从 0 起算"的块里（跨块累加把它吃掉的那种）。

    旧实现同样逐位无感，生产实现必须测得出。
    """
    model = _t034_shaped_model()
    changed = [tensor.clone() for tensor in model]
    changed[-1][0] = 0.001953125  # 2⁻⁹，bf16 精确

    legacy_before = _legacy_fp32_norm(model)
    legacy_after = _legacy_fp32_norm(changed)
    assert legacy_before == legacy_after, "旧 fp32 实现应看不出这个变更"

    before = _chunked_parameter_norm(model)
    after = _chunked_parameter_norm(changed)
    exact_delta = _exact_fp64_norm(changed) - _exact_fp64_norm(model)

    assert after != before, "生产实现没测出零块里的新值 ⇒ 精度缺陷复现"
    assert (after - before) == pytest.approx(exact_delta, rel=1e-9, abs=1e-15)
    assert 0.0 < abs(after - before) < _fp32_ulp(before)


def test_shipped_norm_matches_the_exact_fp64_reference() -> None:
    """正确性（不只是灵敏度）：生产实现必须与 fp64 精确参考一致。"""
    model = _t034_shaped_model()
    assert _chunked_parameter_norm(model) == pytest.approx(
        _exact_fp64_norm(model), rel=1e-15, abs=1e-12
    )


def test_norm_is_empty_safe_and_non_negative() -> None:
    """边界：空参数序列返回 0.0；范数非负（§2.3 边界校验）。"""
    assert _chunked_parameter_norm([]) == 0.0
    assert _chunked_parameter_norm([torch.zeros(4, dtype=torch.bfloat16)]) == 0.0
    assert _chunked_parameter_norm(_t034_shaped_model()) > 0.0
