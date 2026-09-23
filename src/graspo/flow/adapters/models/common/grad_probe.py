"""梯度有限性 / 填充度只读探针（纯逻辑，零 GPU 可单测）。

**为什么需要它（2026-09-23 复核 T018 实测证据后定）**：native PP 的 fail-closed 判据此前
**只看"末 stage 的 loss 是否有限"**——``finite = bool(torch.isfinite(chunk_loss)...)`` 只在
``pp_rank == pp_size - 1`` 的 rank 上赋值，其余 rank 恒为 ``True``，再做一次 WORLD ``MIN``
归约。**梯度是否有限完全不判**。逐 rank 原始 ``rank_metrics`` 复核（T018 · 9B · SFT · 全参 ·
native · 4 卡 ``pp=4``）给出的后果：

- ``step3``：rank2 ``grad_norm_mean=nan``，但 ``optimizer_steps=1``、``skipped_nonfinite=0``
  ⇒ **NaN 梯度被照常 ``optimizer.step()`` 写进权重**（``trainable_norm_after=nan``）；
- ``step4``：rank3 ``grad_norm_mean=nan`` 而它的 ``loss_mean=9.8125`` **仍然有限** ⇒ 仍照常 step；
- ``step5``：直到末 stage 的 **loss** 变 NaN 才硬失败——而报错文案写的是"本步梯度含非有限值"。

本模块把"梯度有限性 + 梯度确实被填充"变成**只读读数**与**纯函数判据**（可零 GPU 单测），
使该缺陷在第 3 步就被拦住，并让失败步也能落出逐 rank 诊断行（§2.2 显式即防呆）。

边界（§1.1/§1.3）：
- **只读**：不修改任何梯度、不调用 ``optimizer``、不做集合通信（跨 rank 归约由调用方做）；
- **显存安全**：一切张量归约分块进行，单块临时量有上限（``embed_tokens`` 1.017e9 参数
  一次性 ``isfinite`` 会产生 1 GiB+ 的 bool 临时，而全参档正处在显存边界）；
- **可单测**：判据 ``grad_gate_verdict`` 是纯函数，输入是三个布尔量。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist

#: 单块元素数：与 ``flow/adapters/models/common/base.py::_WEIGHT_NORM_CHUNK_ELEMENTS``
#: **同源同值**（8<<20 ⇒ fp32 临时 32 MiB）——同一类"分块归约保护显存"的取值，避免第二套口径。
GRAD_FINITE_CHUNK_ELEMENTS = 8 << 20

#: 诊断行里最多列出几个非有限梯度的张量名。
#: ★ 2026-09-23（机制探针 #4 裁定）：由 5 提到 **64** —— 裁决"PP 边界先坏 vs 某层先坏"需要
#: 看到**成片的**非有限张量（实测 pp=4 时某一 rank 75/134 个张量非有限），5 个样本看不到形状。
#: 成本可控：该名单只进 **诊断步**（首步探针 / 失败步）的 rank_metrics 行，不进逐步序列。
MAX_NONFINITE_GRAD_NAMES = 64

#: 判据原因码（写进诊断行；人读映射见 :func:`grad_fail_reason_text`）。
GRAD_FAIL_GRAD_NONFINITE = "grad_nonfinite"
GRAD_FAIL_GRAD_UNPOPULATED = "grad_unpopulated"
GRAD_FAIL_LOSS_NONFINITE = "loss_nonfinite"

#: native PP **首步逐 rank 数值探针**的 phase 名（R3，2026-09-23）。
#: **唯一真相源是本常量**；字面量同步登记在 ``graspo.core.result_judge.DIAGNOSTIC_PHASES``
#: （采集侧按文件路径加载该模块、不能 import graspo），由
#: ``tests/flow/trainer/test_grad_fail_closed.py`` 守住。
#: 该行**不带** ``metrics``（纯诊断）⇒ 采集侧不会误当逐步指标（§2.2）。
PP_NUMERIC_PROBE_PHASE = "pp_numeric_probe"

#: native PP **失败步诊断行**的 phase 名（R1，2026-09-23）。
#: 与上面不同：本行**有意带** ``metrics``——失败步的 ``skipped_nonfinite`` /
#: 逐 rank 梯度读数必须能被 A7 与取证链读到（T018 实测缺口正是"崩溃步零行"：
#: raise 早于 metrics 构造）。它**不进** ``STEP_METRICS_PHASES``（不是成功步的逐步指标），
#: 采集侧会按既有规则把它列入 ``ignored_phase_records`` 并写明 phase 名（显式暴露，不静默）。
FAIL_CLOSED_PHASE = "fail_closed"

#: PP **交换张量**的方向标签（唯一真相源；写进读数行，供"边界携带 vs 本地生成"的裁决）。
#: 语义（对任意一个 stage）：
#:   · ``fwd_received_hidden``：本 stage 从**上游**收到的 hidden（= 局部 layer 0 的输入）；
#:   · ``fwd_sent_hidden``：本 stage 发给**下游**的 hidden（末 stage 不发 ⇒ 不记录）；
#:   · ``bwd_received_grad``：本 stage 从**下游**收到的梯度（末 stage 无 ⇒ 不记录）；
#:   · ``bwd_sent_grad``：本 stage 发给**上游**的梯度（= 局部 layer 0 的输入梯度）。
#: 判读：把 ``bwd_received_grad`` 与 ``bwd_sent_grad`` 的 ``isfinite`` 一比即可裁定
#: "**收到时已坏**"(边界携带) 还是 "**本地生成**"（收到有限、发出非有限）。
PP_EXCHANGE_FWD_RECEIVED = "fwd_received_hidden"
PP_EXCHANGE_FWD_SENT = "fwd_sent_hidden"
PP_EXCHANGE_BWD_RECEIVED = "bwd_received_grad"
PP_EXCHANGE_BWD_SENT = "bwd_sent_grad"


#: 层号正则：命中 ``layers.<i>.``（取**本 stage 的局部**层号；PP 下每个 rank 的层号从 0 起）。
_LAYER_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_index_from_key(name: str) -> int | None:
    """从参数名取局部层号；**非层参数**（``embed_tokens``/``lm_head``/``norm``）⇒ ``None``。"""
    match = _LAYER_IN_NAME.search(name)
    return int(match.group(1)) if match else None


def _param_kind(name: str) -> str:
    """参数的"种类"标签：剥掉 ``layers.<i>.`` 前缀（如 ``mlp.down_proj.weight``）；
    非层参数保留完整名（如 ``lm_head.weight``）——直方图按它聚合。"""
    index = _layer_index_from_key(name)
    if index is None:
        return name
    marker = f"layers.{index}."
    return name.split(marker, 1)[1] if marker in name else name


def _first_nonfinite_flat_index(tensor: torch.Tensor) -> int | None:
    """**首个**非有限元素在扁平视图里的下标（分块定位；只在该张量确含非有限时才调用）。"""
    flat = tensor.detach().reshape(-1)
    step = max(1, int(GRAD_FINITE_CHUNK_ELEMENTS))
    for start in range(0, int(flat.numel()), step):
        chunk = flat[start : start + step]
        bad = ~torch.isfinite(chunk)
        if bool(bad.any()):
            local = int(torch.nonzero(bad, as_tuple=False)[0].item())
            return start + local
    return None


def exchange_tensor_report(name: str, tensor: Any, *, direction: str) -> dict[str, Any]:
    """PP **交换张量**的只读读数（**纯本地、无任何集合通信**，不改张量）。

    为什么需要它（机制探针 #1，2026-09-23 裁定）：#4 的直方图只能看到**参数梯度**，
    看不到**跨界交换的那个张量本身** ⇒ 无法区分"收到时已坏（边界携带）"与"本地生成"。
    本函数给出裁定所需的最小字段：方向、形状、dtype、``isfinite``、``max|·|``（NaN 原样透出）、
    以及首个非有限元素的下标（只在非有限时才计算，成本有界）。

    :param name: 张量名（调用点给的稳定标识，如 ``stage_input`` / ``grad_output``）
    :param tensor: 待测张量（``None`` 由 :func:`pp_exchange_readings` 过滤）
    :param direction: :data:`PP_EXCHANGE_FWD_RECEIVED` 等方向标签之一
    """
    shape = [int(dim) for dim in getattr(tensor, "shape", ()) or ()]
    finite = tensor_is_finite(tensor)
    return {
        "direction": str(direction),
        "name": str(name),
        "shape": shape,
        "numel": int(_tensor_numel(tensor)),
        "dtype": str(getattr(tensor, "dtype", "?")),
        "isfinite": bool(finite),
        "max_abs": _tensor_abs_max(tensor),
        "first_nonfinite_index": None if finite else _first_nonfinite_flat_index(tensor),
    }


def pp_exchange_readings(
    items: Iterable[tuple[str, str, Any]],
    *,
    max_items: int = 16,
) -> list[dict[str, Any]]:
    """``(direction, name, tensor)`` 列表 → 读数列表（``None`` 张量跳过；条数有上限）。

    **零集合通信**：只做本地归约；调用方把它塞进既有的诊断行（首步探针 / 失败步），
    健康步**不新增任何行、不新增任何键**。
    """
    out: list[dict[str, Any]] = []
    for direction, name, tensor in items:
        if tensor is None:
            continue
        if len(out) >= int(max_items):
            break
        out.append(exchange_tensor_report(name, tensor, direction=direction))
    return out


def _tensor_numel(tensor: Any) -> int:
    """元素数（真 torch 走 ``numel()``；不可知 ⇒ shape 乘积；再不行 0）。"""
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:  # noqa: BLE001 —— 诊断取数失败 ⇒ 记 0，绝不因"判不了"而 fail
            return 0
    total = 1
    try:
        for dim in getattr(tensor, "shape", ()) or ():
            total *= int(dim)
    except Exception:  # noqa: BLE001 —— 同上：判不了记 0
        return 0
    return total


def _dirty_block_side(dirty: dict[int, int], all_layers: set[int]) -> str | None:
    """脏块在**本 stage 层区间**里的位置（★口径：按**层号**，**不是观测到的时序**）。

    读数语义（§2.2 显式即防呆：把口径写进字段值，避免下游当成"时序"）：
      · ``None``：无脏层；
      · ``all_layers_dirty``：本 stage 所有层都脏；
      · ``adjacent_to_first_local_layer``：脏块含**最小层号**（离上游边界最近那层）但不含最大层号；
      · ``adjacent_to_last_local_layer``：对称的另一端（离下游/loss 侧最近）；
      · ``spans_both_ends``：两端都脏（中间层可能干净）；
      · ``middle``：脏块在层区间内部（两端都干净）。
    为什么用"位置"而非"时序"：真正的时序要靠 autograd hook 才能观测，而**会给被测对象
    加钩子（观察者效应）**——本探针刻意不这么做（§3.4 不倒退）。
    """
    if not dirty:
        return None
    if all_layers and set(dirty) >= set(all_layers):
        return "all_layers_dirty"
    low, high = min(dirty), max(dirty)
    first_dirty = bool(all_layers) and low == min(all_layers)
    last_dirty = bool(all_layers) and high == max(all_layers)
    if first_dirty and last_dirty:
        return "spans_both_ends"
    if first_dirty:
        return "adjacent_to_first_local_layer"
    if last_dirty:
        return "adjacent_to_last_local_layer"
    return "middle"


def tensor_is_finite(
    tensor: torch.Tensor,
    *,
    chunk_elements: int = GRAD_FINITE_CHUNK_ELEMENTS,
) -> bool:
    """分块判断张量是否**全部**为有限值（NaN/±Inf 任一即 False）。

    分块的两条理由：① 单块临时量有上限（显存安全）；② 命中非有限值立即返回（早退）。
    空张量视为有限（"没有非有限元素"在逻辑上为真）。
    """
    flat = tensor.detach().reshape(-1)
    total = int(flat.numel())
    if total == 0:
        return True
    step = max(1, int(chunk_elements))
    for start in range(0, total, step):
        if not bool(torch.isfinite(flat[start : start + step]).all()):
            return False
    return True


def _tensor_abs_max(tensor: torch.Tensor) -> float:
    """分块求 ``max|tensor|``（fp32 累加，单块临时量有上限；NaN 会自然成为最大值）。"""
    flat = tensor.detach().reshape(-1)
    total = int(flat.numel())
    if total == 0:
        return 0.0
    step = max(1, int(GRAD_FINITE_CHUNK_ELEMENTS))
    best: torch.Tensor | None = None
    for start in range(0, total, step):
        chunk_max = flat[start : start + step].abs().max().float()
        best = chunk_max if best is None else torch.maximum(best, chunk_max)
    return float(best.cpu()) if best is not None else 0.0


def grad_finiteness_report(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    with_max_abs: bool = False,
    max_nonfinite_names: int = MAX_NONFINITE_GRAD_NAMES,
) -> dict[str, Any]:
    """逐张量只读探针：梯度填充度 + 非有限梯度定位（含首个张量名与 ``max|g|``）。

    :param named_parameters: ``model.named_parameters()``（``(name, param)`` 可迭代）
    :param with_max_abs: 是否额外计算全局 ``grad_max_abs`` 与 argmax 张量名。
        **默认 False**：那需要对每个梯度再做一遍 ``max|·|`` 归约；每步都算会显著加开销，
        而"是否被单个元素支配"只在**诊断步**（首步探针 / 失败步）才需要。
    :param max_nonfinite_names: 最多记录几个非有限梯度的张量名（前 N 个）。
    :returns: 只读报告 dict（键名即落盘字段名）：``trainable_tensor_count`` /
        ``grad_populated_count`` / ``grad_missing_count`` / ``grad_nonfinite_count`` /
        ``first_nonfinite_grad_names`` / ``grad_max_abs`` / ``grad_argmax_tensor_name`` /
        ``grad_dtype``。
    """
    populated = 0
    trainable = 0
    nonfinite_count = 0
    nonfinite_names: list[dict[str, Any]] = []
    max_abs = 0.0
    max_abs_name: str | None = None
    grad_dtype: str | None = None
    # ── 机制探针 #4（2026-09-23）：按层 / 按种类的**纯本地**直方图（无任何集合通信）──
    layer_counts: dict[int, int] = {}
    kind_counts: dict[str, int] = {}
    nonlayer_nonfinite: list[str] = []
    local_layers: set[int] = set()
    for name, param in named_parameters:
        if not bool(getattr(param, "requires_grad", False)):
            continue
        trainable += 1
        index = _layer_index_from_key(str(name))
        if index is not None:
            local_layers.add(index)
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        populated += 1
        if grad_dtype is None:
            grad_dtype = str(grad.dtype)
        if not tensor_is_finite(grad):
            nonfinite_count += 1
            if index is None:
                if len(nonlayer_nonfinite) < 16:
                    nonlayer_nonfinite.append(str(name))
            else:
                layer_counts[index] = layer_counts.get(index, 0) + 1
            kind = _param_kind(str(name))
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            if len(nonfinite_names) < int(max_nonfinite_names):
                nonfinite_names.append({"name": str(name), "max_abs": _tensor_abs_max(grad)})
        if with_max_abs:
            value = _tensor_abs_max(grad)
            # ★ NaN **支配**语义：含 NaN 的张量其 ``max|g|`` 就是 NaN，而 ``NaN > x`` 恒为
            # False ⇒ 若不显式处理，"全模型最大梯度"会显示成某个健康张量的正常值，
            # 读者会误以为"没有异常"。这里让 NaN 覆盖有限值（且 argmax 指向该张量），
            # 使该字段忠实反映最坏情况（与 ``first_nonfinite_grad_names`` 互补）。
            if math.isnan(value) or value > max_abs:
                max_abs = value
                max_abs_name = str(name)
    return {
        "trainable_tensor_count": trainable,
        "grad_populated_count": populated,
        "grad_missing_count": trainable - populated,
        "grad_nonfinite_count": nonfinite_count,
        "first_nonfinite_grad_names": nonfinite_names,
        "grad_max_abs": max_abs if with_max_abs else None,
        "grad_argmax_tensor_name": max_abs_name,
        "grad_dtype": grad_dtype,
        # ── 机制探针 #4：直方图 + 首/末局部层命中标记 ─────────────────────────────
        # 判读约定（写进字段名，避免下游猜）：
        #  · ``nonfinite_by_layer``：本 stage **局部层号** → 该层非有限张量个数；
        #    PP 下每个 rank 的层号从 0 起 ⇒ "0 层"就是**离上游边界最近**的那层（其输入梯度
        #    正是从 P2P 收到的 stage_input.grad）。
        #  · ``nonfinite_at_first_local_layer``：非有限是否已覆盖**最靠近上游边界**的层；
        #    ``…_at_last_local_layer``：是否覆盖**最靠近下游/loss 侧**的层。
        #    ⇒ (i) 边界先坏 预测前者先亮（尤其非末 stage）；(ii) 某层自身数值问题 预测
        #    非有限集中在**特定 kind**、且不随"离边界远近"分布。
        #  · ``nonfinite_nonlayer_names``：embed_tokens / lm_head / norm 这类**没有层号**的参数
        #    （末 stage 的 lm_head/norm 是 loss 侧通道）——它们若先坏，指向 loss 侧而非边界。
        "nonfinite_by_layer": {str(key): layer_counts[key] for key in sorted(layer_counts)},
        "nonfinite_by_kind": {key: kind_counts[key] for key in sorted(kind_counts)},
        "nonfinite_nonlayer_names": nonlayer_nonfinite,
        "nonfinite_local_layer_span": (
            [min(layer_counts), max(layer_counts)] if layer_counts else None
        ),
        "local_layer_span": ([min(local_layers), max(local_layers)] if local_layers else None),
        "nonfinite_at_first_local_layer": bool(local_layers and min(local_layers) in layer_counts),
        "nonfinite_at_last_local_layer": bool(local_layers and max(local_layers) in layer_counts),
        # ── #2（2026-09-23）：**顺序/方向**信号 ───────────────────────────────────
        # ``nonfinite_layers_descending``：脏层按**层号降序**列出 —— 口径是"**层号的逆序**"，
        #   它对中段 stage 恰好等于 autograd 处理这些层的**自然顺序**（自末层向首层），
        #   但**不是观测到的时序**（观测时序需 autograd hook，会引入观察者效应 ⇒ 刻意不做）。
        # ``dirty_block_side``：脏块相对本 stage 层区间的位置（见 _dirty_block_side 的口径表）。
        #   与 #1 的交换张量读数合用：收到有限 + 发出非有限 ⇒ 本地生成；再按"脏块靠哪端"
        #   判断传播方向（靠首层=朝上游边界扩散 / 靠末层=朝下游扩散）。
        "nonfinite_layers_descending": sorted(layer_counts, reverse=True),
        "dirty_block_side": _dirty_block_side(layer_counts, local_layers),
    }


def reduced_grad_flags(
    *,
    local_report: dict[str, Any],
    distributed: Any = None,
    device: torch.device | None = None,
) -> tuple[bool, bool]:
    """把本 rank 探针的两个命中标志经 WORLD ``MAX`` 归约（**任一 rank 命中即命中**）。

    **为什么收敛成一个函数**（§1.4 单一真相源）：SFT·PP 与 RL·PP 两条路径要做同一件事，
    而"归约语义"（MAX 而非 SUM/MIN）与"无分布式时退化为本地"都是契约；两处各写一遍必然漂移。
    它同时让"跨 rank 判据"可以**零 GPU** 用 gloo 端到端测试（见
    ``tests/flow/trainer/test_grad_fail_closed.py``）。

    :param local_report: :func:`grad_finiteness_report` 的返回值
    :param distributed: ``torch.distributed`` 模块（或同形鸭子类型）；``None`` ⇒ 不归约
    :param device: 标志张量所在设备（默认 CPU；训练时传模型设备）
    :returns: ``(grad_nonfinite_any, grad_unpopulated_any)``
    """
    flags = torch.tensor(
        [
            1 if int(local_report["grad_nonfinite_count"]) > 0 else 0,
            1 if int(local_report["grad_populated_count"]) == 0 else 0,
        ],
        dtype=torch.int,
        device=device if device is not None else torch.device("cpu"),
    )
    if (
        distributed is not None
        and bool(distributed.is_available())
        and bool(distributed.is_initialized())
    ):
        distributed.all_reduce(flags, op=dist.ReduceOp.MAX)
    return bool(int(flags[0].item())), bool(int(flags[1].item()))


def grad_gate_verdict(
    *,
    loss_all_finite: bool,
    grad_nonfinite_any: bool,
    grad_unpopulated_any: bool,
) -> tuple[bool, str]:
    """**纯判据**：本步是否必须 fail-closed，以及原因码（§2.3 边界校验即防呆）。

    规则（按优先级；全部为**硬失败**，不做"跳过并继续"）：

    1. ``grad_nonfinite_any`` —— 任一 rank 有非有限梯度 ⇒ 失败。
       **这是本次新增的核心判据**：NaN/Inf 梯度一旦进入 ``optimizer.step()`` 就会把权重写成
       NaN（T018 step3 实测），而调用方此前只按 loss 判，完全看不到它。
    2. ``grad_unpopulated_any`` 且 loss 全有限 —— 有 rank 一个梯度都没收到 ⇒ 失败。
       该 rank 本步的参数不会有任何更新；"梯度恰好为 0"与"从未收到梯度"的区别由
       ``grad_populated_count`` 给出（前者 = 可训张量总数，后者 = 0）。
       注意：loss 非有限时该 rank 本就不做 backward，此时由规则 3 报出，避免重复归因。
    3. ``not loss_all_finite`` —— 保留既有判据（末 stage loss 非有限 ⇒ 失败），
       但文案不再说成"梯度含非有限值"。

    :returns: ``(failed, reason)``；``failed=False`` 时 ``reason=""``。
    """
    if grad_nonfinite_any:
        return True, GRAD_FAIL_GRAD_NONFINITE
    if grad_unpopulated_any and loss_all_finite:
        return True, GRAD_FAIL_GRAD_UNPOPULATED
    if not loss_all_finite:
        return True, GRAD_FAIL_LOSS_NONFINITE
    return False, ""


def grad_fail_reason_text(reason: str) -> str:
    """原因码 → 人读文案（写进异常消息；**不再把 loss 问题说成梯度问题**）。"""
    if reason == GRAD_FAIL_GRAD_NONFINITE:
        return "逐 rank 梯度含非有限值（NaN/Inf）——若照常 optimizer.step() 会把权重写成 NaN"
    if reason == GRAD_FAIL_GRAD_UNPOPULATED:
        return "有 rank 本步未收到任何梯度（grad_populated_count=0）⇒ 该 stage 参数不会被更新"
    if reason == GRAD_FAIL_LOSS_NONFINITE:
        return "末 stage 的 loss 含非有限值（既有判据；非梯度判据）"
    return f"未知失败原因码：{reason!r}"


def step_index_one_based(call_index: int) -> int:
    """训练批序号落盘的**统一口径**：``1-based``（与 ``global_step`` 一致）。

    **背景（P16，2026-09-23）**：本包最初两处落盘各不相同 —— ``pp_numeric_probe`` 行硬编码
    ``step=0``；``fail_closed`` 行在 SFT 侧是 1-based（``_train_batch_call_index`` 自增**之后**写）
    而在 RL 侧是 0-based（自增在函数**末尾**）。下游若按 ``step`` 横向对行会错位 —— 正是本项目
    反复踩的"同一概念两套口径"。

    **处置**：先**只读核查全部消费方**（``scripts/collect_results.py`` /
    ``src/graspo/cli/tools.py`` / ``src/graspo/core/result_judge.py``）——**没有任何消费方读
    rank_metrics 行的 ``step`` 做对行/join/去重**（collector 的 ``first_logged_step`` 读 ms-swift
    ``trainer_state.log_history[*].step``；cli 读 ``run_cumulative.step``；逐步指标行本身
    **不含** ``step`` 键）⇒ 按裁定**统一到 1-based**，
    收敛到本函数一处（§1.4 单一真相源），并由测试钉住（含"训练模块不得再出现 0-based 字面量"）。

    :param call_index: 0-based 的调用序号（``GraspoFlowTrainer._train_batch_call_index``）
    :returns: 1-based 的批序号（与 ``global_step`` 同口径）
    """
    return int(call_index) + 1
