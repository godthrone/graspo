"""训练进度指标的键名与模式语义（**纯逻辑**：无 torch、无 graspo 依赖）。

**为什么需要这个模块（它修的是什么缺陷）**

native 的训练事件过去无条件产出 ``lora_norm_before/after/delta`` 与
``nonzero_lora_grad_count``，它们的实现只统计名字含 ``lora_`` 的参数
（``models/common/base.py``）。全参（``tuner_type: full``）模式下**根本不存在**
lora 参数，于是这三个值**结构性恒为 0**——而台账的 A2 判据恰恰是"权重真的变了"，
把恒 0 当成读数就会得出"训练没生效"的错误结论。隐形误导必须消除：

- 全参模式下 ``lora_norm_*`` 显式置 ``None``（宪法 §2.2：None 是唯一合法的"不适用"，
  不用 ``0`` 冒充数值），并**另产出等价的** ``trainable_norm_*``（全部可训参数）；
- 梯度计数同理：键 ``nonzero_grad_count`` 保留（消费者不必改），但附加
  ``grad_count_metric`` 指明它的**定义**——lora 模式是"非零梯度的 LoRA 参数数"，
  全参模式是"已收到梯度的可训参数数"（后者只判 ``grad is not None``，O(1)/参数）。

模块放在 ``flow/`` 顶层且**零依赖**，因此可以在没有 torch 的机器上直接单测
（宪法 §1.3 计算与设施分离）。
"""

from __future__ import annotations

#: 指标的**定义**标识（不是数值），供消费者判断 ``nonzero_grad_count`` 的语义。
LORA_NORM_METRIC = "lora_parameter_l2_norm"
TRAINABLE_NORM_METRIC = "trainable_parameter_l2_norm"
NONZERO_LORA_GRAD_METRIC = "nonzero_lora_grads"
GRAD_POPULATED_METRIC = "grad_populated_trainable_params"


def training_norm_event(tuner_type: str, *, before: float, after: float) -> dict[str, object]:
    """step 前后的权重 L2 范数指标（模式感知，单一真相源）。

    Args:
        tuner_type: ``"lora"`` 或 ``"full"``（``GraspoConfig.effective_tuner_type``）。
        before: step **前**的权重范数（由 ``model.training_progress_norm()`` 给出）。
        after: step **后**的同一范数。

    Returns:
        事件片段。lora 模式键名与数值语义**逐字不变**；full 模式额外给出
        ``trainable_norm_*`` 与 ``norm_metric``，并把 ``lora_norm_*`` 置 ``None``。
    """
    if tuner_type == "full":
        return {
            "tuner_type": "full",
            "norm_metric": TRAINABLE_NORM_METRIC,
            # 全参下没有 lora 参数：不是"数值为 0"，而是"该指标不适用"（None 语义）。
            "lora_norm_before": None,
            "lora_norm_after": None,
            "lora_norm_delta": None,
            "trainable_norm_before": before,
            "trainable_norm_after": after,
            "trainable_norm_delta": after - before,
        }
    return {
        "tuner_type": "lora",
        "norm_metric": LORA_NORM_METRIC,
        "lora_norm_before": before,
        "lora_norm_after": after,
        "lora_norm_delta": after - before,
    }


def grad_count_event(tuner_type: str, *, count: int) -> dict[str, object]:
    """梯度计数指标（模式感知）。

    ``nonzero_grad_count`` 的键名保持不变（消费者无需改），但其**定义**随模式不同：
    用 ``grad_count_metric`` 显式声明，避免跨档比较时把两种语义当成同一个量
    （宪法 §1.4 单一真相源 / §2.2 显式即防呆）。
    """
    return {
        "nonzero_grad_count": int(count),
        "grad_count_metric": (
            GRAD_POPULATED_METRIC if tuner_type == "full" else NONZERO_LORA_GRAD_METRIC
        ),
    }
