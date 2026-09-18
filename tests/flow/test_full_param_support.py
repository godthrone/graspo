"""全参支持的**纯逻辑**单测：训练进度指标语义 + checkpoint 导出守卫。

覆盖两个缺陷：
1. 全参模式下 ``lora_norm_*`` **结构性恒为 0**（只统计 ``lora_`` 参数）——继续产出
   会被误读成"权重没变"，而台账 A2 判据恰恰是"权重真的变了"。
2. 全参 checkpoint 走 LoRA 导出通道时，``merged-hf`` 会算出零 delta 并**静默写出
   未训练的基座副本**（"导出成功、权重没学到"）。

**不依赖 torch**：被测的两个模块（``graspo.flow.progress_metrics`` /
``graspo.flow.checkpoint_semantics``）**零依赖**，因此可以按文件路径直接加载，
连 import 垫片都不需要。

**为什么按文件加载而不是普通 import**：普通 import 会执行 `graspo/__init__.py` →
`graspo.flow.__init__`（import runtime/trainer ⇒ torch），本机没有 torch。若为此
注册 `sys.modules["graspo.flow"]` 替身包，会**改变同目录其它测试文件的收集行为**
（实测：`tests/flow` 从 "4 collected / 20 errors" 变成 "72 collected / 16 errors"，
多出的失败全部来自本工作包之外的用例）。按文件加载是**零全局副作用**的等价方式
（宪法：不扩大范围）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_pure_module(dotted_name: str, relative_path: str):
    """从文件加载一个"零依赖"模块（不执行父包 ``__init__``、不写任何全局状态）。

    Args:
        dotted_name: 仅用于模块名标识（模块是**叶子**：无任何 import 依赖，
            因此无需预先注册父包，也不注册进 ``sys.modules``）。
        relative_path: 相对 ``src/`` 的路径。

    Returns:
        已执行的模块对象。
    """
    path = Path(__file__).resolve().parents[2] / "src" / relative_path
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_checkpoint_semantics = _load_pure_module(
    "graspo.flow.checkpoint_semantics", "graspo/flow/checkpoint_semantics.py"
)
_progress_metrics = _load_pure_module(
    "graspo.flow.progress_metrics", "graspo/flow/progress_metrics.py"
)

import pytest  # noqa: E402

is_full_param_payload = _checkpoint_semantics.is_full_param_payload
require_lora_semantics = _checkpoint_semantics.require_lora_semantics
GRAD_POPULATED_METRIC = _progress_metrics.GRAD_POPULATED_METRIC
LORA_NORM_METRIC = _progress_metrics.LORA_NORM_METRIC
NONZERO_LORA_GRAD_METRIC = _progress_metrics.NONZERO_LORA_GRAD_METRIC
TRAINABLE_NORM_METRIC = _progress_metrics.TRAINABLE_NORM_METRIC
grad_count_event = _progress_metrics.grad_count_event
training_norm_event = _progress_metrics.training_norm_event


# ── 训练进度指标：LoRA 路径逐字不变 ─────────────────────────────────────────


def test_lora_mode_keeps_the_original_keys_and_values():
    event = training_norm_event("lora", before=1.5, after=2.0)

    assert event["lora_norm_before"] == 1.5
    assert event["lora_norm_after"] == 2.0
    assert event["lora_norm_delta"] == pytest.approx(0.5)
    assert event["norm_metric"] == LORA_NORM_METRIC
    assert event["tuner_type"] == "lora"
    # 全参专属键在 lora 模式下**不出现**（不制造"看起来是 0"的假读数）
    assert "trainable_norm_before" not in event


def test_lora_mode_grad_count_declares_its_definition():
    event = grad_count_event("lora", count=7)

    assert event["nonzero_grad_count"] == 7
    assert event["grad_count_metric"] == NONZERO_LORA_GRAD_METRIC


# ── 训练进度指标：全参模式不再产出误导性的恒 0 ──────────────────────────────


def test_full_mode_marks_lora_norm_as_not_applicable():
    """``lora_norm_*`` 必须是 ``None``（不适用），不能是 0.0（会被读成"没变"）。"""
    event = training_norm_event("full", before=100.0, after=101.0)

    assert event["lora_norm_before"] is None
    assert event["lora_norm_after"] is None
    assert event["lora_norm_delta"] is None


def test_full_mode_provides_the_equivalent_trainable_norm():
    event = training_norm_event("full", before=100.0, after=101.0)

    assert event["norm_metric"] == TRAINABLE_NORM_METRIC
    assert event["tuner_type"] == "full"
    assert event["trainable_norm_before"] == 100.0
    assert event["trainable_norm_after"] == 101.0
    assert event["trainable_norm_delta"] == pytest.approx(1.0)


def test_full_mode_grad_count_declares_its_definition():
    event = grad_count_event("full", count=707)

    assert event["nonzero_grad_count"] == 707
    assert event["grad_count_metric"] == GRAD_POPULATED_METRIC


# ── checkpoint 导出守卫 ─────────────────────────────────────────────────────


def test_lora_payload_is_not_mistaken_for_full_param():
    lora_payload = {
        "tuner_type": "lora",
        "lora_state_dict": {"x.lora_a": object()},
        "full_param_state_dict": None,
    }
    assert is_full_param_payload(lora_payload) is False
    require_lora_semantics([lora_payload], operation="merged-hf")  # 不抛


def test_legacy_lora_payload_without_the_new_keys_passes():
    """旧 checkpoint 没有 tuner_type / full_param_state_dict 两个键——不得误判。"""
    require_lora_semantics([{"lora_state_dict": {}}], operation="merged-hf")


@pytest.mark.parametrize(
    "payload",
    [
        {"tuner_type": "full", "lora_state_dict": {}, "full_param_state_dict": {}},
        # 只有权重载荷、没有 tuner_type（异构/旧格式 shard）也要挡住
        {"lora_state_dict": {}, "full_param_state_dict": {"a": 1}},
    ],
)
def test_full_param_payload_is_rejected_with_actionable_message(payload):
    assert is_full_param_payload(payload) is True
    with pytest.raises(ValueError, match="full-parameter"):
        require_lora_semantics([payload], operation="merged-hf")


def test_guard_rejects_if_any_shard_is_full_param():
    """多 shard：任一来自全参就必须整体拒绝（不能只检查第一个）。"""
    payloads = [
        {"tuner_type": "lora", "full_param_state_dict": None},
        {"tuner_type": "full", "full_param_state_dict": {}},
    ]
    with pytest.raises(ValueError, match="peft-adapter"):
        require_lora_semantics(payloads, operation="peft-adapter")
