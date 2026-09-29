"""``tuner_type``（LoRA / 全参）配置校验的**纯逻辑**测试（不依赖 torch）。

覆盖两层：
- 纯函数 ``resolve_tuner_type`` / ``validate_tuner_type_combination``（非法组合的判据）；
- 真实 ``GraspoConfig`` 的字段接线（``tuner_type`` ⇒ ``effective_tuner_type``、
  Literal 取值校验、根级 validator 确实被调用）。

**为什么需要下面那段垫片（本机无 torch 时的纯逻辑测试通道）**

``graspo/__init__.py`` 在 import 期导入 ``graspo.ripple.reward.reward``，
该模块又经 ``graspo/ripple/__init__.py`` → ``ripple/algorithm_core.py:13`` 拉入 torch；
``core/schema.py`` 的 ``reward: RewardConfig = RewardConfig()`` 也会在类体求值一次。
⇒ 在没有 torch 的机器上，连"配置字段长什么样"都无法收集。

垫片只在 **torch 缺失时**替换 ``graspo.ripple.reward.reward`` 这一个模块，
提供 pydantic 校验真正用到的 ``REWARD_REGISTRY`` 与 ``graspo/__init__`` 需要再导出
的三个名字。**不改任何被测代码**；装了 torch 的环境一行都不执行。

垫片是**"用完即摘"**的（``with`` 作用域）：``schema`` 只在**导入期**与**每次实例化**
时查一次该模块名，因此只在这两个时点临时登记、随后立刻 ``sys.modules.pop``。
（实测：本文件加入前后，全仓收集 **702 → 744**——新增的 42 例正是本工作包的测试——
收集错误数**完全不变**（22），即没有顺带"解锁"任何本工作包之外的用例。）
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types


@contextlib.contextmanager
def _no_torch_reward_stub():
    """本机无 torch 时，临时登记 ``graspo.ripple.reward.reward`` 垫片；退出即摘除。

    有 torch 的环境（容器/CI）直接放行，什么都不做；若该模块已被别人导入，
    也不覆盖、不摘除（避免影响他人）。
    """
    already_imported = "graspo.ripple.reward.reward" in sys.modules
    if importlib.util.find_spec("torch") is not None or already_imported:
        yield
        return
    stub = types.ModuleType("graspo.ripple.reward.reward")
    stub.REWARD_REGISTRY = {"graspo": object()}
    stub.GraspoReward = object
    stub.RewardConfig = object
    stub.RewardResult = object
    sys.modules["graspo.ripple.reward.reward"] = stub
    try:
        yield
    finally:
        sys.modules.pop("graspo.ripple.reward.reward", None)


with _no_torch_reward_stub():
    _schema = importlib.import_module("graspo.core.schema")

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

GraspoConfig = _schema.GraspoConfig
resolve_tuner_type = _schema.resolve_tuner_type
validate_tuner_type_combination = _schema.validate_tuner_type_combination


def _validate_config(data: dict) -> object:
    """构造真实 ``GraspoConfig``（本机无 torch 时临时挂 reward 垫片）。"""
    with _no_torch_reward_stub():
        return GraspoConfig.model_validate(data)


# ── resolve_tuner_type：None 语义（宪法 §2.2）─────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "lora"),  # 未指定 ⇒ 默认 lora（向后兼容 v0.24 之前的所有配置）
        ("lora", "lora"),
        ("full", "full"),
    ],
)
def test_resolve_tuner_type_normalizes_none_to_lora(raw, expected):
    assert resolve_tuner_type(raw) == expected


# ── validate_tuner_type_combination：非法组合 fail-closed ────────────────


def _validate(**overrides: object) -> None:
    kwargs: dict[str, object] = {
        "tuner_type": None,
        "backend": "native",
        "lora_adapter_path": None,
        "native_tp_size": 1,
        "native_dp_size": 1,
    }
    kwargs.update(overrides)
    validate_tuner_type_combination(**kwargs)  # type: ignore[arg-type]


def test_lora_path_is_never_restricted():
    """LoRA 路径不受任何新校验影响（回归保护：平级共存）。"""
    _validate(tuner_type=None, lora_adapter_path="adapters/x", native_tp_size=4, native_dp_size=8)
    _validate(tuner_type="lora", lora_adapter_path="adapters/x", native_tp_size=4, native_dp_size=8)


def test_full_rejects_lora_adapter_path():
    with pytest.raises(ValueError, match="adapter_path"):
        _validate(tuner_type="full", lora_adapter_path="adapters/x")


def test_full_with_native_tensor_parallel_is_rejected():
    with pytest.raises(ValueError, match="pipeline parallelism only"):
        _validate(tuner_type="full", native_tp_size=2)


def test_full_with_native_data_parallel_is_rejected():
    with pytest.raises(ValueError, match="pipeline parallelism only"):
        _validate(tuner_type="full", native_dp_size=2)


def test_full_with_native_pipeline_parallel_is_allowed():
    """PP 是 native 全参的推荐分片方式，必须放行。"""
    _validate(tuner_type="full", native_tp_size=1, native_dp_size=1)


def test_full_with_msswift_backend_is_not_restricted_by_native_sizes():
    """msswift 的并行度由 DeepSpeed/FSDP/Megatron 决定，native.tp_size 不适用。"""
    _validate(tuner_type="full", backend="msswift", native_tp_size=4, native_dp_size=8)


# ── GraspoConfig 接线（真实 pydantic 字段与校验）─────────────────────────


def test_graspo_config_defaults_to_lora_and_accepts_full():
    """未指定 ⇒ lora（向后兼容）；显式 full 被接受。"""
    assert _validate_config({}).effective_tuner_type == "lora"
    assert _validate_config({"tuner_type": "full"}).effective_tuner_type == "full"


def test_graspo_config_rejects_unknown_tuner_type():
    with pytest.raises(ValidationError, match="tuner_type"):
        _validate_config({"tuner_type": "qlora"})


def test_graspo_config_applies_the_full_param_combination_validator():
    """全参 + native TP 的 fail-closed 拒绝必须真的接在配置加载上；PP 必须放行。"""
    with pytest.raises(ValidationError, match="pipeline parallelism only"):
        _validate_config({"tuner_type": "full", "native": {"tp_size": 2}})
    assert _validate_config({"tuner_type": "full", "native": {"pp_size": 4}}).native.pp_size == 4
