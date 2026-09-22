"""``native + pp_size>1 + rollout`` 的启动期闸门（缺陷 P6 的 c1）。

契约（三层，全部要钉住）：

1. **默认拒绝**：``train_method=graspo`` + ``backend=native`` + ``pp_size>1`` ⇒
   ``validate_native_runtime_config`` 抛 ``RuntimeError``，且错误信息**点名开关**
   （``native.allow_unverified_pp_rollout``）与本缺陷的证据（T035/T036）。
2. **显式预授权后放行**：同一形状 + ``allow_unverified_pp_rollout: true`` ⇒ 不抛。
   这是"显式预授权"，不是 SKIP/静默跳过（§3.4 坏退路）。
3. **既有档位逐字不变**：``pp_size=1`` 的 GRASPO 档与 SFT 的 ``pp_size=2/4``
   档（T017/T018，已两跑 rc=0）**都必须照常通过**。

另钉住两件事：
* 闸门在**启动期**而不是配置加载期 ⇒ ``GraspoConfig.model_validate`` 对 T035 形状
  仍然成功（既有 46 档与多个测试依赖配置加载契约，§2.3 收紧但不改既有契约面）；
* 数值边界：``pp_rollout_p2p_timeout_sec >= 0``、``pp_rollout_no_progress_sec >= 1``。
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch", reason="native runtime config validation needs torch")

from graspo.core.schema import GraspoConfig  # noqa: E402
from graspo.flow.runtime import validate_native_runtime_config  # noqa: E402

#: T035 档位形状（9B · GRASPO · 全参 · native · 2 卡 ⇒ pp_size=2）。
_T035_SHAPED: dict[str, Any] = {
    "train_method": "graspo",
    "backend": "native",
    "tuner_type": "full",
    "native": {"tp_size": 1, "dp_size": 1, "pp_size": 2},
}
#: T036 档位形状（同上，4 卡 ⇒ pp_size=4）。
_T036_SHAPED: dict[str, Any] = {
    **_T035_SHAPED,
    "native": {"tp_size": 1, "dp_size": 1, "pp_size": 4},
}
#: T017/T018 档位形状（SFT 全参 native pp=2/4，**无 rollout**，实测已 rc=0）。
_T017_SHAPED: dict[str, Any] = {
    "train_method": "sft",
    "backend": "native",
    "tuner_type": "full",
    "native": {"tp_size": 1, "dp_size": 1, "pp_size": 2},
}
#: T028/T040 档位形状（GRASPO native 单卡 ⇒ pp_size=1，46 档里的主力形状）。
_T028_SHAPED: dict[str, Any] = {
    "train_method": "graspo",
    "backend": "native",
    "native": {"tp_size": 1, "dp_size": 1, "pp_size": 1},
}


def _config(data: dict[str, Any]) -> GraspoConfig:
    return GraspoConfig.model_validate(data)


def test_t035_shape_is_refused_at_runtime_startup() -> None:
    cfg = _config(_T035_SHAPED)
    with pytest.raises(RuntimeError) as info:
        validate_native_runtime_config(cfg)
    message = str(info.value)
    # 必须点名开关（可操作）与证据（可追溯）
    assert "native.allow_unverified_pp_rollout" in message
    assert "pp_size=2" in message
    assert "T035" in message and "T036" in message
    assert "unbounded" in message or "无界" in message


def test_t036_shape_is_refused_at_runtime_startup() -> None:
    with pytest.raises(RuntimeError, match="pp_size=4"):
        validate_native_runtime_config(_config(_T036_SHAPED))


def test_explicit_opt_in_allows_t035_shape() -> None:
    data = {
        **_T035_SHAPED,
        "native": {**_T035_SHAPED["native"], "allow_unverified_pp_rollout": True},
    }
    validate_native_runtime_config(_config(data))  # 不抛


def test_gate_lives_at_startup_not_at_config_load() -> None:
    """配置加载仍成功（既有 46 档的加载期契约不变），拒绝发生在启动期。"""
    cfg = _config(_T035_SHAPED)
    assert cfg.native.pp_size == 2
    assert cfg.native.allow_unverified_pp_rollout is False
    with pytest.raises(RuntimeError):
        validate_native_runtime_config(cfg)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(_T028_SHAPED, id="graspo-native-pp1"),
        pytest.param(
            {"train_method": "graspo", "backend": "native", "native": {"dp_size": 4}},
            id="graspo-native-dp4-pp1",
        ),
        pytest.param(_T017_SHAPED, id="sft-native-pp2-no-rollout"),
        pytest.param(
            {**_T017_SHAPED, "native": {**_T017_SHAPED["native"], "pp_size": 4}},
            id="sft-native-pp4-no-rollout",
        ),
    ],
)
def test_existing_shapes_are_untouched(shape: dict[str, Any]) -> None:
    """**pp_size=1 的 46 档与 SFT pp=2/4 必须逐字照常通过。**"""
    validate_native_runtime_config(_config(shape))


def test_defaults_are_documented_and_bounded() -> None:
    cfg = _config(_T028_SHAPED)
    assert cfg.native.pp_rollout_p2p_timeout_sec == 600
    assert cfg.native.pp_rollout_no_progress_sec == 300
    assert cfg.native.allow_unverified_pp_rollout is False


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("pp_rollout_p2p_timeout_sec", -1, "pp_rollout_p2p_timeout_sec"),
        ("pp_rollout_no_progress_sec", 0, "pp_rollout_no_progress_sec"),
    ],
)
def test_numeric_bounds_are_fail_closed(key: str, value: int, match: str) -> None:
    data = {**_T028_SHAPED, "native": {**_T028_SHAPED["native"], key: value}}
    with pytest.raises(ValueError, match=match):
        validate_native_runtime_config(_config(data))


def test_zero_timeout_means_disabled_but_is_not_a_boundary_error() -> None:
    """``pp_rollout_p2p_timeout_sec: 0`` 是**合法**的（= 关闭有界等待，退化为旧行为）。"""
    data = {**_T028_SHAPED, "native": {**_T028_SHAPED["native"], "pp_rollout_p2p_timeout_sec": 0}}
    validate_native_runtime_config(_config(data))
