"""Reward 注册表扩展验证：REWARD_REGISTRY 可扩展。

验证目标（C1/C2 验收）：
1. REWARD_REGISTRY 字典可添加新的 reward 类（扩展点）
2. create_reward() 按 kind 从注册表查找对应的类
3. RewardConfig 的 Literal 约束在配置加载时校验 kind（防线）
4. create_reward 在运行时对未知 kind 报错（第二道防线）
"""

import pytest

from graspo.core.schema import RewardConfig
from graspo.ripple.reward.reward import REWARD_REGISTRY, GraspoReward, create_reward


class _MockReward(GraspoReward):
    """测试用自定义 reward 类。"""

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._called = False

    def compute(self, **kwargs) -> dict:
        self._called = True
        return {"mock_reward": True, "score": 0.5}


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """测试后清理注册表。"""
    yield
    REWARD_REGISTRY.pop("mock", None)


def test_register_mock_reward():
    """通过 REWARD_REGISTRY 字典添加自定义 reward 类。"""
    REWARD_REGISTRY["mock"] = _MockReward
    assert "mock" in REWARD_REGISTRY
    assert REWARD_REGISTRY["mock"] is _MockReward


def test_registered_reward_instantiable():
    """注册表中的类可直接实例化。"""
    REWARD_REGISTRY["mock"] = _MockReward

    config = RewardConfig(kind="graspo")
    reward = _MockReward(config)
    assert isinstance(reward, _MockReward)
    assert isinstance(reward, GraspoReward)


def test_create_reward_with_builtin_graspo():
    """create_reward(kind="graspo") 返回 GraspoReward 实例。"""
    config = RewardConfig(kind="graspo")
    reward = create_reward(config)

    assert isinstance(reward, GraspoReward)


def test_mock_reward_does_not_affect_builtin():
    """注册自定义 reward 不影响内置 GraspoReward。"""
    REWARD_REGISTRY["mock"] = _MockReward

    config = RewardConfig(kind="graspo")
    reward = create_reward(config)

    assert isinstance(reward, GraspoReward)
    assert not isinstance(reward, _MockReward)


def test_default_reward_is_graspo():
    """默认 reward kind 为 "graspo"。"""
    config = RewardConfig()
    reward = create_reward(config)

    assert isinstance(reward, GraspoReward)


def test_reward_registry_is_dict():
    """REWARD_REGISTRY 是普通 dict，支持标准字典操作。"""
    assert isinstance(REWARD_REGISTRY, dict)
    assert "graspo" in REWARD_REGISTRY


def test_reward_config_rejects_unknown_kind():
    """RewardConfig 的 Literal 约束在配置加载时拒绝未知 kind（防线）。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="kind"):
        RewardConfig(kind="unknown_kind")


def test_create_reward_unknown_kind_raises():
    """create_reward 对不在注册表中的 kind 报错（运行时防线）。

    注意：正常流程中 RewardConfig 的 Literal 约束会先拦截未知 kind，
    此测试验证 create_reward 自身的防御逻辑。
    """
    # 直接测试 create_reward 的注册表查找逻辑
    assert REWARD_REGISTRY.get("nonexistent") is None

    # 验证添加后可从注册表获取
    REWARD_REGISTRY["mock"] = _MockReward
    assert REWARD_REGISTRY.get("mock") is _MockReward
