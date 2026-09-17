"""L3 集成测试：SFT 双后端分派闭环（方案 §9.3 / 决策 D2、D5）。

断言的是**后端选择与工厂的插拔契约**，不启动任何训练、不加载模型：
只构造配置、解析注册表、检查工厂形状与失败路径的精确性。

设计要点（对应方案 §9.3 判据）：

- ①/③ ``select_backend`` 在 native/msswift 间切换返回正确 ``BackendSelection``
- ④ SFT 不再有"该后端不支持 SFT"的 ``NotImplementedError``；两个后端都能解析出工厂
- ⑤ 注册表条目存在（entry_points / 开发回退同一真相源）
- 无 ms-swift 环境下用 ``pytest.importorskip``/显式断言跳过而不是静默通过
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from graspo.core.discovery import _discover, resolve_backend_builder
from graspo.core.schema import GraspoConfig

# ── 与 tests/flow/test_selector.py 同款加载方式 ─────────────────────────────
# 直接加载 backend_selection.py，避免触发 flow/__init__.py → trainer → torch 导入链。

_SELECTOR_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "graspo" / "flow" / "backend_selection.py"
)
_spec = importlib.util.spec_from_file_location(
    "graspo.flow.backend_selection", _SELECTOR_PATH, submodule_search_locations=[]
)
_selector = importlib.util.module_from_spec(_spec)
sys.modules["graspo.flow.backend_selection"] = _selector
_spec.loader.exec_module(_selector)

select_backend = _selector.select_backend


def _sft_config(backend: str) -> GraspoConfig:
    return GraspoConfig.model_validate({"train_method": "sft", "backend": backend})


# ── 判据 ①：后端切换返回正确 selection ─────────────────────────────────────


@pytest.mark.parametrize("backend", ["native", "msswift"])
def test_select_backend_returns_requested_backend_for_sft(backend):
    selection = select_backend(_sft_config(backend))

    assert selection.name == backend
    assert selection.requested == backend


def test_sft_and_rl_share_the_same_selector():
    """SFT 与 RL 用同一个选择器（决策 D5：不写死二选一分支）。"""
    assert select_backend(_sft_config("native")).name == "native"
    assert select_backend(GraspoConfig.model_validate({"train_method": "graspo"})).name == "native"


def test_backend_switch_does_not_leak_state():
    """同一 config 反复切换不残留状态（判据 ③）。"""
    native = select_backend(_sft_config("native"))
    msswift = select_backend(_sft_config("msswift"))

    assert (native.name, msswift.name) == ("native", "msswift")
    assert native.requested == "native"
    assert msswift.requested == "msswift"


# ── 判据 ④：SFT 双后端都能解析出工厂（核心：D2 缺口闭合）──────────────────


def test_sft_registry_contains_both_backends():
    registry = _discover("graspo.sft_backends")

    assert {"native", "msswift"} <= set(registry)


def test_resolve_native_sft_builder_returns_callable():
    builder = resolve_backend_builder("native", train_method="sft")

    assert callable(builder)


def test_resolve_msswift_sft_builder_returns_trainer_with_train():
    """msswift 的 SFT 入口可解析，且满足 worker 依赖的 ``.train(smoke)`` 契约。

    （曾经的 bug：工厂返回 partial，worker 调 ``.train()`` 直接 AttributeError。）
    """
    builder = resolve_backend_builder("msswift", train_method="sft")
    trainer = builder(_sft_config("msswift"), None)

    assert callable(builder)
    assert hasattr(trainer, "train"), f"{type(trainer).__name__} must expose train(smoke)"


def test_both_sft_backends_expose_identical_trainer_contract():
    """两个后端的 SFT 工厂契约形状一致：``factory(config, sel) -> 含 train 的对象``。

    native 侧需要 torch 才能实例化，故分环境断言：无 torch 时只断言工厂可解析。
    """
    msswift_trainer = resolve_backend_builder("msswift", train_method="sft")(
        _sft_config("msswift"), None
    )
    assert hasattr(msswift_trainer, "train")

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not importable — native SFT trainer cannot be instantiated here")
    from graspo.flow.trainer.sft_trainer import SFTTrainer, create_native_sft_trainer

    native_trainer = create_native_sft_trainer(_sft_config("native"), None)
    assert isinstance(native_trainer, SFTTrainer)
    assert hasattr(native_trainer, "train")


def test_native_sft_builder_shape_matches_contract():
    """native SFT 工厂契约：``factory(config, selection) -> 含 train(smoke) 的训练器``。

    ``SFTTrainer.__init__`` 会构造 ``GraspoFlowRuntime``（需要 torch），所以这里
    只验证工厂本身可解析 + 契约方法名存在，不做实例化（开发机无 GPU）。
    """
    pytest.importorskip("torch", reason="native SFT trainer needs torch to instantiate")
    from graspo.flow.trainer.sft_trainer import SFTTrainer, create_native_sft_trainer

    assert callable(create_native_sft_trainer)
    assert hasattr(SFTTrainer, "train")


def test_native_sft_factory_rejects_wrong_train_method():
    """防呆：SFT 工厂拒绝非 SFT 配置（不静默按 RL 处理）。"""
    pytest.importorskip("torch", reason="importing sft_trainer needs torch")
    from graspo.flow.trainer.sft_trainer import create_native_sft_trainer

    rl_config = GraspoConfig.model_validate({"train_method": "graspo", "backend": "native"})

    with pytest.raises(ValueError, match="train_method='sft'"):
        create_native_sft_trainer(rl_config)


def test_msswift_sft_is_wired_not_a_placeholder():
    """E2 后 msswift SFT 已接通：不再是"训练循环待接入"的 ``NotImplementedError``。

    本测试只断言**前置条件与状态**，不启动训练（真实训练由端到端实验在 GPU 验证节点上验收）：

    - 环境无 ms-swift → ``RuntimeError`` + 可操作的安装指引（不是静默假训练器）
    - 环境有 ms-swift → ``ms_sft_available()`` 为真，且历史占位异常不再出现
      （真实执行路径由 ``tests/flow/msswift/test_msswift_wiring.py`` 用替身覆盖）
    """
    from graspo.flow.msswift.sft_trainer import ms_sft_available

    builder = resolve_backend_builder("msswift", train_method="sft")
    trainer = builder(_sft_config("msswift"), None)

    if importlib.util.find_spec("swift") is None:
        with pytest.raises(RuntimeError) as excinfo:
            trainer.train(smoke=True)
        message = str(excinfo.value)
        assert message.startswith("backend='msswift' SFT requires"), message
        assert "pip install graspo[msswift]" in message, message
        return

    assert ms_sft_available() is True


def test_msswift_sft_dispatch_does_not_import_torch_or_swift():
    """分派层不触发重量级导入（宪法 §1.3：计算与设施分离）。

    断言方式：记录分派前的 ``sys.modules``，分派后只允许**不增加** ms-swift /
    torch 的导入（其他测试可能已导入它们，因此比较前后集合而不是空集）。
    """
    before = set(sys.modules)

    resolve_backend_builder("msswift", train_method="sft")(_sft_config("msswift"), None)

    added = set(sys.modules) - before
    assert "swift" not in added, f"dispatch must not import ms-swift: {sorted(added)[:5]}"
    assert "torch" not in added, f"dispatch must not import torch: {sorted(added)[:5]}"


# ── 判据 ⑤：注册项与"新增后端 0 行改动"的可验证性 ─────────────────────────


def test_both_registries_are_separate_and_complete():
    """SFT 与 RL 各有独立注册表，且都含 native/msswift。"""
    rl = _discover("graspo.backends")
    sft = _discover("graspo.sft_backends")

    assert {"native", "msswift"} <= set(rl)
    assert {"native", "msswift"} <= set(sft)


def test_resolve_rejects_unknown_backend_with_registry_listing():
    with pytest.raises(ValueError, match="no-such-backend"):
        resolve_backend_builder("no-such-backend", train_method="sft")


def test_resolve_rejects_removed_backend_names():
    for removed in ("auto", "hf-reference", "megatron-vllm"):
        with pytest.raises(ValueError, match="Unsupported backend"):
            select_backend(_sft_config("native"), requested=removed)


def test_third_backend_needs_zero_code_changes_in_selection_logic():
    """第三后端可插拔：注册表新增一个条目即可被解析，选择逻辑无需改动。

    用 monkeypatch 在 ``_DEV_FALLBACKS`` 上模拟"未来第三个后端"注册，
    断言 ``resolve_backend_builder`` 能发现它——证明扩展点是注册表而非 if/else。
    """
    import graspo.core.discovery as discovery

    sentinel = lambda: "third-backend-factory"  # noqa: E731
    fallbacks = dict(discovery._DEV_FALLBACKS)
    fallbacks["graspo.sft_backends"] = dict(fallbacks["graspo.sft_backends"])
    fallbacks["graspo.sft_backends"]["thirdparty"] = "os.path:join"

    original = discovery._DEV_FALLBACKS
    discovery._DEV_FALLBACKS = fallbacks
    try:
        builder = resolve_backend_builder("thirdparty", train_method="sft")
    finally:
        discovery._DEV_FALLBACKS = original

    assert builder is __import__("os.path", fromlist=["join"]).join
    assert sentinel() == "third-backend-factory"
