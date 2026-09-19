"""``native.offload_optimizer_state``（WP-X2：native 全参优化器态 CPU offload）的
**纯逻辑**测试（不依赖 torch）。

覆盖三层：
- 纯函数 ``validate_native_offload_combination``（非法组合的判据）；
- 真实 ``GraspoConfig`` 的字段接线（默认关闭、字段映射、拼写错误仍被拒）；
- **负向用例**：两条 fail-closed 规则必须真的拦截（本文件另有"能真失败"的
  变异实测证据，见工位 ``report.md`` §⑤）。

**为什么需要下面那段垫片**：与 ``tests/core/test_tuner_type.py`` 同一个原因——
``graspo/__init__.py`` 在 import 期经 Ripple 拉入 torch，本机（开发机）无 torch。
垫片只在 **torch 缺失时**替换 ``graspo.ripple.reward.reward`` 一个模块，**不改任何
被测代码**；装了 torch 的环境一行都不执行。
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types


@contextlib.contextmanager
def _no_torch_reward_stub():
    """本机无 torch 时，临时登记 ``graspo.ripple.reward.reward`` 垫片；退出即摘除。"""
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
GraspoFlowConfig = _schema.GraspoFlowConfig
validate_native_offload_combination = _schema.validate_native_offload_combination

#: `T016` 档位形状的原样配置：9B · SFT · 全参 · native · 1 卡。
_T016_SHAPED = {
    "train_method": "sft",
    "backend": "native",
    "tuner_type": "full",
    "native": {"pp_size": 1, "tp_size": 1, "dp_size": 1, "offload_optimizer_state": True},
}


def _validate_config(data: dict) -> object:
    """构造真实 ``GraspoConfig``（本机无 torch 时临时挂 reward 垫片）。"""
    with _no_torch_reward_stub():
        return GraspoConfig.model_validate(data)


def _validate(**overrides: object) -> None:
    kwargs: dict[str, object] = {
        "offload_optimizer_state": False,
        "backend": "native",
        "native_adapter": "graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter",
    }
    kwargs.update(overrides)
    validate_native_offload_combination(**kwargs)  # type: ignore[arg-type]


# ── 默认关闭：既有配置行为逐位不变 ───────────────────────────────────────


def test_field_defaults_to_disabled_in_the_native_schema():
    assert GraspoFlowConfig().offload_optimizer_state is False
    assert _validate_config({}).native.offload_optimizer_state is False


def test_default_offload_is_false_not_a_sentinel():
    """§2.2：默认值是布尔假，不是 ``0``/``""`` 之类的非 None 哨兵。"""
    default = GraspoFlowConfig.model_fields["offload_optimizer_state"].default
    assert default is False


def test_disabled_offload_is_never_restricted():
    """★ 回归保护：开关关闭时，任何 backend / adapter 组合都**不受**新校验影响。"""
    _validate(offload_optimizer_state=False, backend="msswift")
    _validate(offload_optimizer_state=False, native_adapter="some.other.Adapter")
    # 关闭状态下连"全参 + msswift + 别的适配器"也不得被这条新规则拦。
    assert _validate_config(
        {"backend": "msswift", "native": {"offload_optimizer_state": False}}
    ).native.offload_optimizer_state is False


def test_existing_configs_without_the_key_are_untouched():
    """不含该键的既有配置：加载成功、字段取默认假。"""
    cfg = _validate_config({"train_method": "sft", "tuner_type": "full", "native": {"pp_size": 2}})
    assert cfg.native.offload_optimizer_state is False
    assert cfg.native.pp_size == 2


# ── 正向：T016 形状的配置必须放行 ────────────────────────────────────────


def test_t016_shaped_config_is_accepted():
    """native · SFT · 全参 · 1 卡 · offload 开启 ⇒ 合法（这就是要上机的档）。"""
    cfg = _validate_config(_T016_SHAPED)
    assert cfg.native.offload_optimizer_state is True
    assert cfg.native.pp_size == 1
    assert cfg.effective_tuner_type == "full"


def test_adapter_module_prefix_allows_sibling_classes():
    """同模块内的子类/别名路径也应通过（按模块前缀判定，不按逐字相等）。"""
    _validate(native_adapter="graspo.flow.adapters.models.qwen35_36.adapter:SubclassAdapter")


# ── 字段接线：映射正确 + 拼写错误仍被拒 ──────────────────────────────────


def test_field_is_mapped_into_the_native_section():
    cfg = _validate_config({"native": {"offload_optimizer_state": True}})
    assert cfg.native.offload_optimizer_state is True


def test_typo_in_the_field_name_is_still_rejected():
    """§7.2：拼错的字段名不得静默忽略（``extra="forbid"``）。"""
    with pytest.raises(ValidationError, match="offload_optimizer_states"):
        _validate_config({"native": {"offload_optimizer_states": True}})


def test_non_boolean_value_is_rejected():
    """配置契约是布尔，不接受无法判真假的字符串——避免"看起来开了其实没开"。

    注：pydantic 的宽松布尔**接受** ``"yes"``/``"on"``/``"1"`` 这类常见写法，
    这是既有的全项目约定，不在这里改变；本用例只锁"真的判不出来 ⇒ 拒绝"。
    """
    with pytest.raises(ValidationError):
        _validate_config({"native": {"offload_optimizer_state": "maybe"}})


# ── 负向用例：两条 fail-closed 规则必须真的拦 ────────────────────────────


def test_offload_with_msswift_backend_is_rejected():
    """规则 1：native.* 的开关在 msswift 后端上无人消费 ⇒ 假配置，必须拒。"""
    with pytest.raises(ValueError, match="only wired into the native backend"):
        _validate(offload_optimizer_state=True, backend="msswift")
    with pytest.raises(ValidationError, match="only wired into the native backend"):
        _validate_config(
            {"backend": "msswift", "native": {"offload_optimizer_state": True}}
        )


def test_offload_with_non_qwen35_adapter_is_rejected():
    """规则 2：接线只落在 qwen35_36 的 mixin 上 ⇒ 别的 model family 必须拒。"""
    with pytest.raises(ValueError, match="silently keep the optimizer state on the GPU"):
        _validate(
            offload_optimizer_state=True,
            native_adapter="graspo.flow.adapters.models.qwen3.adapter:Qwen3Adapter",
        )
    with pytest.raises(ValidationError, match="silently keep the optimizer state on the GPU"):
        _validate_config(
            {
                "native": {
                    "offload_optimizer_state": True,
                    "adapter": "graspo.flow.adapters.models.qwen3.adapter:Qwen3Adapter",
                }
            }
        )


def test_rejection_messages_point_at_the_alternative():
    """错误信息必须**可行动**（§2.3/§13.1）：说出为什么、以及改用什么。"""
    with pytest.raises(ValueError) as excinfo:
        _validate(offload_optimizer_state=True, backend="msswift")
    message = str(excinfo.value)
    assert "zero2_offload" in message
    assert "fake config" in message


# ── 接线契约（AST 静态检查，仍然不依赖 torch）───────────────────────────
#
# offload 的接线是"在 qwen35_36 的 mixin 上覆盖 ``_build_optimizer``"。**MRO 顺序
# 一改（别的 mixin 也定义同名方法、或 SFT mixin 被挪到 TransformerAdapter 之后），
# 覆盖就会静默失效** —— 配置照样加载、日志照样打、显存却一点没省。
# 本机无 torch 无法 import Qwen35Adapter，因此用 AST 把这条契约锁死。

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

_QWEN_DIR = Path("src/graspo/flow/adapters/models/qwen35_36")
_ADAPTER_PY = _QWEN_DIR / "adapter.py"
_BASE_PY = Path("src/graspo/flow/adapters/transformer_adapter.py")


def _class_defs(path: Path) -> dict[str, ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _defined_methods(node: ast.ClassDef) -> set[str]:
    return {
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_offload_hook_is_reachable_in_the_mro():
    """SFT mixin 的 ``_build_optimizer`` 覆盖必须真的抢在 ``TransformerAdapter`` 之前。"""
    adapter_classes = _class_defs(_ADAPTER_PY)
    qwen35 = adapter_classes["Qwen35Adapter"]
    bases = [ast.unparse(base).split(".")[-1] for base in qwen35.bases]
    assert "_Qwen35SFTTrainingMethods" in bases, bases
    assert "TransformerAdapter" in bases, bases
    assert bases.index("_Qwen35SFTTrainingMethods") < bases.index("TransformerAdapter"), bases


def test_only_the_sft_mixin_defines_the_optimizer_hook():
    """MRO 里先于 ``TransformerAdapter`` 的 mixin 只允许 SFT 那一个定义该钩子。

    否则覆盖会被更靠前的 mixin 抢走，offload 静默失效。
    """
    adapter_classes = _class_defs(_ADAPTER_PY)
    bases = [
        ast.unparse(base).split(".")[-1] for base in adapter_classes["Qwen35Adapter"].bases
    ]
    mixins = [name for name in bases if name != "TransformerAdapter"]

    definers: dict[str, set[str]] = {}
    for path in sorted(_QWEN_DIR.glob("*.py")):
        for name, node in _class_defs(path).items():
            if name in mixins and "_build_optimizer" in _defined_methods(node):
                definers.setdefault(name, set()).add(path.name)

    assert definers == {"_Qwen35SFTTrainingMethods": {"training_sft.py"}}, definers


def test_base_adapter_still_defines_the_optimizer_hook():
    """覆盖必须 ``super()`` 到一个**真实存在**的基类实现（不是空挂）。"""
    base_classes = _class_defs(_BASE_PY)
    assert "_build_optimizer" in _defined_methods(base_classes["TransformerAdapter"])


def test_sft_mixin_hook_delegates_to_super():
    """SFT mixin 的覆盖必须先调 ``super()._build_optimizer()``（基类逻辑一字不改）。"""
    sft = _class_defs(_QWEN_DIR / "training_sft.py")["_Qwen35SFTTrainingMethods"]
    hook = next(
        item
        for item in sft.body
        if isinstance(item, ast.FunctionDef) and item.name == "_build_optimizer"
    )
    source = ast.unparse(hook)
    assert "super()._build_optimizer()" in source

