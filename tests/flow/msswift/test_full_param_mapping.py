"""``tuner_type`` → ms-swift ``--tuner_type`` 映射的契约测试（纯逻辑，不依赖 torch / ms-swift）。

覆盖两侧：
- **全量路径正确**：``tuner_type: full`` ⇒ 产出 ``--tuner_type full``，且**不**产出
  LoRA 专属参数（全参下发 ``--lora_rank`` 是"看起来生效、实际被忽略"的假配置）。
- **LoRA 路径不受影响**：默认（未指定）与 ``tuner_type: lora`` 仍产出
  ``--tuner_type lora`` + 完整的 LoRA 参数向量（平级共存的回归保护）。
- **AutoTP 归位**：``msswift.deepspeed_autotp_size`` 只在全参下放行，非全参仍 fail-closed。

上游取值依据（【实测确证】，不是猜的）：
- ms-swift 4.5.3 源码（`.local/refs/ms-swift-4.5.3/`）`src/swift/arguments/base_args/base_args.py:29-31`
  的 `get_supported_tuners()` 含 `'full'`；Megatron 路径 `megatron_args.py:449`
  直接写 `tuner_type: Literal['lora', 'full', 'lora_llm'] = 'full'`。
- 目标 GPU 服务器实跑证据（记录见 `.local/` 下 `task-megatron-smoke/evidence/`）：`88-train-v5-error.txt:14`（CLI usage 行
  `[--tuner_type {lora,full,lora_llm}]`）与 `81-train-v3-status.txt`（真实运行的
  `MegatronSftArguments(..., tuner_type='full', ...)` 参数 dump）。

**不依赖 torch**：被测算子 ``graspo_to_ms_swift_argv`` / ``validate_combinations`` 是纯映射
（只做属性访问 + 拼 argv）。

**机器无 torch 时怎么导入**（`graspo/__init__.py` 在 import 期就拉 torch，见
`flow/msswift/__init__.py` docstring 与 `core/schema.py` 的 `RewardConfig()`）：

1. 所有 graspo 侧导入**延迟到运行时**（`_graspo_modules()`，由测试函数触发），收集期
   一行 graspo 代码都不 import；
2. 用到时临时装一个**仅 torch 缺失时生效**的 reward 垫片，用完 ``sys.modules.pop``；
3. ``_rope_compat`` / ``_config_mapping`` 按**文件路径**加载（不执行父包 ``__init__``、
   不登记 ``sys.modules``），只在加载 ``_config_mapping`` 期间临时登记 ``_rope_compat``
   的真实模块名，加载完立刻摘掉。

**为什么这么绕**：只要在**收集期**注册了 ``sys.modules["graspo.flow"]`` 替身包、或成功
import 了 ``graspo.core.schema``，同目录/同仓库其它测试文件里对应的 import 就会跟着
"成功"，于是它们被额外收集并在运行期失败（实测 `tests/flow` 从 "4 collected / 20 errors"
变成 "72 collected / 16 errors"，多出的失败全部来自本工作包之外的用例）。延迟 + 临时登记
后实测**零全局副作用**：全仓收集 702 → 744（新增的 42 例正是本工作包的测试），
错误数完全不变（22）。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"


def _load_module(dotted_name: str, relative_path: str):
    """按文件路径加载模块（不执行父包 ``__init__``，也**不登记** ``sys.modules``）。

    登记真实模块名会让**其它**测试文件的 ``from graspo.flow.msswift._config_mapping
    import ...`` 意外成功（实测会多收集 4 个文件并失败），因此这里保持零全局状态；
    确实需要临时登记的场景由调用方显式 try/finally 处理。
    """
    spec = importlib.util.spec_from_file_location(dotted_name, _SRC / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if importlib.util.find_spec("torch") is None:  # pragma: no cover - 取决于运行环境
    # `_config_mapping` 内部会 `from graspo.flow.msswift._rope_compat import ...`：
    # 因此**只在加载 `_config_mapping` 期间**临时登记 `_rope_compat`，随后立刻摘掉，
    # 保证测试跑完后 sys.modules 与本文件存在前完全一致（零全局副作用）。
    _ROPE_NAME = "graspo.flow.msswift._rope_compat"
    sys.modules[_ROPE_NAME] = _load_module(_ROPE_NAME, "graspo/flow/msswift/_rope_compat.py")
    try:
        _mapping_module = _load_module(
            "graspo.flow.msswift._config_mapping", "graspo/flow/msswift/_config_mapping.py"
        )
    finally:
        sys.modules.pop(_ROPE_NAME, None)

    graspo_to_ms_swift_argv = _mapping_module.graspo_to_ms_swift_argv
    validate_combinations = _mapping_module.validate_combinations
else:  # pragma: no cover - 有 torch 的正常环境
    from graspo.flow.msswift._config_mapping import (
        graspo_to_ms_swift_argv,
        validate_combinations,
    )

import contextlib  # noqa: E402

import pytest  # noqa: E402


@contextlib.contextmanager
def _no_torch_reward_stub():
    """本机无 torch 时，临时登记 ``graspo.ripple.reward.reward`` 垫片；退出即摘除。

    ``core/schema.py`` 只在**导入期**与**每次实例化**时查一次该模块名，因此只在这两个
    时点临时登记即可，测试跑完后模块表与本文件存在前完全一致（零全局副作用）。
    有 torch 的环境直接放行；该模块若已被别人导入则不覆盖、不摘除。
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


_LAZY: dict[str, object] = {}


def _graspo_modules():
    """**运行时**（而非收集期）才导入 graspo 侧模块，并把它们缓存进 ``_LAZY``。

    为什么要延迟：本机无 torch 时，只要在**收集期**成功 import 了
    ``graspo.core.schema``，同目录其它测试文件里那句 ``from graspo.core.schema import
    GraspoConfig`` 也会跟着"成功"（模块已缓存），于是它们被额外收集、随后在运行期失败
    （实测 `tests/flow` 从 "4 collected / 20 errors" 变成 "72 collected / 16 errors"）。
    推迟到运行时导入，收集期的行为与本文件不存在时**完全一致**。
    """
    if "mapping" in _LAZY:
        return _LAZY

    with _no_torch_reward_stub():
        _LAZY["schema"] = importlib.import_module("graspo.core.schema")

    # `_config_mapping` 内部会 `from graspo.flow.msswift._rope_compat import ...`：
    # 因此**只在加载 `_config_mapping` 期间**临时登记 `_rope_compat`，随后立刻摘掉。
    rope_name = "graspo.flow.msswift._rope_compat"
    sys.modules[rope_name] = _load_module(rope_name, "graspo/flow/msswift/_rope_compat.py")
    try:
        _LAZY["mapping"] = _load_module(
            "graspo.flow.msswift._config_mapping", "graspo/flow/msswift/_config_mapping.py"
        )
    finally:
        sys.modules.pop(rope_name, None)
    return _LAZY


def _config(*, tuner_type: str | None = None, lora: dict | None = None, **msswift: object):
    """真实 ``GraspoConfig``（修正字段值，避免用替身对象造成契约漂移）。"""
    modules = _graspo_modules()
    data: dict = {
        "backend": "msswift",
        "model": {"model_path": "models/Qwen3.5-9B"},
        "data": {"train_path": "unused.jsonl", "max_prompt_length": 2048},
        "training": {"max_new_tokens": 256},
        "msswift": dict(msswift),
    }
    if tuner_type is not None:
        data["tuner_type"] = tuner_type
    if lora is not None:
        data["lora"] = lora
    schema = modules["schema"]
    with _no_torch_reward_stub():
        return schema.GraspoConfig.model_validate(data)


def _argv(config, stage: str = "sft") -> list[str]:
    mapping = _graspo_modules()["mapping"]
    return mapping.graspo_to_ms_swift_argv(
        config, stage=stage, dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )


def _validate_combinations(config) -> None:
    _graspo_modules()["mapping"].validate_combinations(config)


def _value_of(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    return argv[argv.index(flag) + 1]


# ── 全量路径 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["sft", "rlhf"])
def test_full_param_maps_to_tuner_type_full(stage):
    argv = _argv(_config(tuner_type="full"), stage=stage)

    assert _value_of(argv, "--tuner_type") == "full"


@pytest.mark.parametrize(
    "flag", ["--lora_rank", "--lora_alpha", "--lora_dropout", "--target_modules", "--adapters"]
)
def test_full_param_does_not_emit_lora_only_flags(flag):
    argv = _argv(_config(tuner_type="full"))

    assert flag not in argv


# ── 全量语义统一：解冻 ViT 与 aligner（矩阵 §4「训练全部权重」）─────────────


@pytest.mark.parametrize("stage", ["sft", "rlhf"])
def test_full_param_unfreezes_vit_and_aligner_by_default(stage):
    """ms-swift 对多模态 full 微调默认冻结 ViT/aligner；graspo 必须显式解冻，
    否则同一能力格在两个后端语义分叉（native 训全部权重 / ms-swift 只训语言主干）。"""
    argv = _argv(_config(tuner_type="full"), stage=stage)

    assert _value_of(argv, "--freeze_vit") == "false"
    assert _value_of(argv, "--freeze_aligner") == "false"


@pytest.mark.parametrize("raw", [None, "lora"])
def test_lora_path_does_not_emit_freeze_flags(raw):
    """LoRA 路径逐字不变：不透传 freeze_vit/freeze_aligner（上游行为不变）。"""
    argv = _argv(_config(tuner_type=raw))

    assert "--freeze_vit" not in argv
    assert "--freeze_aligner" not in argv


def test_full_param_freeze_flags_are_configurable():
    """显式配置覆盖模式默认值（唯一真相源的开关，不是环境变量）。"""
    argv = _argv(_config(tuner_type="full", freeze_vit=True, freeze_aligner=False))

    assert _value_of(argv, "--freeze_vit") == "true"
    assert _value_of(argv, "--freeze_aligner") == "false"


# ── LoRA 路径回归（平级共存）────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [None, "lora"])
def test_lora_path_is_unchanged_when_tuner_type_is_unset_or_lora(raw):
    """未指定（None）与显式 lora 必须产出完全相同的 argv——向后兼容。"""
    argv = _argv(_config(tuner_type=raw))

    assert _value_of(argv, "--tuner_type") == "lora"
    assert _value_of(argv, "--lora_rank") == "16"
    assert _value_of(argv, "--lora_alpha") == "32"
    assert _value_of(argv, "--lora_dropout") == "0.1"


def test_lora_path_still_emits_target_modules_and_adapters():
    argv = _argv(
        _config(
            tuner_type="lora",
            lora={"target_modules": ["q_proj", "v_proj"], "adapter_path": "adapters/x"},
        )
    )

    assert _value_of(argv, "--tuner_type") == "lora"
    assert "q_proj" in argv and "v_proj" in argv
    assert "adapters/x" in argv


def test_unset_tuner_type_and_lora_produce_identical_argv():
    assert _argv(_config(tuner_type=None)) == _argv(_config(tuner_type="lora"))


# ── AutoTP 归位 ─────────────────────────────────────────────────────────────


def test_autotp_is_allowed_with_full_param():
    """全参入口落地后，AutoTP 的合法前置（全参）成立，启动前校验必须放行。"""
    _validate_combinations(_config(tuner_type="full", deepspeed="zero2", deepspeed_autotp_size=2))


@pytest.mark.parametrize("raw", [None, "lora"])
def test_autotp_is_still_rejected_without_full_param(raw):
    """非全参时 AutoTP 仍是非法组合（上游只支持全参），fail-closed 不放宽。"""
    with pytest.raises(ValueError, match="deepspeed_autotp_size"):
        _validate_combinations(_config(tuner_type=raw, deepspeed="zero2", deepspeed_autotp_size=2))


def test_validate_combinations_accepts_default_config():
    _validate_combinations(_config())
    assert "--deepspeed_autotp_size" not in _argv(_config())
