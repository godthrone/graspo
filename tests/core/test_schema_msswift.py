"""L3 契约测试：``msswift`` 配置段（方案 §3 M3 / 决策 D5；宪法 §2.1、§7.2）。

断言：
- 段名与 ``backend`` 取值一致（一名一物）：``backend: msswift`` ↔ ``msswift:`` 段
- ``extra="forbid"`` 在**段内**与**嵌套 Megatron 段内**都拒绝未知字段
- 缺省值语义：``None`` = 未提供（与"显式 0/False"不同）
- ``msswift: null`` 等价于缺省（与其它段的 None 段防御一致）
- 声明即消费：``msswift`` 段的每个字段都能在映射层找到对应 ms-swift 参数
  （防"假配置"，§7.2）
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from graspo.core.schema import GraspoConfig, MsSwiftConfig, MsSwiftMegatronConfig
from graspo.flow.msswift._config_mapping import (
    _MEGATRON_PASSTHROUGH,
    _MSSWIFT_RLHF_ONLY,
    _MSSWIFT_SCALAR_PASSTHROUGH,
)

_MAPPING_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "graspo"
    / "flow"
    / "msswift"
    / "_config_mapping.py"
)

#: ``msswift`` 段里**模式条件消费**的字段：只在 ``tuner_type: full`` 的分支里注入
#: （D-16：解冻 ViT/aligner，使 ms-swift 的"全参"与 native 一致地训练全部权重）。
#: 它们**不进** ``_MSSWIFT_SCALAR_PASSTHROUGH``——那是全模式透传清单，带进去会让
#: LoRA 路径也产出 ``--freeze_vit`` / ``--freeze_aligner``，破坏上游行为逐字不变。
_MSSWIFT_MODE_CONDITIONAL: tuple[str, ...] = ("freeze_vit", "freeze_aligner")


def test_section_name_matches_backend_value():
    """单一真相源：段名 = backend 取值（不出现 ms_swift 别名）。"""
    config = GraspoConfig.model_validate({"backend": "msswift", "msswift": {"fsdp": "fsdp2"}})

    assert config.backend == "msswift"
    assert config.msswift.fsdp == "fsdp2"
    assert "ms_swift" not in GraspoConfig.model_fields


def test_unknown_field_in_msswift_section_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        GraspoConfig.model_validate({"msswift": {"sequence_paralle_size": 2}})

    assert "sequence_paralle_size" in str(excinfo.value)


def test_unknown_field_in_megatron_section_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        GraspoConfig.model_validate({"msswift": {"megatron": {"tensor_parallel_size": 2}}})

    assert "tensor_parallel_size" in str(excinfo.value)


def test_megatron_is_a_nested_typed_model_not_a_dict():
    """§9.1：嵌套结构用 pydantic 模型，不做 dict 套 dict。"""
    config = GraspoConfig.model_validate({"msswift": {"megatron": {"context_parallel_size": 2}}})

    assert isinstance(config.msswift, MsSwiftConfig)
    assert isinstance(config.msswift.megatron, MsSwiftMegatronConfig)
    assert config.msswift.megatron.context_parallel_size == 2


def test_none_semantics_are_distinct_from_zero():
    """``None`` = 未提供；显式 0/False 是用户意图，两者不混同（§2.2）。"""
    default = GraspoConfig.model_validate({"msswift": {}})
    explicit = GraspoConfig.model_validate({"msswift": {"zero_hpz_partition_size": 0}})

    assert default.msswift.zero_hpz_partition_size is None
    assert explicit.msswift.zero_hpz_partition_size == 0


def test_null_section_is_equivalent_to_missing():
    """`msswift: null` 与缺省等价（与其它段的 None 段防御一致，见 GraspoConfig.from_dict）。"""
    config = GraspoConfig.from_dict({"msswift": None})

    assert config.msswift.sequence_parallel_size == 1


def test_sequence_parallel_size_rejects_non_integer():
    with pytest.raises(ValidationError):
        GraspoConfig.model_validate({"msswift": {"sequence_parallel_size": "two"}})


def test_backend_agnostic_defaults_do_not_change_native_config():
    """新增 msswift 段不改变 native 默认行为（§1.2 对修改关闭）。"""
    config = GraspoConfig.model_validate({})

    assert config.backend == "native"
    assert config.native.dp_size == 1
    assert config.msswift.sequence_parallel_size == 1


def test_every_msswift_field_is_consumed_by_the_mapping():
    """声明即消费（§7.2 假配置防呆）：段里每个字段名都出现在映射层的显式清单中。

    两类例外，都是**显式登记 + 单独断言**，不是免检：

    - ``attn_impl``：经 ``resolve_attn_impl()`` 显式解析（带优先级），只出现在源码
      里，不在同名透传清单中。
    - ``freeze_vit`` / ``freeze_aligner``（:data:`_MSSWIFT_MODE_CONDITIONAL`）：
      **模式条件消费**——只在 ``full`` 分支里 ``_extend``，LoRA 分支必须逐字不变。
      因此**绝不能**放进 ``_MSSWIFT_SCALAR_PASSTHROUGH``（那会让 LoRA 路径也产出这两个
      flag，破坏上游行为不变），必须在 full 分支单独注入。行为侧由
      ``tests/flow/msswift/test_full_param_mapping.py`` 全量覆盖；这里补一条源码级断言，
      保证例外清单里的字段**在映射层真的被消费**（写下名字却没实现 = 假配置回归）。
    """
    source = _MAPPING_SOURCE.read_text(encoding="utf-8")

    declared = set(MsSwiftConfig.model_fields) - {"megatron"}
    consumed = (
        set(_MSSWIFT_SCALAR_PASSTHROUGH)
        | set(_MSSWIFT_RLHF_ONLY)
        | set(_MSSWIFT_MODE_CONDITIONAL)
        | {"attn_impl", "nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port"}
    )
    assert declared <= consumed, f"declared but never mapped: {sorted(declared - consumed)}"

    # 例外清单不是免检清单：每个模式条件字段都必须在映射源码里被当作 ms-swift 参数注入。
    for name in _MSSWIFT_MODE_CONDITIONAL:
        assert f'"{name}",' in source, (
            f"{name} is listed in _MSSWIFT_MODE_CONDITIONAL but never _extend()ed "
            "by the mapping — declared-but-unmapped (假配置回归)"
        )

    declared_megatron = set(MsSwiftMegatronConfig.model_fields)
    assert declared_megatron <= set(_MEGATRON_PASSTHROUGH), (
        f"megatron fields never mapped: {sorted(declared_megatron - set(_MEGATRON_PASSTHROUGH))}"
    )


def test_mapping_lists_are_explicit_not_introspected():
    """防呆：映射清单是显式字面量，不是 ``dataclasses.fields()`` 之类自省。

    若改成自省，新增 schema 字段会被"顺手"透传，绕过 review——这条测试把该约定钉死。
    """
    source = _MAPPING_SOURCE.read_text(encoding="utf-8")

    assert "_MSSWIFT_SCALAR_PASSTHROUGH: tuple[str, ...] = (" in source
    assert "model_fields" not in source, "mapping must not introspect the schema"
    assert "dataclasses.fields" not in source, "mapping must not introspect dataclasses"
    assert "getattr(section, name)" in source
