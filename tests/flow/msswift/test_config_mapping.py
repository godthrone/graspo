"""L1：``msswift`` 配置段 → ms-swift 参数的纯映射（方案 §8 矩阵透传路径）。

断言的是**纯计算**：给定 graspo 配置，产出的参数向量里出现/不出现哪些 ms-swift 参数。
不启动训练、不加载模型、不写文件（宪法 §1.3；§9.1 的 L1 判据）。

覆盖方案 §8 的 18 行里需要"可配置透传"的部分：
S1-S7 标准路径、MG1-MG11 Megatron 路径、以及长文附项。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.msswift._config_mapping import (
    graspo_to_ms_swift_argv,
    launcher_env,
    megatron_passthrough_argv,
    msswift_passthrough_argv,
    resolve_attn_impl,
)


def _config(**msswift: object) -> GraspoConfig:
    return GraspoConfig.model_validate(
        {
            "backend": "msswift",
            "model": {"model_path": "models/Qwen3.5-9B", "attn_implementation": None},
            "data": {"train_path": "unused.jsonl", "max_prompt_length": 2048},
            "training": {"max_new_tokens": 256},
            "msswift": dict(msswift),
        }
    )


def _value_of(argv: list[str], flag: str) -> str | None:
    """参数向量里 ``--flag`` 后面跟的值；不存在返回 None。"""
    if flag not in argv:
        return None
    return argv[argv.index(flag) + 1]


def _to_argv(config: GraspoConfig, stage: str = "rlhf") -> list[str]:
    return graspo_to_ms_swift_argv(
        config, stage=stage, dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )


def _to_megatron_argv(config: GraspoConfig) -> list[str]:
    """Megatron 通道的分片参数向量（MG1-MG11 只在这条通道里合法）。"""
    return megatron_passthrough_argv(config)


# ── 基础形状 ────────────────────────────────────────────────────────────────


def test_argv_carries_model_dataset_and_output_dir():
    argv = _to_argv(_config())

    assert _value_of(argv, "--model") == "models/Qwen3.5-9B"
    assert _value_of(argv, "--dataset") == "/tmp/ds.jsonl"
    assert _value_of(argv, "--output_dir") == "/tmp/out"


def test_sft_stage_uses_max_length_and_no_completion_length():
    argv = _to_argv(_config(), stage="sft")

    assert _value_of(argv, "--max_length") == "2048"
    assert "--max_completion_length" not in argv


def test_rlhf_stage_splits_prompt_and_completion_budgets():
    """§7.4 参数正交：prompt 上限与 completion 上限是两个独立参数。"""
    argv = _to_argv(_config(), stage="rlhf")

    assert _value_of(argv, "--max_length") == "2048"
    assert _value_of(argv, "--max_completion_length") == "256"


def test_rlhf_stage_maps_rollout_group_size_to_num_generations():
    config = GraspoConfig.model_validate(
        {"backend": "msswift", "training": {"rollout_group_size": 4}}
    )

    assert _value_of(_to_argv(config), "--num_generations") == "4"


def test_invalid_stage_is_rejected():
    with pytest.raises(ValueError, match="stage must be"):
        graspo_to_ms_swift_argv(
            _config(),
            stage="grpo",
            dataset_path="x",
            output_dir="y",  # type: ignore[arg-type]
        )


# ── §8.1 标准路径 S1-S7 ────────────────────────────────────────────────────


def test_s7_sequence_parallel_size_defaults_to_one_and_is_always_explicit():
    """默认 SP=1 也显式透传：不透传就等于把默认值交给 ms-swift 版本决定（§1.4）。"""
    assert _value_of(_to_argv(_config()), "--sequence_parallel_size") == "1"


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_s7_sequence_parallel_size_is_passed_through(size):
    argv = _to_argv(_config(sequence_parallel_size=size))

    assert _value_of(argv, "--sequence_parallel_size") == str(size)


@pytest.mark.parametrize(
    "field,value",
    [
        ("deepspeed", "zero3"),
        ("deepspeed", "zero2_offload"),
        ("fsdp", "fsdp2"),
        ("device_map", "auto"),
        ("zero_hpz_partition_size", 8),
        ("deepspeed_autotp_size", 2),
    ],
)
def test_s2_s6_sharding_flags_are_passed_through(field, value):
    argv = _to_argv(_config(**{field: value}))

    assert _value_of(argv, f"--{field}") == str(value)


def test_s1_launcher_args_go_to_environment_not_argv():
    """S1 在 ms-swift 侧是 launcher 环境变量（T1 复验结论），不是参数字段。"""
    config = _config(nproc_per_node=2, nnodes=1, master_port=29501)
    argv = _to_argv(config)

    assert "--nproc_per_node" not in argv
    env = launcher_env(config)
    assert env["NPROC_PER_NODE"] == "2"
    assert env["MASTER_PORT"] == "29501"
    assert "NODE_RANK" not in env  # None = 未提供 → 不覆盖既有环境


# ── 长文附项 ────────────────────────────────────────────────────────────────


def test_long_context_flags_are_passed_through():
    argv = _to_argv(
        _config(rope_scaling="dynamic", max_model_len=32768, packing=True, padding_free=True)
    )

    # rope_scaling 在映射层就换成 transformers 5.x 的新键（E2b 实测缺陷 1）：
    # ms-swift 的 --rope_scaling 收 JSON 字符串，它自己的 _init_rope_scaling 读 rope_type。
    assert json.loads(_value_of(argv, "--rope_scaling")) == {"rope_type": "dynamic"}
    assert _value_of(argv, "--max_model_len") == "32768"
    assert _value_of(argv, "--packing") == "true"
    assert _value_of(argv, "--padding_free") == "true"


def test_attn_impl_explicit_precedence():
    """显式优先级：msswift.attn_impl 优先于 model.attn_implementation（无隐藏 fallback）。"""
    from_legacy = GraspoConfig.model_validate(
        {"backend": "msswift", "model": {"attn_implementation": "flash_attention_2"}}
    )
    assert _value_of(_to_argv(from_legacy), "--attn_impl") == "flash_attention_2"

    overridden = GraspoConfig.model_validate(
        {
            "backend": "msswift",
            "model": {"attn_implementation": "flash_attention_2"},
            "msswift": {"attn_impl": "flash_attn"},
        }
    )
    assert _value_of(_to_argv(overridden), "--attn_impl") == "flash_attn"
    assert resolve_attn_impl(overridden) == "flash_attn"


def test_none_fields_are_not_passed_through():
    """None = 未提供（§2.2 None 语义）：不透传，交 ms-swift 默认值。"""
    argv = _to_argv(_config())

    for flag in ("--deepspeed", "--fsdp", "--device_map", "--rope_scaling", "--max_model_len"):
        assert flag not in argv


# ── §8.2 Megatron 路径 MG1-MG11 ────────────────────────────────────────────


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_batch_size", 32),
        ("data_sharding", True),
        ("use_distributed_optimizer", True),
        ("use_megatron_fsdp", True),
        ("data_parallel_sharding_strategy", "optim_grads"),
        ("tensor_model_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
        ("pipeline_model_parallel_layout", "Et*3|(tt|)*29,m|L"),
        ("sequence_parallel", True),
        ("context_parallel_size", 2),
        ("cp_comm_type", "a2a+p2p"),
        ("sequence_packing_scheduler", "dp_balanced"),
        ("expert_model_parallel_size", 4),
        ("expert_tensor_parallel_size", 2),
        ("virtual_pipeline_model_parallel_size", 2),
        ("microbatch_group_size_per_vp_stage", 4),
        ("fp8_param_gather", True),
    ],
)
def test_megatron_rows_1_to_11_are_passed_through(field, value):
    argv = _to_megatron_argv(_config(megatron={field: value}))

    # ms-swift 侧的布尔字面量是 ``true``/``false``（不是 Python 的 True/False）
    expected = "true" if value is True else str(value)
    assert _value_of(argv, f"--{field}") == expected


def test_megatron_params_absent_when_not_configured():
    argv = _to_megatron_argv(_config())

    assert "--tensor_model_parallel_size" not in argv
    assert "--context_parallel_size" not in argv


def test_megatron_params_stay_out_of_the_standard_channel():
    """实测（T1，4.5.3）：Megatron 参数不属于 ``SftArguments``/``RLHFArguments``。

    混进标准通道会被 ms-swift 的参数解析拒绝，因此标准向量里绝不能出现它们。
    """
    config = _config(megatron={"tensor_model_parallel_size": 2, "sequence_parallel": True})

    standard = _to_argv(config)
    megatron = _to_megatron_argv(config)

    assert "--tensor_model_parallel_size" not in standard
    assert "--sequence_parallel" not in standard
    assert "--tensor_model_parallel_size" in megatron


def test_megatron_passthrough_can_be_omitted_explicitly():
    argv = msswift_passthrough_argv(
        _config(megatron={"tensor_model_parallel_size": 2}), include_megatron=False
    )
    direct = megatron_passthrough_argv(_config(megatron={"tensor_model_parallel_size": 2}))

    assert "--tensor_model_parallel_size" not in argv
    assert "--tensor_model_parallel_size" in direct


# ── 计算与设施分离（§1.3）───────────────────────────────────────────────────


def test_mapping_is_pure_attribute_access_and_imports_nothing_heavy(monkeypatch):
    """映射层只做属性读取：即使传入"哑"配置对象也能工作，不需要 torch/ms-swift。

    这是 §1.3（计算与设施分离）的直接证据——映射逻辑可以在没有 GPU/没有 ms-swift
    的环境里被验证。
    """
    import sys

    class _AbsentConfig:
        """任意字段都返回 None 的哑配置（模拟"什么都没配"的 msswift.megatron 段）。"""

        def __getattr__(self, name: str) -> None:
            return None

    stub = SimpleNamespace(
        # 配置级字段（`GraspoConfig.effective_tuner_type` 是 schema 上的真实 property，
        # 由 `9bc1cf1` 引入）：映射层的 `--tuner_type` 唯一取值来源。哑桩必须显式给出，
        # 否则映射层读不到它——这不是"映射层要求太重"，而是配置契约的一部分。
        effective_tuner_type="lora",
        model=SimpleNamespace(
            model_path="m",
            trust_remote_code=True,
            torch_dtype="bfloat16",
            chat_template_kwargs={},
            gradient_checkpointing=True,
            attn_implementation=None,
        ),
        data=SimpleNamespace(train_path="d", max_prompt_length=1024),
        lora=SimpleNamespace(r=8, alpha=16, dropout=0.0, target_modules=None, adapter_path=None),
        training=SimpleNamespace(
            output_dir="",
            run_name="r",
            seed=1,
            max_epochs=1,
            learning_rate=1e-5,
            weight_decay=0.0,
            max_grad_norm=1.0,
            gradient_accumulation_micro_batches=1,
            save_steps=-1,
            save_checkpoint_every_epoch=False,
            max_new_tokens=64,
            rollout_group_size=2,
            temperature=1.0,
            top_p=1.0,
        ),
        msswift=SimpleNamespace(
            device_map=None,
            deepspeed=None,
            zero_hpz_partition_size=None,
            deepspeed_autotp_size=None,
            fsdp=None,
            sequence_parallel_size=2,
            rope_scaling=None,
            max_model_len=None,
            max_pixels=None,
            packing=False,
            padding_free=False,
            use_liger_kernel=False,
            use_logits_to_keep=False,
            use_vllm=None,
            vllm_mode=None,
            attn_impl=None,
            per_device_train_batch_size=None,
            megatron=_AbsentConfig(),
        ),
    )
    for name in ("torch", "swift"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    argv = graspo_to_ms_swift_argv(stub, stage="sft", dataset_path="ds", output_dir="out")

    assert _value_of(argv, "--sequence_parallel_size") == "2"
    assert "torch" not in sys.modules
    assert "swift" not in sys.modules


# ── list 型参数的序列化形态（E2b 实测缺陷 2 的回归锁）────────────────────────


def test_list_value_is_serialized_as_space_separated_multi_value():
    """list → ``--flag a b``（空格分隔多值），**不是** JSON 字面量。

    E2b 实测：``lora.target_modules: [all-linear]`` 曾被写成
    ``--target_modules '["all-linear"]'``，ms-swift 的 ``List[str]`` 字段把整段 JSON
    当成单个元素 → ``Target modules {'["all-linear"]'} not found``。
    """
    from graspo.flow.msswift._config_mapping import _flag

    assert _flag("target_modules", ["q_proj", "v_proj"]) == [
        "--target_modules",
        "q_proj",
        "v_proj",
    ]
    assert _flag("target_modules", ["all-linear"]) == ["--target_modules", "all-linear"]
    argv = _to_argv(_config())
    assert json.dumps(["all-linear"]) not in argv
    assert not any(token.startswith('["') for token in argv)


def test_single_value_and_dict_and_bool_serialization_are_unchanged():
    """标量/``dict``/``bool`` 的形态必须与修复前逐字一致（不做无谓变更）。"""
    from graspo.flow.msswift._config_mapping import _flag

    assert _flag("lora_rank", 8) == ["--lora_rank", "8"]
    assert _flag("learning_rate", 1e-5) == ["--learning_rate", "1e-05"]
    assert _flag("packing", True) == ["--packing", "true"]
    assert _flag("packing", False) == ["--packing", "false"]
    assert _flag("rope_scaling", {"type": "yarn"}) == [
        "--rope_scaling",
        '{"type": "yarn"}',
    ]


def test_empty_list_is_treated_as_not_provided():
    """空列表 = 没有值可传，不能 emit ``--flag``（会被 nargs='+' 判为缺参）。"""
    from graspo.flow.msswift._config_mapping import _flag

    assert _flag("target_modules", []) == []


def test_lora_target_modules_round_trips_as_multi_value():
    """``lora.target_modules`` 真值透传：list → 空格分隔，不再是单元素 JSON 串。"""
    argv = graspo_to_ms_swift_argv(
        GraspoConfig.model_validate(
            {
                "backend": "msswift",
                "model": {"model_path": "m"},
                "data": {"train_path": "d", "max_prompt_length": 2048},
                "lora": {"target_modules": ["all-linear"]},
            }
        ),
        stage="sft",
        dataset_path="/tmp/ds.jsonl",
        output_dir="/tmp/out",
    )
    index = argv.index("--target_modules")

    assert argv[index + 1] == "all-linear"
    # 紧跟其后的下一个 token 必须是另一个 flag（说明没有多吞参数）
    assert argv[index + 2].startswith("--")


# ── 组合约束的前置校验（E2b 缺陷 3/4：防御性报错，**不是能力修复**）────────────


def test_fp8_param_gather_without_fp8_mode_fails_before_launch():
    """E2b 缺陷 3：`fp8_param_gather` 单独打开 → 启动前给可读报错，而不是 ChildFailedError。"""
    from graspo.flow.msswift._config_mapping import validate_combinations

    config = _config(megatron={"fp8_param_gather": True})

    with pytest.raises(ValueError, match="fp8_param_gather"):
        validate_combinations(config)


def test_deepspeed_autotp_with_lora_fails_before_launch():
    """E2b 缺陷 4：AutoTP 官方仅支持全参，graspo 恒为 LoRA → 启动前拦下。"""
    from graspo.flow.msswift._config_mapping import validate_combinations

    config = _config(deepspeed_autotp_size=2)

    with pytest.raises(ValueError, match="deepspeed_autotp_size"):
        validate_combinations(config)


def test_validate_combinations_accepts_the_defaults():
    """默认配置（不碰这两项）必须原样通过——校验不得变成无差别拦截。"""
    from graspo.flow.msswift._config_mapping import validate_combinations

    validate_combinations(_config())
    validate_combinations(_config(megatron={"fp8_param_gather": False}))


# ── RoPE 键名适配（E2b 缺陷 1）────────────────────────────────────────────────


def test_suggested_rope_parameters_renames_the_legacy_key():
    """``type`` → ``rope_type``（transformers 5.x 的新键）；不发明取值。"""
    from graspo.flow.msswift._rope_compat import suggested_rope_parameters

    assert suggested_rope_parameters("yarn") == {"rope_type": "yarn"}
    assert suggested_rope_parameters({"type": "dynamic", "factor": 2.0}) == {
        "rope_type": "dynamic",
        "factor": 2.0,
    }
    assert suggested_rope_parameters('{"type": "linear"}') == {"rope_type": "linear"}
    assert suggested_rope_parameters(None) is None


def test_rope_scaling_is_mapped_once_with_the_new_key():
    """`--rope_scaling` 必须**只出现一次**，且取值是新键 JSON（否则最后一个赢，来源不可追）。"""
    argv = _to_argv(_config(rope_scaling="yarn", max_model_len=131072))

    assert argv.count("--rope_scaling") == 1
    assert json.loads(_value_of(argv, "--rope_scaling")) == {"rope_type": "yarn"}


def test_rope_compatibility_context_is_a_noop_without_rope_scaling():
    """`rope_scaling` 为 None 时**不装任何补丁**（未使用该参数时行为必须逐字不变）。"""
    from graspo.flow.msswift._rope_compat import rope_parameters_compatible

    try:
        from transformers import PreTrainedModel  # noqa: F401
    except ImportError:
        pytest.skip("transformers is not installed in this environment")

    import inspect

    from transformers import PreTrainedModel

    # 注意：`PreTrainedModel.from_pretrained` 每次访问都返回**新的** bound method 对象，
    # 不能用 `is` 比较（实测踩过）；判据是类上的**静态 descriptor**。
    before = inspect.getattr_static(PreTrainedModel, "from_pretrained")
    with rope_parameters_compatible(None):
        assert inspect.getattr_static(PreTrainedModel, "from_pretrained") is before, (
            "rope_scaling=None 时不得装任何补丁"
        )
    assert inspect.getattr_static(PreTrainedModel, "from_pretrained") is before


def test_rope_compatibility_context_restores_from_pretrained():
    """补丁必须**保持 classmethod 语义**并且退出即还原。

    E3 实测踩过：第一版包装器漏了 `@classmethod`，于是 `cls` 被绑定成第一个位置参数
    （模型类本身），真实报错却是 `HFValidationError: Repo id ... '<class ...Qwen3ForCausalLM>'`
    ——错误信息与真实原因相隔极远。
    """
    from graspo.flow.msswift._rope_compat import rope_parameters_compatible

    try:
        from transformers import PreTrainedModel
    except ImportError:
        pytest.skip("transformers is not installed in this environment")

    import inspect

    before = inspect.getattr_static(PreTrainedModel, "from_pretrained")
    with rope_parameters_compatible("yarn"):
        patched = inspect.getattr_static(PreTrainedModel, "from_pretrained")
        assert patched is not before, "rope_scaling 非 None 时必须装补丁"
        assert isinstance(patched, classmethod), "补丁必须仍是 classmethod（cls 由描述符注入）"
    assert inspect.getattr_static(PreTrainedModel, "from_pretrained") is before, (
        "退出必须还原原 descriptor"
    )
