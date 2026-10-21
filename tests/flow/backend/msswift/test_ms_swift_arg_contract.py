"""契约测试：映射产出的**每一个** ``--flag`` 都必须是 ms-swift 参数类的合法字段。

**为什么需要这条测试（它是这次实测踩到的坑）**

第一版映射把 ``--trust_remote_code`` / ``--chat_template_kwargs`` / ``--use_vllm``
透传给了 ms-swift，而 ms-swift 4.5.3 的 ``SftArguments`` 里根本没有前两个、
``use_vllm`` 只存在于 RLHF 参数类——结果 SFT 冒烟在参数解析阶段就失败
（``ValueError: remaining_argv: [...]``）。Graspo 侧的单元测试当时是绿的，
因为它们只断言"我们写了什么"，没有断言"ms-swift 认不认"。

本条测试把"ms-swift 认不认"变成可自动检查的判据：**参数名集合 ⊆ 目标参数类字段集合**。
它需要真实安装的 ms-swift，因此无 ms-swift 的环境 skip（不是静默通过）。

覆盖范围：SFT 与 RL(GRPO) 两个阶段 × 默认配置 + 一个"全字段打开"的配置。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from graspo.core.lora import LORA_TARGET_MODULES_REGEX
from graspo.core.schema import GraspoConfig

swift = pytest.importorskip("swift", reason="ms-swift is required for the argument contract test")

_ALL_SHARDING_CONFIG = {
    "backend": "msswift",
    "model": {"model_path": "m", "attn_implementation": "flash_attn"},
    "data": {"train_path": "d", "max_prompt_length": 4096},
    # ★ 目标口径（主席 2026-10-21）：两个后端统一用**同一条真实 HF 模块名正则**，
    #   list / preset 形态已 fail-closed（`core/lora.py:114`）。
    "lora": {"target_modules": LORA_TARGET_MODULES_REGEX},
    "msswift": {
        "sequence_parallel_size": 2,
        "deepspeed": "zero3",
        "device_map": "auto",
        "zero_hpz_partition_size": 8,
        "deepspeed_autotp_size": 2,
        "rope_scaling": "dynamic",
        "max_model_len": 32768,
        "packing": True,
        "padding_free": True,
        "use_liger_kernel": True,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "megatron": {
            "global_batch_size": 32,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "sequence_parallel": True,
            "context_parallel_size": 2,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 2,
            "microbatch_group_size_per_vp_stage": 4,
            "fp8_param_gather": True,
            "fp4_param_gather": True,
            "use_megatron_fsdp": True,
            "data_parallel_sharding_strategy": "optim_grads",
            "strict_fsdp_dtensor_load": True,
            "data_sharding": True,
            "data_parallel_random_init": True,
            "overlap_grad_reduce": True,
            "use_distributed_optimizer": True,
            "tp_comm_overlap": True,
            "overlap_p2p_comm": True,
            "align_param_gather": True,
            "pipeline_model_parallel_layout": "Et*3|(tt|)*29,m|L",
            "decoder_first_pipeline_num_layers": 1,
            "decoder_last_pipeline_num_layers": 1,
            "cp_comm_type": "a2a+p2p",
            "cp_partition_mode": "zigzag",
            "sequence_packing_scheduler": "dp_balanced",
        },
    },
}


def _argument_fields(cls: type) -> set[str]:
    """收集参数类（含其 dataclass 基类）的全部字段名。"""
    names: set[str] = set()
    for base in cls.__mro__:
        if dataclasses.is_dataclass(base):
            names |= {field.name for field in dataclasses.fields(base)}
    return names


def _flags(argv: list[str]) -> list[str]:
    return [token[2:] for token in argv if token.startswith("--")]


@pytest.mark.parametrize("stage", ["sft", "rlhf"])
@pytest.mark.parametrize("config_data", [{}, _ALL_SHARDING_CONFIG])
def test_every_mapped_flag_is_a_valid_ms_swift_argument(stage, config_data):
    from graspo.flow.backend.msswift._config_mapping import graspo_to_ms_swift_argv

    # 两个 stage 分支必须落到同一个局部名（后续断言只写一次）。
    # 上游类名是 CamelCase，别名只能取小写 ⇒ 逐条豁免 N813：别名是**局部变量**，
    # 不是公开导出（改动导入别名会改变两个分支的可读性，且无收益）。
    if stage == "sft":
        from swift.arguments import SftArguments as _args_cls  # noqa: N813
    else:
        from swift.arguments import RLHFArguments as _args_cls  # noqa: N813

    config = GraspoConfig.model_validate(config_data)
    argv = graspo_to_ms_swift_argv(
        config, stage=stage, dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )

    valid = _argument_fields(_args_cls)
    unknown = sorted(flag for flag in _flags(argv) if flag not in valid)

    assert not unknown, (
        f"stage={stage}: ms-swift {_args_cls.__name__} does not accept {unknown} — "
        "these flags would make ms-swift's parse_args reject the whole run."
    )


@pytest.mark.parametrize(
    "args_path,args_name",
    [
        ("swift.megatron.arguments.sft_args", "MegatronSftArguments"),
        ("swift.megatron.arguments.rlhf_args", "MegatronRLHFArguments"),
    ],
)
def test_megatron_passthrough_is_valid_for_the_megatron_channel(args_path, args_name):
    """Megatron 通道：MG1-MG11 的可配置透传向量必须是该通道参数类的合法字段。

    实测（T1，4.5.3）：这些参数名**不是**标准 ``SftArguments`` / ``RLHFArguments``
    的字段——混进标准通道会让 ms-swift 直接拒绝整次运行。因此 Megatron 参数由
    ``megatron_passthrough_argv`` 单独产出，并且只在 Megatron 通道使用。
    """
    from graspo.flow.backend.msswift._config_mapping import (
        graspo_to_ms_swift_argv,
        megatron_passthrough_argv,
    )

    module = pytest.importorskip(args_path, reason="megatron extras are optional")
    args_cls = getattr(module, args_name, None)
    if args_cls is None:  # pragma: no cover - 极端版本差异
        pytest.skip(f"{args_name} not exposed by this ms-swift build")

    config = GraspoConfig.model_validate(_ALL_SHARDING_CONFIG)
    standard_argv = graspo_to_ms_swift_argv(
        config, stage="sft", dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )
    megatron_argv = megatron_passthrough_argv(config)

    # §8.2 的 11 行在本配置下都有取值 → 必须全部出现
    expected = {
        "global_batch_size",
        "data_sharding",
        "data_parallel_random_init",
        "overlap_grad_reduce",
        "use_distributed_optimizer",
        "use_megatron_fsdp",
        "data_parallel_sharding_strategy",
        "strict_fsdp_dtensor_load",
        "tensor_model_parallel_size",
        "tp_comm_overlap",
        "pipeline_model_parallel_size",
        "overlap_p2p_comm",
        "align_param_gather",
        "pipeline_model_parallel_layout",
        "decoder_first_pipeline_num_layers",
        "decoder_last_pipeline_num_layers",
        "sequence_parallel",
        "context_parallel_size",
        "cp_comm_type",
        "cp_partition_mode",
        "sequence_packing_scheduler",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "microbatch_group_size_per_vp_stage",
        "fp8_param_gather",
        "fp4_param_gather",
    }
    assert expected <= set(_flags(megatron_argv))
    assert not (expected & set(_flags(standard_argv)))

    valid = _argument_fields(args_cls)
    unknown = sorted(flag for flag in _flags(megatron_argv) if flag not in valid)
    assert not unknown, f"{args_name} does not accept {unknown}"


def test_deepspeed_flags_are_rejected_by_the_megatron_channel():
    """Megatron 通道没有 DeepSpeed 参数（实测）——它们只在标准通道合法。

    这条断言把"M3 透传路径按通道分流"这件事钉死：同一份配置，标准通道发
    DeepSpeed/FSDP 参数，Megatron 通道发 MG1-MG11；两边的参数集合不混。
    """
    from graspo.flow.backend.msswift._config_mapping import megatron_passthrough_argv

    megatron_argv = megatron_passthrough_argv(GraspoConfig.model_validate(_ALL_SHARDING_CONFIG))

    for flag in (
        "--deepspeed",
        "--zero_hpz_partition_size",
        "--deepspeed_autotp_size",
        "--fsdp",
        "--sequence_parallel_size",
    ):
        assert flag not in megatron_argv


def test_rlhf_only_fields_are_not_sent_to_sft():
    """``use_vllm`` 只存在于 RLHF 参数类：SFT 阶段必须被挡在映射之外（实测教训）。"""
    from graspo.flow.backend.msswift._config_mapping import graspo_to_ms_swift_argv

    config = GraspoConfig.model_validate(_ALL_SHARDING_CONFIG)
    sft_argv = graspo_to_ms_swift_argv(
        config, stage="sft", dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )
    rlhf_argv = graspo_to_ms_swift_argv(
        config, stage="rlhf", dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )

    assert "--use_vllm" not in sft_argv
    assert "--use_vllm" in rlhf_argv


def test_rlhf_stage_pins_the_grpo_rlhf_type():
    """RLHF 参数类的 ``rlhf_type`` 默认是 ``dpo``：不显式指定会按 DPO 编码数据集。

    实测（4.5.3）：默认值下 GRPO 的提示词数据会被 DPO 编码器拒绝
    （``ValueError: inputs.rejected is None``），6 条样本全被过滤 → 空数据集。
    """
    from graspo.flow.backend.msswift._config_mapping import graspo_to_ms_swift_argv

    config = GraspoConfig.model_validate({"backend": "msswift"})
    rlhf_argv = graspo_to_ms_swift_argv(
        config, stage="rlhf", dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )

    assert _value_of(rlhf_argv, "--rlhf_type") == "grpo"


def _value_of(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


# ── 目标口径的**取值形态**契约（E2b 实测缺陷 2 的后续形态）────────────────────


def test_target_regex_is_passed_as_one_token_and_fills_peft_target_modules():
    """目标只走 ``--target_regex``，且必须是**单 token**（形态错误同样致命）。

    E2b 实测过 ``--target_modules '["all-linear"]'``：JSON 字面量被 ``nargs='+'`` 当成
    单个元素，ms-swift 得到 ``['["all-linear"]']`` ⇒ ``Target modules … not found``。
    主席 2026-10-21 的口径变更把目标通道收敛为**一条真实 HF 模块名正则**，因此本测试
    把映射产出的 argv 交给 **ms-swift 自己的** ``swift.utils.parse_args``，断言：

    ① 有且只有一条 ``--target_regex <正则>``（正则内无空格 ⇒ 不会被拆成多 token）；
    ② 绝不出现 ``--target_modules``（旧 list 通道已删除，⛔ 不留隐式兼容）；
    ③ 解析后 ``target_regex`` = 该常量，且 ``target_modules`` 被它填成同一串
       （`swift/arguments/tuner_args.py:126-127` 的 ``__post_init__`` 行为 =
       ms-swift 官方正则入口）。

    无 ms-swift / 模型目录不在本机时 skip（不静默通过）。
    """
    from swift.arguments import SftArguments
    from swift.utils import parse_args

    from graspo.flow.backend.msswift._config_mapping import graspo_to_ms_swift_argv

    # ms-swift 的 parse_args 会校验 --model 路径存在（本测试要过一次真实解析）。
    # 模型目录不在本机时 skip（不静默通过）。
    _qwen_path = "/models/Qwen3-8B"
    if not Path(_qwen_path).is_dir():
        pytest.skip(f"{_qwen_path} is not available in this environment")

    config = GraspoConfig.model_validate(
        {
            "backend": "msswift",
            "model": {"model_path": _qwen_path},
            "data": {"train_path": "/tmp/ds.jsonl", "max_prompt_length": 4096},
            "lora": {"target_modules": LORA_TARGET_MODULES_REGEX},
            "training": {"output_dir": "/tmp/out"},
        }
    )
    argv = graspo_to_ms_swift_argv(
        config, stage="sft", dataset_path="/tmp/ds.jsonl", output_dir="/tmp/out"
    )

    assert " " not in LORA_TARGET_MODULES_REGEX, "正则含空格 ⇒ 会被 nargs='+' 拆成多 token"
    assert argv.count("--target_regex") == 1, argv
    assert argv[argv.index("--target_regex") + 1] == LORA_TARGET_MODULES_REGEX
    assert "--target_modules" not in argv, "旧 list 通道必须已删除（⛔ 不留隐式兼容）"

    parsed, remaining = parse_args(SftArguments, argv)

    assert remaining == [], f"ms-swift rejected {remaining}"
    assert parsed.target_regex == LORA_TARGET_MODULES_REGEX
    assert parsed.target_modules == LORA_TARGET_MODULES_REGEX, (
        "ms-swift 的 target_regex 必须落成 PEFT 的正则分支（str，而不是 list）"
    )


def test_multi_value_parsing_matches_the_three_observed_forms():
    """把"哪种形态能过"写成可执行证据（三种形态由本测试直接调用 ms-swift 的
    ``swift.utils.parse_args`` 复现，不依赖任何仓库外的探针脚本）。

    ★ 存档性质：这是 **ms-swift 自身**对 ``--target_modules`` 三种形态的解析行为
    （2026-10 实测）。本轮口径变更后 Graspo 已不再发 ``--target_modules``
    （目标只走 ``--target_regex``，见上一条测试）；本表留作上游行为存档。

    ==============================  =================================
    argv 片段                        ms-swift 解析结果
    ==============================  =================================
    ``--target_modules a b``         ``['a', 'b']``            ✅
    ``--target_modules '["a","b"]'`` ``['["a","b"]']``          ❌
    ``--target_modules a,b``         ``['a,b']``               ❌
    ==============================  =================================
    """
    from swift.arguments import SftArguments
    from swift.utils import parse_args

    _qwen_path = "/models/Qwen3-8B"
    if not Path(_qwen_path).is_dir():
        pytest.skip(f"{_qwen_path} is not available in this environment")

    base = [
        "--model",
        _qwen_path,
        "--dataset",
        "/tmp/ds.jsonl",
        "--output_dir",
        "/tmp/out",
    ]
    space, _ = parse_args(SftArguments, base + ["--target_modules", "q_proj", "v_proj"])
    jsonish, _ = parse_args(SftArguments, base + ["--target_modules", '["q_proj","v_proj"]'])
    comma, _ = parse_args(SftArguments, base + ["--target_modules", "q_proj,v_proj"])

    assert list(space.target_modules) == ["q_proj", "v_proj"]
    assert list(jsonish.target_modules) == ['["q_proj","v_proj"]']
    assert list(comma.target_modules) == ["q_proj,v_proj"]
