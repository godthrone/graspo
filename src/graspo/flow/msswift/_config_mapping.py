"""graspo 配置 → ms-swift 训练参数的**纯映射**（计算层，零 IO、零 ms-swift 导入）。

**职责边界**

- **本模块**：把 ``GraspoConfig``（含 ``msswift`` 段）翻译成 ms-swift 的参数向量
  （``--key value`` 形式）与 launcher 环境变量。是"graspo 配置 ↔ ms-swift 参数"
  的**唯一映射点**（宪法 §1.4 单一真相源）。
- **不负责**：导入 ms-swift、读写文件、启动训练、构造数据集。这些都在
  ``flow/msswift/{sft_trainer,trainer}.py``（设施层）。

**为什么是纯映射而不是 ``**kwargs`` 透传**（宪法 §2.2 显式即防呆）

每一个 graspo 字段到 ms-swift 参数的对应关系都写死在 ``_BASE_MAPPING`` 等表里，
读代码即可回答"这个值从哪来、到哪去"。不做 ``**config`` 展开，也不做环境变量
fallback 链——用户改了参数，行为一定随之变化，不存在"改了没用"的假配置
（§7.2）。``msswift`` 段的字段名与 ms-swift 官方参数名逐字对应，透传段因此不需要
一张猜名字的表。

**graspo 字段 → ms-swift 的"不可映射"清单（显式声明，不假装支持）**

T1 复验（ms-swift 4.5.3）实测：``trust_remote_code`` 与全局 ``chat_template_kwargs``
**不是** ms-swift 参数类的字段（传了会被 ``parse_args`` 直接拒绝）。因此：

- ``model.trust_remote_code``：msswift 路径不透传（ms-swift 按模型类型自行决定）；
  该字段仍是 native 后端的真相源，由 :func:`native_only_fields` 报告给调用方。
- ``model.chat_template_kwargs``：ms-swift 侧是**数据集列**（per-sample），
  因此由 ``dataset.py`` 写进每一行，而不是命令行参数。
- ``lora.target_preset``：是 native 侧基于模块名的预设（``core/lora.py``），
  ms-swift 的 ``target_modules`` 是另一套（regex / ``all-linear``）——不做猜测式转换，
  由 :func:`native_only_fields` 提示用户显式设置 ``lora.target_modules``。

**两处显式优先级**（不隐藏、不静默）

- ``--attn_impl``：``msswift.attn_impl`` 非 None 时优先，否则取
  ``model.attn_implementation``（与 native 后端共用的字段，避免出现"假配置"）。
- ``--max_model_len`` / ``--packing`` / ``--padding_free`` 只由 ``msswift`` 段提供
  （native 后端没有对应语义），不存在第二来源。

**argv 而不是 ``SftArguments(**kwargs)``**

ms-swift 的参数类依赖它自己的 ``parse_args`` 做别名/默认值/嵌套解析，直接构造
``SftArguments`` 需要复刻这套逻辑。因此本模块产出 argv，由调用方交给
``swift.pipelines.sft_main`` / ``RLHFArguments``——**那是 ms-swift 的 Python API
入口**（ms-swift 自己的 ``swift sft`` CLI 也是调用同一个函数），不经过子进程、
不经过 shell，不依赖 ms-swift 能读懂 graspo 的 YAML。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from graspo.flow.msswift._rope_compat import suggested_rope_parameters

#: 训练阶段：SFT 与 RL(GRPO) 的 ms-swift 参数集不同（如 max_completion_length 仅 RL 有）。
Stage = Literal["sft", "rlhf"]


def _flag(name: str, value: Any) -> list[str]:
    """把一个 graspo 配置值写成 ms-swift CLI 的参数片段。

    三种形态，逐一显式（§2.2 显式即防呆）：

    - ``bool`` → ``--name true`` / ``--name false``（ms-swift 用字面量，不是 ``True``）。
    - ``list`` → ``--name a b c``（**空格分隔多值**，第 1 项紧跟 flag，其余各占一个 arg）。
    - ``dict`` → ``--name '{"k": v}'``（JSON 字符串；ms-swift 侧多数 dict 型字段走它自己的
      ``json_parse_to_dict``，只有单张量形态合法）。

    Args:
        name: ms-swift 参数名（不含前导 ``--``）。
        value: 配置值；调用方保证不是 ``None``（见 :func:`_extend`）。

    Returns:
        参数片段；``list`` 时长度 = ``1 + len(value)``。

    **为什么 list 不能走 JSON**（E2b 实测缺陷，2026-09-16）

    第一版对 ``dict``/``list`` 一律 ``json.dumps``，于是 ``lora.target_modules: [all-linear]``
    被写成 ``--target_modules '["all-linear"]'``。``target_modules`` 在 ms-swift 里是
    ``List[str]``，HfArgumentParser 的 ``nargs='+'`` 会把整个 JSON 字面量当成**单个**元素
    → ``Target modules {'["all-linear"]'} not found``——多值参数语义整体失效。

    探针实证（``swift.utils.parse_args``，ms-swift 自己的参数解析入口）：

    ==========================  ==========================================
    argv 片段                    解析结果
    ==========================  ==========================================
    ``--target_modules a b``     ``['a', 'b']`` ✅
    ``--target_modules '["a","b"]'``  ``['["a","b"]']`` ❌（单元素字面量）
    ``--target_modules a,b``     ``['a,b']`` ❌（不按逗号切分）
    ``--target_modules a``       ``['a']`` ✅
    ==========================  ==========================================

    因此：**只有列表按空格展开**，标量与 dict 行为保持不变（dict 仍需 JSON，因为
    ms-swift 的 ``rope_scaling`` 等字段就是拿字符串去 ``json.loads``）。
    """
    if isinstance(value, bool):
        return [f"--{name}", "true" if value else "false"]
    if isinstance(value, list):
        # 空列表 = "没有值可传"：emit `--name` 会被 ms-swift 的 nargs='+' 判为缺参，
        # strict 解析后 remaining_argv 非空 → 整次运行被拒（实测口径）。
        # 因此空列表等价于"未提供"（§2.2 None 语义），而不是产出一个畸形 flag。
        if not value:
            return []
        return [f"--{name}", *(str(item) for item in value)]
    if isinstance(value, dict):
        return [f"--{name}", json.dumps(value, ensure_ascii=False)]
    return [f"--{name}", str(value)]


def _extend(argv: list[str], name: str, value: Any) -> None:
    """``value is None`` 表示"未提供"，不透传（§2.2 None 语义）。"""
    if value is None:
        return
    argv.extend(_flag(name, value))


def _rope_scaling_json(value: Any) -> str | None:
    """``rope_scaling`` → JSON 字符串（键名换成 transformers 5.x 的 ``rope_type``）。

    归一化的单一真相源是 ``_rope_compat.suggested_rope_parameters``（纯函数、零重型
    依赖），此处只做序列化——两处不各写一份键名映射（§1.4）。
    """
    resolved = suggested_rope_parameters(value)
    return None if resolved is None else json.dumps(resolved, ensure_ascii=False)


#: ``msswift`` 段 → ms-swift 参数名的透传表（字段名逐字对应，仅列出需要改名的项）。
#: 其余同名字段由 :func:`msswift_passthrough_argv` 逐个显式列出——不做 ``__dict__`` 展开。
_MSSWIFT_SCALAR_PASSTHROUGH: tuple[str, ...] = (
    # S2-S7 分片类型（§8.1）
    "device_map",
    "deepspeed",
    "zero_hpz_partition_size",
    "deepspeed_autotp_size",
    "fsdp",
    "sequence_parallel_size",
    # 长文附项
    "rope_scaling",
    "max_model_len",
    "packing",
    "padding_free",
    "use_liger_kernel",
    "use_logits_to_keep",
    "per_device_train_batch_size",
)

#: 只在 RL(GRPO) 阶段合法的 ``msswift`` 字段（SFT 参数类里不存在，传了就是非法参数）。
_MSSWIFT_RLHF_ONLY: tuple[str, ...] = ("use_vllm", "vllm_mode", "num_iterations")

#: Megatron 段（§8.2 MG1-MG11）→ ms-swift ``MegatronArguments`` 参数名（同名）。
#: 显式列出而不是遍历 dataclass：新增字段时必须显式加进来，不会被"顺手"透传。
_MEGATRON_PASSTHROUGH: tuple[str, ...] = (
    # MG1
    "global_batch_size",
    "data_sharding",
    "data_parallel_random_init",
    "overlap_grad_reduce",
    # MG2
    "use_distributed_optimizer",
    # MG3
    "use_megatron_fsdp",
    "data_parallel_sharding_strategy",
    "strict_fsdp_dtensor_load",
    # MG4
    "tensor_model_parallel_size",
    "tp_comm_overlap",
    # MG5
    "pipeline_model_parallel_size",
    "overlap_p2p_comm",
    "align_param_gather",
    "pipeline_model_parallel_layout",
    "decoder_first_pipeline_num_layers",
    "decoder_last_pipeline_num_layers",
    # MG6
    "sequence_parallel",
    # MG7
    "context_parallel_size",
    "cp_comm_type",
    "cp_partition_mode",
    "sequence_packing_scheduler",
    # MG8 / MG9
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    # MG10
    "virtual_pipeline_model_parallel_size",
    "microbatch_group_size_per_vp_stage",
    # MG11
    "fp8_param_gather",
    "fp4_param_gather",
)


def resolve_attn_impl(config: Any) -> str | None:
    """ms-swift ``--attn_impl`` 的取值（显式优先级，见模块 docstring）。"""
    override = config.msswift.attn_impl
    if override is not None:
        return str(override)
    legacy = config.model.attn_implementation
    return None if legacy is None else str(legacy)


def resolve_rope_scaling(config: Any) -> str | None:
    """ms-swift ``--rope_scaling`` 的取值：**transformers 5.x 的新键名**（JSON 字符串）。

    **为什么在映射层就换成新键**（E2b 实测缺陷 1，2026-09-16）

    ``msswift.rope_scaling: yarn`` 原先按字面量透传，ms-swift 把它写成旧格式
    ``{'type': 'yarn', ...}`` 塞进 config；transformers 5.12 读的是
    ``config.rope_parameters["rope_type"]`` → ``KeyError: 'rope_type'``，模型加载即崩
    （E2b 的 13 档 rope_scaling **全档 0 成功**）。

    这里只改**键名**（``type`` → ``rope_type``），不发明取值：

    - ms-swift ``model_args.py::_init_rope_scaling`` 本来就**读**新键
      （``rope_scaling.get('rope_type', rope_scaling.get('type', 'default'))``），
      因此新键对 ms-swift 完全兼容；它随后补 ``factor`` /
      ``original_max_position_embeddings``（保持 ms-swift 的语义，graspo 不重算）。
    - 值仍是``--rope_scaling`` 的合法形态（字符串）；JSON 字符串由 ms-swift 自己的
      ``json_parse_to_dict`` 解析，与 ``dict`` 型 config 字段同一机制。

    ``None`` = 未提供（不透传）。模型加载时的**兜底适配**见
    ``_rope_compat.rope_parameters_compatible``（ms-swift 仍会覆写 config 的
    ``rope_scaling`` 属性，那一步需要边界补丁，见该模块 docstring）。
    """
    return _rope_scaling_json(config.msswift.rope_scaling)


def msswift_passthrough_argv(
    config: Any,
    *,
    include_megatron: bool = False,
    stage: Stage = "rlhf",
) -> list[str]:
    """``msswift`` 段的透传参数（标准路径 + 长文），供测试与 :func:`graspo_to_ms_swift_argv` 共用。

    Args:
        config: ``GraspoConfig`` 实例。
        include_megatron: 是否追加 Megatron 路径参数（§8.2 MG1-MG11）。
        stage: ``"sft"`` 时不透传 RL 专有字段（``use_vllm`` 等）——ms-swift 的
            ``SftArguments`` 里没有这些字段，传了会直接被拒绝（实测）。

    Returns:
        ``["--sequence_parallel_size", "2", ...]``；未提供的字段（``None``）不出现。
    """
    section = config.msswift
    argv: list[str] = []
    for name in _MSSWIFT_SCALAR_PASSTHROUGH:
        if name == "rope_scaling":
            # ``rope_scaling`` 需要键名适配（transformers 5.x 的 ``rope_type``），
            # 由下面的 ``resolve_rope_scaling`` 统一处理——**只透传一次**，
            # 不在循环里再发一遍（重复 flag 会让取值来源变成"最后一个赢"，§1.4）。
            continue
        value = getattr(section, name)
        # sequence_parallel_size 默认 1 也要透传：ms-swift 的默认值可能随版本变化，
        # 显式写出才能保证"配置即真相"（§1.4）。
        _extend(argv, name, value)
    _extend(argv, "attn_impl", resolve_attn_impl(config))
    _extend(argv, "rope_scaling", resolve_rope_scaling(config))
    if stage == "rlhf":
        for name in _MSSWIFT_RLHF_ONLY:
            value = getattr(section, name)
            if name == "num_iterations" and value is None:
                # graspo 的默认优化轮次是 1（native 侧已删除 optimize_iterations_per_step）
                value = 1
            _extend(argv, name, value)
    if include_megatron:
        argv.extend(megatron_passthrough_argv(config))
    return argv


def megatron_passthrough_argv(config: Any) -> list[str]:
    """Megatron 路径的透传参数（§8.2 MG1-MG11）。

    **只能用于 Megatron 启动通道**（``swift.megatron.megatron_sft_main`` /
    ``megatron_rlhf_main``，参数类 ``MegatronSftArguments`` / ``MegatronRLHFArguments``）。
    实测（T1，4.5.3）：这些参数名**不是** ``SftArguments`` / ``RLHFArguments`` 的字段，
    混进标准通道会让 ms-swift 的参数解析直接拒绝整次运行——因此本函数与标准映射分开。

    **本轮的边界（如实声明）**：本函数产出的是 Megatron 通道的**分片参数向量**
    （MG1-MG11 逐项与 ``MegatronSftArguments`` 字段同名，由契约测试锁定）。
    Megatron 通道的训练超参词汇与标准通道**不同**（Megatron 用
    ``--lr`` / ``--clip_grad`` / ``--global_batch_size`` / ``--train_iters`` 等，
    而标准通道用 ``--learning_rate`` / ``--max_grad_norm`` / ``--num_train_epochs``），
    且 DeepSpeed 系列参数（``deepspeed`` / ``zero_hpz_partition_size`` /
    ``deepspeed_autotp_size``）在该通道下**不存在**。因此"拼出一条可直接启动的
    Megatron 命令"需要一张 Megatron 专属超参映射表——按 D8 A 档（Megatron 仅透传
    + 启动冒烟）与 E2a 范围，该映射表留待 E2b/后续；本轮交付的是
    **"MG1-MG11 全部可配置、参数名与 ms-swift 逐字一致"** 这一可验证的透传路径。
    """
    megatron = config.msswift.megatron
    argv: list[str] = []
    for name in _MEGATRON_PASSTHROUGH:
        _extend(argv, name, getattr(megatron, name))
    return argv


def launcher_env(config: Any) -> dict[str, str]:
    """S1 数据并行 DDP 的 launcher 环境变量（ms-swift 侧由 ``swift.cli.main`` 读取）。

    **T1 复验结论（ms-swift 4.5.3）**：``NPROC_PER_NODE`` / ``NNODES`` /
    ``NODE_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT`` 不是 ``TrainArguments``
    字段，而是 launcher 环境变量（``swift.cli.main.get_torchrun_args`` 把它们转成
    ``torchrun`` 参数）。graspo 侧同样以环境变量承载这一层，值来自
    ``msswift`` 段——用户仍然只在 YAML 里配置（§7.1：环境变量不是配置来源，
    这里只做对第三方接口的适配）。

    未提供的字段不进环境（不覆盖既有值）。
    """
    section = config.msswift
    mapping = {
        "nproc_per_node": "NPROC_PER_NODE",
        "nnodes": "NNODES",
        "node_rank": "NODE_RANK",
        "master_addr": "MASTER_ADDR",
        "master_port": "MASTER_PORT",
    }
    env: dict[str, str] = {}
    for field_name, env_name in mapping.items():
        value = getattr(section, field_name)
        if value is not None:
            env[env_name] = str(value)
    return env


def graspo_to_ms_swift_argv(
    config: Any,
    *,
    stage: Stage,
    dataset_path: str,
    output_dir: str,
    extra_argv: list[str] | None = None,
) -> list[str]:
    """把 graspo 配置翻译成 ms-swift 的参数向量。

    Args:
        config: ``GraspoConfig`` 实例。
        stage: ``"sft"`` 或 ``"rlhf"``（决定 ``max_completion_length`` 等 RL 专有参数）。
        dataset_path: 已转换好的 ms-swift 数据集文件（由设施层准备，本函数只引用）。
        output_dir: 本次运行的输出目录（graspo ``training.output_dir``；已由调用方
            按 ``training.run_name`` 做过隔离，本函数不再二次拼接）。
        extra_argv: 调用方追加的参数（如 ``--max_steps 1`` 冒烟边界）。追加在末尾，
            因此**可以覆盖**前面的同名参数——这是唯一的覆盖入口，且由调用方显式传入。

    Returns:
        ``["--model", "/path", "--dataset", "/path", ...]``。

    Note:
        映射只读 graspod 的 ``model`` / ``data`` / ``lora`` / ``training`` 段 +
        ``msswift`` 段。任何一段里没有出现过的语义，本函数不会凭空发明。
    """
    if stage not in ("sft", "rlhf"):
        raise ValueError(f"stage must be 'sft' or 'rlhf', got {stage!r}")

    model = config.model
    data = config.data
    lora = config.lora
    training = config.training
    section = config.msswift

    argv: list[str] = []
    # ── 模型（graspo model 段 → ms-swift BaseArguments）────────────────────
    _extend(argv, "model", model.model_path)
    _extend(argv, "torch_dtype", model.torch_dtype)
    if model.gradient_checkpointing:
        # 只在开启时透传：ms-swift 侧默认已是梯度检查点，显式关闭会改变显存/速度语义，
        # 而 graspo 的 False 在 ms-swift 路径上没有等价语义 → 不假装支持。
        argv.extend(["--gradient_checkpointing", "true"])

    # ── 数据 ─────────────────────────────────────────────────────────────
    _extend(argv, "dataset", dataset_path)
    if stage == "sft":
        # 与 native SFT 同语义：整条序列（prompt+response）上限 = data.max_prompt_length
        # （`flow/trainer/sft_trainer.py:132` 的 max_seq_len 即此值）。
        _extend(argv, "max_length", data.max_prompt_length)
    else:
        # RL：prompt 上限与 completion 上限是两个正交参数（§7.4 参数正交）。
        _extend(argv, "max_length", data.max_prompt_length)
        _extend(argv, "max_completion_length", training.max_new_tokens)

    # ── 参数化模式（graspo tuner_type → ms-swift `--tuner_type`）──────────
    # 唯一取值来源是 `GraspoConfig.effective_tuner_type`（None ⇒ lora，向后兼容）。
    # ms-swift 4.5.3 的合法取值为 `{lora,full,lora_llm}`（实跑证据：
    # task-megatron-smoke/evidence/88-train-v5-error.txt:14 的 CLI help），
    # graspo 只映射 lora|full。
    tuner_type = config.effective_tuner_type
    argv.extend(["--tuner_type", tuner_type])
    if tuner_type == "lora":
        # LoRA 专属参数只在与 LoRA 同用时才产出：全参下发 lora_rank/alpha/dropout
        # 是"看起来生效、实际被忽略"的假配置（宪法 §1.4 / §7.2）。
        _extend(argv, "lora_rank", lora.r)
        _extend(argv, "lora_alpha", lora.alpha)
        _extend(argv, "lora_dropout", lora.dropout)
        if lora.target_modules is not None:
            _extend(argv, "target_modules", list(lora.target_modules))
        if lora.adapter_path:
            _extend(argv, "adapters", [str(lora.adapter_path)])
    else:
        # 全参：显式解冻 ViT 与 aligner，使 ms-swift 的"全参"与 native 侧一致
        # （都训练全部权重，含视觉塔与 aligner；矩阵 §4 的定义就是"训练全部权重"）。
        # 上游默认 freeze_vit/freeze_aligner=True（多模态 full 微调只训语言主干），
        # 不显式传就会造成**两后端语义分叉**、破坏 54 档台账的同档可比性。
        # 取值来源唯一：`msswift.freeze_vit` / `msswift.freeze_aligner`
        # （None = 用模式默认值 false；显式 true/false 为用户覆盖）。
        _extend(
            argv,
            "freeze_vit",
            False if section.freeze_vit is None else bool(section.freeze_vit),
        )
        _extend(
            argv,
            "freeze_aligner",
            False if section.freeze_aligner is None else bool(section.freeze_aligner),
        )

    # ── 训练超参（graspo training 段）───────────────────────────────────
    _extend(argv, "output_dir", output_dir)
    _extend(argv, "run_name", training.run_name or None)
    _extend(argv, "seed", training.seed)
    _extend(argv, "num_train_epochs", training.max_epochs)
    _extend(argv, "learning_rate", training.learning_rate)
    _extend(argv, "weight_decay", training.weight_decay)
    _extend(argv, "max_grad_norm", training.max_grad_norm)
    _extend(argv, "gradient_accumulation_steps", training.gradient_accumulation_micro_batches)
    if training.save_steps and training.save_steps > 0:
        argv.extend(["--save_strategy", "steps"])
        _extend(argv, "save_steps", training.save_steps)
    elif training.save_checkpoint_every_epoch:
        argv.extend(["--save_strategy", "epoch"])

    if stage == "rlhf":
        # RLHF 参数类的 `rlhf_type` 默认是 'dpo'（DPO 数据集需要 chosen/rejected），
        # 不显式指定就会拿 GRPO 的提示词数据去按 DPO 编码，报
        # `ValueError: inputs.rejected is None`。graspo 的 RL 就是 GRPO，显式写死。
        argv.extend(["--rlhf_type", "grpo"])
        _extend(argv, "num_generations", training.rollout_group_size)
        _extend(argv, "temperature", training.temperature)
        _extend(argv, "top_p", training.top_p)
        argv.extend(["--steps_per_generation", "1"])
        # KL 惩罚系数固定为 0：graspo 的 GRPO 目标是纯 PPO-clip
        # （`ripple/loss.py::GRASPORippleLoss` 没有 KL 项），而 ms-swift 的 GRPO
        # 默认会在 beta≠0 时引入参考模型 KL。固定 0 才能保证两个后端**同语义**；
        # 这是算法层决定（不属后端配置），因此不放进 `msswift` 段。
        argv.extend(["--beta", "0.0"])
        # GRPO 的数据集列（`targets`）要透传给奖励函数，必须保留未使用列。
        argv.extend(["--remove_unused_columns", "false"])

    # ── msswift 段透传（标准路径 + 长文）──────────────────────────────────
    # 注意：Megatron 参数**不在**这里——它们只对 Megatron 启动通道合法，
    # 见 ``megatron_passthrough_argv`` / ``graspo_to_ms_swift_megatron_argv``。
    argv.extend(msswift_passthrough_argv(config, include_megatron=False, stage=stage))

    if extra_argv:
        argv.extend(extra_argv)
    return argv


#: graspo 侧存在、但 msswift 后端不映射的字段（不假装支持；由调用方记录 WARNING）。
_NATIVE_ONLY_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "model.trust_remote_code",
        "ms-swift 4.5.3 has no `trust_remote_code` argument; it decides per model type. "
        "The field still governs the native backend.",
    ),
    (
        "lora.target_preset",
        "graspo's target_preset is a native-side module-name preset (core/lora.py); ms-swift "
        "`target_modules` uses regex / 'all-linear'. Set lora.target_modules explicitly to "
        "control LoRA targets on the msswift backend.",
    ),
)


def native_only_fields(config: Any) -> list[str]:
    """返回"在 graspo 配置里存在、但 msswift 后端不透传"的字段名清单。

    调用方（``sft_trainer`` / ``trainer`` 门面）据此记录 WARNING，让用户知道哪些
    配置在 msswift 后端不生效——**透明退路**（宪法 §3.2），不是静默忽略。
    """
    active: list[str] = ["model.trust_remote_code"]
    if config.lora.target_modules is None:
        active.append("lora.target_preset")
    return active


def native_only_field_notes() -> dict[str, str]:
    """字段名 → 人类可读说明（单一真相源，供日志复用）。"""
    return dict(_NATIVE_ONLY_FIELDS)


# ── 组合约束的前置校验（防御性报错，**不是能力修复**）────────────────────────
#
# E2b 缺陷清单 3/4/5 的三条组合约束都是**上游**（megatron-core / DeepSpeed / ms-swift）
# 的限制，不是 graspo 能修的：本函数只把它们变成**启动前**可读的报错，避免用户在
# 训练跑到几十分钟后才拿到 `ChildFailedError` 这种没有根因的消息。
# 明确声明：**这是防御性报错改进，不是能力修复**（不改变任何一条约束本身）。


def validate_combinations(config: Any) -> None:
    """启动前校验已知的非法组合，失败时抛出可操作的中文/英文混合说明。

    Raises:
        ValueError: 命中一条已知的非法组合（消息里带上游出处与可行替代）。

    Note:
        只校验"能确定性判定的"组合；涉及机器/环境的约束（如
        ``CUDA_DEVICE_MAX_CONNECTIONS`` 同时被 Megatron-FSDP 要求 ``>1``、被
        Megatron-SP 要求 ``=1``）不在本函数里判——它取决于用户怎么起进程，
        只能在报错文档里说明（见 ``megatron_passthrough_argv`` 的 docstring）。
    """
    section = config.msswift
    megatron = section.megatron

    # 缺陷 3：fp8_param_gather 必须与 fp8 模式同开（megatron-core 0.17.1
    # `transformer_config.py:1160`：`ValueError: fp8_param must be used together with fp8 mode.`）
    if megatron.fp8_param_gather:
        raise ValueError(
            "msswift.megatron.fp8_param_gather=true requires FP8 mode to be enabled at the "
            "same time: megatron-core rejects `fp8_param` without `fp8` "
            "(`ValueError: fp8_param must be used together with fp8 mode.`). "
            "Note (measured 2026-09-16): passing --fp8_format/--fp8_recipe is NOT "
            "sufficient on ms-swift 4.5.3 + megatron-core 0.17.1 — the FP8 parameter-gather "
            "path is not usable through this channel yet. Disable fp8_param_gather."
        )

    # 缺陷 4：DeepSpeed AutoTP 只支持全参微调（官方限制），与 LoRA 组合会形状错误
    # `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2762x4 and 8x2048)`
    # 全参入口落地后，AutoTP 的唯一合法前置就是 `tuner_type: full`——因此拒绝条件
    # 从"恒拒绝"收紧为"非全参时拒绝"（宪法 §2.3：非法组合仍然 fail-closed，
    # 只是不再把合法组合一起挡掉）。
    if section.deepspeed_autotp_size and config.effective_tuner_type != "full":
        raise ValueError(
            "msswift.deepspeed_autotp_size (S5 DeepSpeed AutoTP) only supports FULL "
            "fine-tuning upstream; combining it with LoRA produces "
            "`RuntimeError: mat1 and mat2 shapes cannot be multiplied` (measured "
            "2026-09-16, evidence: e2b matrix §6 defect 4). "
            "Set `tuner_type: full` in the config, or drop deepspeed_autotp_size."
        )
