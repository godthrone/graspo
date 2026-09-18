"""配置模型定义：GraspoConfig 及所有子配置段的 pydantic 模型，加载即校验。"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: 训练参数化模式：``lora`` = 只训 LoRA 适配器；``full`` = 全参（全量）微调。
#: 与 ms-swift 的 ``--tuner_type`` **同轴**（上游合法取值为
#: ``lora|full|lora_llm``，graspo 只支持前两个）。native 与 msswift 两个后端
#: 共用这一个开关——同一语义只有一个字段（宪法 §1.4 单一真相源）。
TunerType = Literal["lora", "full"]


def resolve_tuner_type(tuner_type: TunerType | None) -> TunerType:
    """把"未指定"归一为默认值：``None`` ⇒ ``"lora"``。

    v0.24 之前的全部配置都没有这个字段，语义等价于 LoRA，因此 None 必须落到
    ``lora``（向后兼容）。**不使用** ``""`` / ``"none"`` 等哨兵值表达"未指定"
    （宪法 §2.2 None 语义），检查一律用 ``is None``。
    """
    if tuner_type is None:
        return "lora"
    return tuner_type


def validate_tuner_type_combination(
    *,
    tuner_type: TunerType | None,
    backend: str,
    lora_adapter_path: str | None,
    native_tp_size: int,
    native_dp_size: int,
) -> None:
    """全参模式的已知非法组合，**启动前** fail-closed 拒绝（宪法 §2.3 边界校验即防呆）。

    只做纯逻辑判断，不 import torch / 不读文件——因此可在无 torch 的机器上单测
    （由 :meth:`GraspoConfig._validate_tuner_type_combination` 调用）。

    两条规则：

    1. ``full`` 与 ``lora.adapter_path`` 互斥。``adapter_path`` 是"往冻结基座上挂
       已训好的 LoRA"，全参模式的语义里没有可挂载的适配器；两者同时出现必然是配置
       写错，必须当场报错而不是静默忽略其中一个。
    2. ``native`` 后端的 ``full`` 目前**只支持 PP 分片**（``tp_size>1`` 或
       ``dp_size>1`` 一律拒绝）。原因是既有梯度同步实现（``flow/lora/lora_linear.py``
       的 ``_sync_dp_lora_grads`` / ``_sync_nonsharded_lora_grads``）**只覆盖 LoRA
       参数**：全参下 DP 副本的基座梯度不会 all-reduce（静默训错），TP 各 rank 的
       分片基座梯度同步语义也未经适配。让它们在"没同步"的状态下跑起来比拒绝更危险
       ——宁可 fail-closed。全参的 TP 通道走 ms-swift + DeepSpeed AutoTP
       （``msswift.deepspeed_autotp_size``），不经 native 的 ``tp_size``。
    """
    if resolve_tuner_type(tuner_type) != "full":
        return
    if lora_adapter_path is not None:
        raise ValueError(
            "tuner_type=full cannot be combined with lora.adapter_path: "
            "adapter_path loads a pre-trained LoRA adapter onto a frozen base model, "
            "which is the opposite of full-parameter fine-tuning. "
            "Drop lora.adapter_path (or set tuner_type=lora)."
        )
    if backend == "native" and (int(native_tp_size) > 1 or int(native_dp_size) > 1):
        raise ValueError(
            "native backend tuner_type=full currently supports pipeline parallelism only: "
            f"tp_size={int(native_tp_size)}, dp_size={int(native_dp_size)} are not supported "
            "because the native gradient-sync helpers only cover LoRA parameters "
            "(full-parameter DP/TP gradients would be silently unsynchronized). "
            "Use pp_size>1 for native full-parameter training, or use the msswift backend "
            "with msswift.deepspeed_autotp_size (DeepSpeed AutoTP) for tensor parallelism."
        )


class RewardConfig(BaseModel):
    """奖励评分配置，所有字段在加载时校验，拒绝未知字段。"""

    model_config = ConfigDict(extra="forbid")

    kind: str = "graspo"
    check_think: bool = False
    check_json_markdown: bool = True
    check_list_order: bool = False
    marker_reward_weight: float = 10.0
    content_reward_weight: float = 100.0
    anti_useless_str_reward_weight: float = 1.0
    anti_useless_str_half_reward_len: int = 100
    numeric_tolerance: float = 0.2

    @model_validator(mode="after")
    def _validate_reward_kind(self) -> "RewardConfig":
        """延迟导入 REWARD_REGISTRY 以避免循环依赖，校验 kind 是否已注册。"""
        from graspo.ripple.reward.reward import REWARD_REGISTRY  # noqa: PLC0415

        if self.kind not in REWARD_REGISTRY:
            raise ValueError(
                f"Unknown reward kind {self.kind!r}; "
                f"available: {sorted(REWARD_REGISTRY)}"
            )
        return self


class LoRAConfig(BaseModel):
    """LoRA 微调配置。"""

    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    adapter_path: str | None = None
    target_preset: str = "language_safe"
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseModel):
    """模型加载配置。"""

    model_config = ConfigDict(extra="forbid")

    model_path: str = ""
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict[str, Any] = {}


class LRSchedulerConfig(BaseModel):
    """学习率调度器配置。默认 ``type="constant"`` 保持 LR 不变，向后兼容。

    ``decay_steps`` 是纯调度参数：warmup 结束后的衰减跨度（optimizer-step
    粒度），调度在此步数内从 ``learning_rate`` 衰减到
    ``learning_rate × min_lr_ratio``，之后保持最低 LR 不变。训练长度只由
    ``training.max_epochs`` 控制（§7.4 参数单职责——max_steps 已于 v0.23.0
    移除：它曾同时承担提前终止与调度跨度两个角色，单位不一致互相矛盾）。
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "constant"  # "constant" | "cosine" | "linear"
    warmup_steps: int = 0  # 线性 warmup 步数（仅 cosine / linear）
    min_lr_ratio: float = 0.0  # 最终 LR = learning_rate × min_lr_ratio
    decay_steps: int = 0  # warmup 后的衰减跨度（optimizer-step）；type != constant 时必填


class TrainingConfig(BaseModel):
    """训练超参数配置。"""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = ""
    run_name: str = ""
    # 预授权退路（§3.3）：是否允许覆盖已有输出目录。默认 False（安全默认），
    # 用户必须在配置中显式设置为 true 才能覆盖已有训练产出。
    overwrite_output_dir: bool = False
    seed: int = 42
    # 训练长度唯一控制参数（v0.23.0 起：max_steps 已移除，见 LRSchedulerConfig）
    max_epochs: int = 100
    rollout_group_size: int = 8
    # 每次 rollout queue 的 prompt 数（默认 8）。与 micro_batch_size 解耦：
    # 队列决定采样吞吐与 replay 阈值（G × queue），micro_batch_size
    # 决定训练 forward 每批序列数（显存峰值）。OOM 时单独调小后者即可，
    # 不需要牺牲吞吐。
    rollout_queue_batch_size: int = 8
    # 梯度累积：攒几个 micro_batch 后做一次 optimizer step。
    # 有效 batch/GPU = micro_batch_size × gradient_accumulation_micro_batches
    # 全局有效 batch = 有效 batch/GPU × dp_size
    gradient_accumulation_micro_batches: int = 4
    # optimize_iterations_per_step removed — always 1 iteration per step.
    # GRPO with repeated iterations on stale old_log_probs causes catastrophic
    # forgetting of shared tokens (e.g. tool-call formatting).
    rollout_max_retries: int = 5
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    policy_ratio_clip_eps: float = 0.2
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    save_steps: int = -1
    save_checkpoint_every_epoch: bool = True
    save_checkpoint_time_period_minutes: int = 0
    perfect_skip_reward_threshold: float = 1.0
    reject_unparseable_groups: bool = True
    resume_from_checkpoint: str | None = None
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    # DP 学习率缩放策略：dp_size > 1 时按 linear scaling rule 调整 lr
    # "linear" = lr × dp_size；"none" = 不缩放
    lr_scaling: Literal["linear", "none"] = "linear"

    def effective_learning_rate(self, dp_size: int = 1) -> float:
        """DP 缩放后的有效学习率。

        linear scaling rule（§1.4 单一真相源）：有效 lr 由 learning_rate、
        lr_scaling、dp_size 三者唯一确定，不分散到多个调用点各自计算。
        """
        if self.lr_scaling == "linear" and dp_size > 1:
            return self.learning_rate * float(dp_size)
        return self.learning_rate

    @model_validator(mode="after")
    def _validate_output_dir(self) -> TrainingConfig:
        """默认输出目录推导：outputs/<run_name>，run_name 自动生成。"""
        output_dir = str(self.output_dir or "").strip()
        run_name = str(self.run_name or "").strip()
        if not output_dir:
            if not run_name:
                run_name = _generate_run_name()
            self.output_dir = str(Path("outputs") / run_name)
            self.run_name = run_name
        elif not run_name:
            self.run_name = str(Path(output_dir).name)
        return self

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_max_steps(cls, data: Any) -> Any:
        """防呆：max_steps 已于 v0.23.0 移除，给出明确迁移提示而非模糊报错。

        训练长度只由 max_epochs 控制；学习率衰减跨度用 lr_scheduler.decay_steps。
        """
        if isinstance(data, dict) and "max_steps" in data:
            raise ValueError(
                "training.max_steps has been removed (v0.23.0): training length is "
                "controlled by training.max_epochs only; configure the LR decay span "
                "via lr_scheduler.decay_steps (in optimizer steps, warmup-exclusive)"
            )
        return data

    @model_validator(mode="after")
    def _validate_save_checkpoint_time_period(self) -> TrainingConfig:
        if self.save_checkpoint_time_period_minutes < 0:
            raise ValueError(
                f"training.save_checkpoint_time_period_minutes must be >= 0, "
                f"got {self.save_checkpoint_time_period_minutes}"
            )
        return self

    @model_validator(mode="after")
    def _validate_scheduler_decay_steps(self) -> TrainingConfig:
        if self.lr_scheduler.type != "constant" and self.lr_scheduler.decay_steps <= 0:
            raise ValueError(
                f"lr_scheduler.decay_steps must be > 0 when "
                f"lr_scheduler.type={self.lr_scheduler.type!r}, "
                f"got {self.lr_scheduler.decay_steps}"
            )
        return self

    @property
    def replay_buffer_optimize_threshold(self) -> int:
        return int(self.rollout_queue_batch_size) * int(self.rollout_group_size)


class DataConfig(BaseModel):
    """训练数据配置。"""

    model_config = ConfigDict(extra="forbid")

    train_path: str = ""
    max_prompt_length: int = 2048


class GraspoFlowConfig(BaseModel):
    """GraspoFlow 分布式训练配置 — TP+DP+PP+SP+Checkpoint 五位一体。

    world_size = dp_size × tp_size × pp_size，由框架自动校验。
    各维度正交：调整任何一个不影响其他维度的语义。
    """

    model_config = ConfigDict(extra="forbid")

    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    # 是否在 DP rank 间复制 LoRA 权重（默认 true）。true=各 rank 独立副本，
    # 梯度 AVG all-reduce；false=权重分片，需额外通信。LoRA 参数量小，推荐 true。
    dp_replicate_lora: bool = True
    # 模型适配器路径，默认使用 qwen35_36（兼容 Qwen3.5/3.6 系列）
    adapter: str = "graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter"
    placement_strategy: str = "auto"
    # 手动指定每层的 stage 分布 [start, end) 区间，设置后覆盖 placement_strategy
    layer_ranges: list[list[int]] | None = None
    sequence_parallel: bool = False
    pp_micro_batch_size: int = 1
    # 每条数据管线一次 forward 的样本数。TP/PP 不改变数据分布，
    # 一条管线跨越 tp_size × pp_size 个 GPU，只算 1 个 micro batch。
    # DP 下每条管线独立处理自己的数据分片。
    micro_batch_size: int = 1
    use_kv_cache_for_rollout: bool = True
    empty_cache_after_rollout_split: bool = False
    empty_cache_before_train: bool = False
    raw_log_enabled: bool = True
    readable_log_enabled: bool = True
    synchronize_cuda_timing: bool = False
    pp_max_inflight_microbatches: int = 0
    # PP 调度策略（默认 1F1B）。旧 GPipe（全 forward 后全 backward，bubble 最高、
    # 不重叠 forward/backward）已按宪法 §18.1 删除。1F1B 依赖双向进程组
    # （pp_group_fwd/pp_group_bwd）+ 显式 tag + 背压，早期单 peer-pair FIFO
    # 的双向消息错配死锁已用此机制消除。
    pp_scheduler: Literal["one_f_one_b", "1f1b"] = "one_f_one_b"

    @model_validator(mode="after")
    def _validate_dp(self) -> GraspoFlowConfig:
        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        return self


class MsSwiftMegatronConfig(BaseModel):
    """Megatron-SWIFT 路径的分片参数（`ms-swift-implementation-plan.md` §8.2 MG1-MG11）。

    字段名与 ms-swift ``MegatronArguments`` 的参数名**逐字一致**（去掉前导 ``--``），
    因此"配置字段 → ms-swift 参数"的映射不需要猜测（宪法 §2.2 显式即防呆）。

    验收深度（D8 A 档）：Megatron 路径**只做配置透传 + 启动冒烟**，不做长训验证。
    本段只在用户显式提供时透传；未提供（``None``/缺省）的参数不进 ms-swift argv，
    由 ms-swift 自己的默认值决定——不透传不代表"关闭"，语义上不等价于 0/False。
    """

    model_config = ConfigDict(extra="forbid")

    # MG1 数据并行 DP（DP 度由 总 GPU /（TP×PP×CP）自动推导）
    global_batch_size: int | None = None
    data_sharding: bool | None = None
    data_parallel_random_init: bool | None = None
    overlap_grad_reduce: bool | None = None
    # MG2 分布式优化器（Megatron 默认即 ZeRO-1）
    use_distributed_optimizer: bool | None = None
    # MG3 Megatron-FSDP
    use_megatron_fsdp: bool | None = None
    data_parallel_sharding_strategy: (
        Literal["no_shard", "optim", "optim_grads", "optim_grads_params"] | None
    ) = None
    strict_fsdp_dtensor_load: bool | None = None
    # MG4 张量并行 TP
    tensor_model_parallel_size: int | None = None
    tp_comm_overlap: bool | None = None
    # MG5 流水线并行 PP
    pipeline_model_parallel_size: int | None = None
    overlap_p2p_comm: bool | None = None
    align_param_gather: bool | None = None
    pipeline_model_parallel_layout: str | None = None
    decoder_first_pipeline_num_layers: int | None = None
    decoder_last_pipeline_num_layers: int | None = None
    # MG6 序列并行 SP（**仅当 TP > 1 时生效**；布尔开关，与标准路径的
    # `msswift.sequence_parallel_size` 是两套机制，勿混用）
    sequence_parallel: bool | None = None
    # MG7 上下文并行 CP（长上下文）
    context_parallel_size: int | None = None
    cp_comm_type: Literal["p2p", "all_gather", "a2a", "a2a+p2p"] | None = None
    cp_partition_mode: str | None = None
    sequence_packing_scheduler: Literal["dp_balanced", "default_dynamic_cp"] | None = None
    # MG8 / MG9 专家并行 EP / 专家张量并行 ETP（MoE 模型）
    expert_model_parallel_size: int | None = None
    expert_tensor_parallel_size: int | None = None
    # MG10 虚拟流水线 VPP
    virtual_pipeline_model_parallel_size: int | None = None
    microbatch_group_size_per_vp_stage: int | None = None
    # MG11 低精度参数分片（FP8/FP4）
    fp8_param_gather: bool | None = None
    fp4_param_gather: bool | None = None


class MsSwiftConfig(BaseModel):
    """ms-swift 后端配置段（``backend: msswift`` 时生效）。

    **段名与 ``backend`` 取值一致（``msswift``）**——同一后端只有一个名字
    （一名一物，宪法 §2.2），避免 ``ms_swift`` / ``msswift`` 两个别名。

    **单一真相源（§1.4）**：ms-swift 参数只在这一段配置，映射层
    （``flow/msswift/_config_mapping.py``）只从这一段取值；graspo 既有的
    ``model`` / ``data`` / ``lora`` / ``training`` 段继续是训练语义的唯一来源，
    两者不重叠、不互相覆盖。

    字段名与 ms-swift 官方参数名逐字对应（去前导 ``--``），覆盖
    `ms-swift-implementation-plan.md` §8 矩阵的标准路径 7 项（S1-S7）与长文附项；
    Megatron 路径 11 项（MG1-MG11）在同名嵌套段 ``megatron`` 下。

    ``None`` = 未提供（不透传，交 ms-swift 默认值），语义与"显式 0/False"不同
    （宪法 §2.2 None 语义）。
    """

    model_config = ConfigDict(extra="forbid")

    # ── S1 数据并行 DDP（launcher 层；ms-swift 侧是环境变量，非 CLI 参数）────
    nproc_per_node: int | None = None
    nnodes: int | None = None
    node_rank: int | None = None
    master_addr: str | None = None
    master_port: int | None = None
    # ── S2 device_map 模型并行（层切分）──────────────────────────────────
    device_map: str | None = None
    # ── S3 DeepSpeed ZeRO-0/1/2/3(+offload) ─────────────────────────────
    # 取值 zero0|zero1|zero2|zero3|zero2_offload|zero3_offload，或自定义 ds 配置路径
    deepspeed: str | None = None
    # ── S4 ZeRO++（节点内权重分片 + 跨节点数据分片）──────────────────────
    zero_hpz_partition_size: int | None = None
    # ── S5 DeepSpeed AutoTP（张量并行；仅 zero0/1/2 + 仅全参）────────────
    deepspeed_autotp_size: int | None = None
    # ── S6 FSDP2（与 DeepSpeed 互斥）─────────────────────────────────────
    fsdp: str | None = None
    # ── S7 序列并行 SP（Ulysses + Ring-Attention 共用此参数；默认 1=关闭）──
    sequence_parallel_size: int = 1
    # ── 全参（`tuner_type: full`）的冻结开关（ms-swift `TunerArguments`）────
    # ms-swift 对多模态模型的 full 微调**默认** `freeze_vit=True`、`freeze_aligner=True`
    # （上游 4.5.3：`arguments/tuner_args.py:122-124` + `_init_multimodal_full`；
    # Megatron 通道 `megatron/arguments/megatron_args.py:451` 同为 True），即
    # "只训语言主干"。而 graspo 的「全量（全参）」承诺"训练全部权重"
    # （`docs/capability-matrix.md` §4），native 侧也确实放开了全部参数。
    # ⇒ 为消除**两后端语义分叉**，full 模式下这两个开关默认解析为 `false` 并显式
    # 透传（LoRA 模式完全不透传——逐字保持原行为）。
    # None = 用上述模式默认值；显式 true/false = 用户覆盖（唯一真相源，只有这一处开关）。
    freeze_vit: bool | None = None
    freeze_aligner: bool | None = None
    # ── 长文附项（与 SP/ZeRO3/FSDP2 组合使用）───────────────────────────
    rope_scaling: str | None = None  # yarn | dynamic
    max_model_len: int | None = None
    packing: bool = False
    padding_free: bool = False
    attn_impl: str | None = None  # flash_attn | flash_attention_2 | ...
    use_liger_kernel: bool = False
    # ── 长文显存的关键开关（E2b §4.2 定位的瓶颈对策）────────────────────
    # ms-swift 在 packing 下对**整条 packed 序列**算 logits；64K × 151k 词表 × bf16
    # ≈ 19.8 GiB 单份，是 128K 在 8×80GB 上 OOM 的主因。打开它只对需要的位置算 logits。
    # 与 `penalty_ignore_labels` / completion-only 训练语义相关，默认 False（不改变默认行为）。
    use_logits_to_keep: bool = False
    # ── Megatron 路径（仅透传 + 启动冒烟，D8 A 档）────────────────────────
    megatron: MsSwiftMegatronConfig = MsSwiftMegatronConfig()
    # ── 每个 rollout batch 上的优化轮次（ms-swift `--num_iterations`）──────
    # graspo 的默认是 **1**（native 侧已删除 optimize_iterations_per_step：
    # 在陈旧的 old_log_probs 上重复迭代会导致共享 token 的灾难性遗忘）。
    # 这里保留可配是为了**验证 PPO ratio 修复**——只有 num_iterations > 1 时，
    # 第二步的策略前向才会相对第一步的基线发生移动，ratio 才可能 ≠ 1。
    num_iterations: int | None = None
    # ── 每设备训练微批大小（ms-swift 侧 `--per_device_train_batch_size`）──
    # 与 native 的 `native.micro_batch_size` 语义相同但**不能共用**：native 的网格由
    # tp/dp/pp 决定，ms-swift 的网格由 DeepSpeed/FSDP/Megatron 决定，两者不是同一个量。
    # None = 不透传（交 ms-swift/HF 默认值）。
    per_device_train_batch_size: int | None = None
    # ── RL 采样通道（G3：vLLM rollout 与 HF forward 的 logprob 差异是
    #    PPO ratio ≠ 1 的真实来源，故 rollout 引擎必须可配、不可隐式）──────
    use_vllm: bool | None = None
    vllm_mode: Literal["colocate", "server"] | None = None


class ExportConfig(BaseModel):
    """模型导出配置。通过 ``graspo export --config <yaml>`` 驱动。"""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = ""
    export_format: str = "peft-adapter"
    export_output: str = ""
    final_formats: list[str] = []


class LaunchConfig(BaseModel):
    """分布式启动配置。

    GPU 选择由用户通过 run.sh ``--gpus`` 控制（绝不使用 CUDA_VISIBLE_DEVICES，
    与 --gpus 混用会导致 NCCL 死锁），不在配置中指定。
    ``nproc_per_node`` 默认从 ``dp_size × tp_size × pp_size`` 自动推导。
    """

    model_config = ConfigDict(extra="forbid")

    nproc_per_node: int | None = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    python: str | None = None


class GraspoConfig(BaseModel):
    """GRASPO 训练主配置，单一 YAML 入口，加载即校验。"""

    model_config = ConfigDict(extra="forbid")

    train_method: Literal["graspo", "sft"] = "graspo"
    backend: str = "native"
    native: GraspoFlowConfig = GraspoFlowConfig()
    msswift: MsSwiftConfig = MsSwiftConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    lora: LoRAConfig = LoRAConfig()
    export: ExportConfig = ExportConfig()
    launch: LaunchConfig = LaunchConfig()
    reward: RewardConfig = RewardConfig()
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # ── 训练参数化模式（native / msswift 两个后端的唯一开关）────────────────
    # None = 未指定 ⇒ lora（向后兼容，v0.24 之前的所有配置都没有这个字段）。
    # 取 full = 全参（全量）微调：native 侧放开全部基座参数的 requires_grad，
    # msswift 侧映射为 `--tuner_type full`。
    # 消费点只读 ``effective_tuner_type``（归一后再用），不直接读原始字段，
    # 避免每个调用点各自判 None（宪法 §1.4 单一真相源）。
    tuner_type: TunerType | None = None

    @property
    def effective_tuner_type(self) -> TunerType:
        """归一后的训练模式：``None`` ⇒ ``"lora"``（见 :func:`resolve_tuner_type`）。"""
        return resolve_tuner_type(self.tuner_type)

    @model_validator(mode="after")
    def _validate_tuner_type_combination(self) -> GraspoConfig:
        """全参模式的已知非法组合在配置加载时就被拒绝（宪法 §2.3）。

        纯逻辑委托给 :func:`validate_tuner_type_combination`，使该校验可在无 torch
        的机器上单测。
        """
        validate_tuner_type_combination(
            tuner_type=self.tuner_type,
            backend=self.backend,
            lora_adapter_path=self.lora.adapter_path,
            native_tp_size=self.native.tp_size,
            native_dp_size=self.native.dp_size,
        )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraspoConfig:
        """从 YAML 文件加载配置，加载时完成全部校验。

        校验失败时输出人类可读的错误信息（哪个字段、期望什么、可用字段有哪些），
        而不是原始 pydantic traceback。
        """
        import yaml
        from pydantic import ValidationError

        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            return cls.from_dict(yaml.safe_load(text))
        except ValidationError as exc:
            _report_config_errors(exc, path, cls)
            raise SystemExit(1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraspoConfig:
        """从字典构建配置，pydantic ``model_validate`` 一次性校验所有字段。

        **防呆：** 不手动挑键——顶层任何未知键（拼写错误）
        由 ``extra="forbid"`` 直接拒绝，而不是静默忽略后用默认值训练。
        仅做一件显式处理：None 段防御——显式 ``section: null`` 等价于缺省
        （合法），未知键仍被拒绝。
        """
        data = dict(data or {})
        data["native"] = dict(data.get("native") or {})
        for section in (
            "msswift",
            "model",
            "data",
            "lora",
            "export",
            "launch",
            "reward",
            "training",
        ):
            if data.get(section) is None:
                data[section] = {}
        return cls.model_validate(data)


class Sample(BaseModel):
    """单条训练样本，包含 messages、targets 和可选的 tools（不可变）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = {}
    media: list[dict[str, Any]] = []

    @property
    def expects_tool_calls(self) -> bool:
        for target in self.targets:
            output = target.get("output") if isinstance(target, dict) else None
            if isinstance(output, dict) and output.get("tool_calls") is not None:
                return True
        return False

    @property
    def prompt_preview(self) -> str:
        parts: list[str] = []
        for message in self.messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            parts.append(f"{role}: {_content_preview(content)}")
        return "\n\n".join(part for part in parts if part)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


# ── 辅助函数 ────────────────────────────────────────────────────────────────


def _report_config_errors(
    exc: Any, config_path: Path, config_cls: Any
) -> None:
    """将 pydantic ValidationError 转成人类可读的错误信息。

    不输出原始 traceback，而是按字段分组，列出每个问题字段、
    期望的值、可用字段列表。
    """
    import sys

    errors = exc.errors()
    # 构建各段可用字段的映射：顶层 + 每个子段
    config_section_fields: dict[str, list[str]] = {
        section: sorted(
            [
                k
                for k in config_cls.model_fields[section].annotation.model_fields  # type: ignore[union-attr]
            ]
            if section in config_cls.model_fields
            else []
        )
        for section in (
            "native",
            "msswift",
            "model",
            "data",
            "lora",
            "export",
            "launch",
            "reward",
            "training",
        )
    }
    config_section_fields[""] = sorted(
        k for k in config_cls.model_fields if k not in config_section_fields
    )

    lines: list[str] = []
    lines.append(f"\n配置校验失败 ({config_path}):\n")
    for error in errors:
        loc = ".".join(str(p) for p in error["loc"])
        msg = error["msg"]
        err_type = error["type"]

        if err_type == "extra_forbidden":
            # 拼写错误或已删除的字段
            lines.append(f"  {loc}: 未知字段，当前版本不支持此配置项")
            # 尝试找到所属段，列出可用字段
            section = str(error["loc"][0]) if error["loc"] else ""
            if section in config_section_fields and config_section_fields[section]:
                fields = ", ".join(config_section_fields[section])
                lines.append(f"    {section} 段可用字段: {fields}")
            elif "" in config_section_fields:
                fields = ", ".join(config_section_fields[""])
                lines.append(f"    顶层可用字段: {fields}")
        elif err_type == "missing":
            lines.append(f"  {loc}: 缺少必填字段")
        elif err_type == "string_type":
            lines.append(f"  {loc}: 期望字符串，实际收到 {error.get('input')}")
        elif err_type == "int_type":
            lines.append(f"  {loc}: 期望整数，实际收到 {error.get('input')}")
        elif err_type == "bool_type":
            lines.append(f"  {loc}: 期望布尔值，实际收到 {error.get('input')}")
        elif err_type == "literal_error":
            lines.append(f"  {loc}: 期望 {error.get('expected')}，实际收到 {error.get('input')}")
        else:
            lines.append(f"  {loc}: {msg}")

    print("\n".join(lines), file=sys.stderr)


_RUN_NAME_CACHE: dict[str, str] = {}
"""模块级缓存，确保同一秒内多次调用（如 torchrun 多 worker）得到相同的 run_name。"""


def _generate_run_name() -> str:
    """生成基于时间戳的唯一运行标识。"""
    if "current" not in _RUN_NAME_CACHE:
        _RUN_NAME_CACHE["current"] = f"graspo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return _RUN_NAME_CACHE["current"]


def _content_preview(content: Any) -> str:
    """生成消息内容的可读预览。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            item_type = str(item.get("type") or "").lower()
            if item_type == "text":
                parts.append(str(item.get("text") or ""))
            elif item_type in {"image", "image_url"}:
                parts.append("<image>")
            elif item_type in {"video", "video_url"}:
                parts.append("<video>")
            else:
                parts.append(f"<{item_type or 'content'}>")
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)
