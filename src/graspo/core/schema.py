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

#: 训练方法（算法）枚举 —— **全项目唯一真相源**（宪法 §1.4）。
#: 与能力矩阵 `docs/capability-matrix.md` §4「算法」行一一对应：
#: ``graspo`` = GRASPO(RL)、``sft`` = 监督微调、``cpt`` = 继续预训练、
#: ``opd`` = on-policy 蒸馏。新增算法 = 此字面量加一个取值（§1.2 对扩展开放），
#: 具体实现由 ``core/discovery.py`` 的注册表路由解析。
TrainMethod = Literal["graspo", "sft", "cpt", "opd"]

#: ``train_method`` → **支持它的后端**（单一真相源，宪法 §1.4）。
#: 依据 `docs/capability-matrix.md` §4：CPT / OPD 在 **native 侧就是 `⛔ 不支持`**
#: ⇒ 这里 fail-closed，而不是把它们静默路由到 SFT/RL 训练器（那等于拿着
#: 另一种算法去训练，属宪法 §3.4 的"坏退路"）。
TRAIN_METHOD_BACKENDS: dict[str, tuple[str, ...]] = {
    "graspo": ("native", "msswift"),
    "sft": ("native", "msswift"),
    "cpt": ("msswift",),
    "opd": ("msswift",),
}

#: 上表的**值**并集 = 本项目已知的后端名（用于区分"已知后端但该方法不支持"与
#: "后端名根本不存在"——后者只由 `flow/backend_selection.select_backend` 判，§1.4）。
KNOWN_BACKENDS: frozenset[str] = frozenset(
    name for backends in TRAIN_METHOD_BACKENDS.values() for name in backends
)


def validate_train_method_combination(
    *,
    train_method: str,
    backend: str,
    distill_teacher_model_path: str | None,
) -> None:
    """``train_method`` × ``backend`` × 教师配置的**非法组合**在配置加载时即拒绝。

    纯逻辑函数（零 IO、零重依赖），因此可在无 torch 的机器上独立单测
    （宪法 §1.3 层次边界）——与 :func:`validate_tuner_type_combination` 同一手法。

    **只拒绝"确定非法"的组合**，不为好看的完整性发明约束（§6.1 简单优先）：

    - ``train_method`` 不在 :data:`TRAIN_METHOD_BACKENDS` 里 ⇒ 拒绝（防呆：拼错的算法名
      不得静默退化成默认算法）。
    - ``backend`` 是**本项目已知的后端之一**、但不在该方法的后端集合里 ⇒ 拒绝。
      非已知后端名（如 ``"thirdparty"``）**不在这里判**——那是
      ``flow/backend_selection.select_backend`` 的职责，避免两处各判一次（§1.4）。
    - ``train_method == "opd"`` 必须有**具体**的教师模型路径：用户 2026-09-18 拍板
      "教师 = ``Qwen3.8-27B``，学生 = ``Qwen3.5-9B``"，不再留"教师待定"。

    Args:
        train_method: ``GraspoConfig.train_method`` 的原始值。
        backend: ``GraspoConfig.backend`` 的原始值。
        distill_teacher_model_path: ``distill.teacher_model_path``（``None`` = 未提供）。

    Raises:
        ValueError: 命中上述任一非法组合。
    """
    supported = TRAIN_METHOD_BACKENDS.get(train_method)
    if supported is None:
        raise ValueError(
            f"train_method must be one of {sorted(TRAIN_METHOD_BACKENDS)}, got {train_method!r}"
        )
    if backend in KNOWN_BACKENDS and backend not in supported:
        # `KNOWN_BACKENDS` = `TRAIN_METHOD_BACKENDS` 的**值**并集（见上面的定义）。
        raise ValueError(
            f"train_method={train_method!r} is not supported on backend={backend!r}; "
            f"supported backends for it: {', '.join(supported)}. "
            "CPT and OPD are ms-swift-only capabilities "
            "(docs/capability-matrix.md §4 lists them as unsupported on native)."
        )
    if train_method == "opd" and not (distill_teacher_model_path or "").strip():
        raise ValueError(
            "train_method='opd' requires a concrete teacher model: set "
            "`distill.teacher_model_path` (the user-pinned teacher is Qwen3.8-27B, "
            "student is Qwen3.5-9B). An empty value is rejected rather than treated as "
            "'teacher to be decided'."
        )


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


#: native 侧 offload 的接线只落在 qwen35_36 适配器的 mixin 上
#: （``_Qwen35SFTTrainingMethods._build_optimizer``，WP-X2）。这是**接线范围
#: 的事实**，不是设计取舍：别的 model family 要开启，得先有同样的接线。
#: 用模块前缀而不是"逐字等于默认值"，是为了让同模块内的子类/别名路径也能通过。
_NATIVE_OFFLOAD_ADAPTER_MODULE = "graspo.flow.adapters.models.qwen35_36."


def validate_native_offload_combination(
    *,
    offload_optimizer_state: bool,
    backend: str,
    native_adapter: str,
) -> None:
    """native 优化器态 offload 的已知非法组合，**启动前** fail-closed 拒绝（§2.3）。

    只做纯逻辑判断，不 import torch / 不读文件——因此可在无 torch 的机器上单测
    （由 :meth:`GraspoConfig._validate_native_offload_combination` 调用）。

    **只看显式开启**：``offload_optimizer_state`` 为假时直接返回 ⇒ 既有配置
    （全部不含该键）行为逐位不变。

    两条规则，都是"假配置"而不是"难用"：

    1. 与 ``backend != "native"`` 互斥。开关在 ``native.*`` 命名空间下，
       只有 native 后端消费它；跑 msswift 时打开它不会有任何作用。按 §7.2
       "声明了却无人消费的字段是假配置"，宁可当场报错，也不要让用户以为
       省了显存、实际跑到一半才 OOM。
    2. 与 ``native.adapter`` 指向非 qwen35_36 适配器互斥。offload 的接线落在
       qwen35_36 的 mixin 上（见 ``_NATIVE_OFFLOAD_ADAPTER_MODULE``）；换别的
       model family 时开启它同样是假配置，必须报错而不是静默不生效。
    """
    if not offload_optimizer_state:
        return
    if backend != "native":
        raise ValueError(
            "native.offload_optimizer_state=true cannot be combined with "
            f"backend={backend!r}: the native optimizer-state offload is only wired "
            "into the native backend, so enabling it while running msswift would "
            "silently do nothing (a declared-but-unconsumed field is a fake config, "
            "§7.2). Set native.offload_optimizer_state=false, or use "
            "msswift.deepspeed=zero2_offload (the msswift-side equivalent)."
        )
    if not str(native_adapter).startswith(_NATIVE_OFFLOAD_ADAPTER_MODULE):
        raise ValueError(
            "native.offload_optimizer_state=true cannot be combined with "
            f"native.adapter={native_adapter!r}: the offload is only wired into the "
            f"{_NATIVE_OFFLOAD_ADAPTER_MODULE} adapters "
            "(Qwen3.5/3.6 mixin _build_optimizer). Enabling it for another model "
            "family would silently keep the optimizer state on the GPU."
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


class PretrainConfig(BaseModel):
    """CPT（继续预训练）的**算法级**配置 —— **后端中立**（宪法 §1.4 单一真相源）。

    **为什么独立成段而不是塞进 ``msswift`` 段**：CPT 的语义（"在纯文本上做
    next-token 续训、每个 token 都计损、不套对话模板"）是**算法层的**，不是某个
    后端的私有参数。把它放在中立段，后端映射层只从这一处取值——两个后端不会
    各写一份配置（哪怕本期只有 ms-swift 实现了 CPT，见
    :data:`TRAIN_METHOD_BACKENDS`）。

    字段一律 ``X | None = None``（宪法 §2.2）：``None`` = 未提供 ⇒ 用 **CPT 语义
    默认值**（下面逐字段写明），显式值与 ``None`` 语义不同，检查一律用 ``is None``。
    """

    model_config = ConfigDict(extra="forbid")

    #: 损失覆盖范围。CPT 语义 = "每个 token 都算损失" ⇒ ``None`` ⇒ ``"all"``。
    #: （ms-swift 侧等价于 `swift pt` 的 ``--loss_scale all``。）
    loss_scale: str | None = None
    #: 是否套用对话模板。CPT 语义 = "在纯文本上续训" ⇒ ``None`` ⇒ ``False``。
    #: （ms-swift 侧等价于 `swift pt` 的 ``--use_chat_template false``。）
    use_chat_template: bool | None = None


class DistillConfig(BaseModel):
    """OPD（on-policy 蒸馏）的**教师 / 学生**配置 —— **后端中立**（宪法 §1.4）。

    教师是**具体、可配置的模型路径**，不是"待定"：用户 2026-09-18 拍板
    教师 = ``Qwen3.8-27B``、学生 = ``Qwen3.5-9B``；``train_method: opd`` 时
    :func:`validate_train_method_combination` 强制 ``teacher_model_path`` 非空。

    **学生**就是既有的 ``model.model_path``（GraspoConfig 顶层已有字段）——
    不为它再发明第二个字段（同一语义只有一个字段，§1.4）。
    教师 logprob 由教师模型**现场前向**算出（教师是同一训练进程里的独立冻结模型），
    因此**没有**"教师 logprob 数据列"这种契约，见本包 report §④ 的缺口登记。

    字段名与 ms-swift ``TeacherModelArguments`` / GKD 超参名**逐字对应**：这样
    "配置字段 → 后端参数"的映射不需要猜名字（宪法 §2.2 显式即防呆）。
    """

    model_config = ConfigDict(extra="forbid")

    #: 教师模型路径（`--teacher_model`）。``None`` = 未提供 ⇒ OPD 配置非法（见上）。
    teacher_model_path: str | None = None
    #: 教师 LoRA 适配器路径列表（`--teacher_adapters`）。``None`` = 未提供（不透传）。
    teacher_adapters: list[str] | None = None
    #: 教师模型的 DeepSpeed 配置（`--teacher_deepspeed`）。``None`` = 继承学生侧。
    teacher_deepspeed: str | None = None
    #: 教师前向之外把教师权重换出到 CPU（`--offload_teacher_model`）。
    #: ``None`` = 未提供（交后端默认值 = 不换出），与显式 ``false`` 语义不同。
    offload_teacher_model: bool | None = None
    #: GKD 的 top-k logits 粒度（`--gkd_logits_topk`）。``None`` = 全词表。
    gkd_logits_topk: int | None = None
    #: GKD 的 lambda（`--lmbda`）。``None`` = 交后端默认值。
    lmbda: float | None = None
    #: GKD 与 SFT loss 的混合权重（`--sft_alpha`）。``None`` = 交后端默认值。
    sft_alpha: float | None = None
    #: 教师-学生散度插值系数（`--beta`：0=前向 KL，1=反向 KL，0.5=JSD）。
    #: ``None`` = 交后端默认值（ms-swift GKD 默认 0.5），不在这里发明取值。
    beta: float | None = None


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
    # ── 优化器态 CPU offload（WP-X2；**默认关闭**，开启前需显式预授权）────────
    # 背景（实测，不是算式）：native 全参 1 卡档 `T016`（9B·SFT·全参）在
    # `optimizer.step()` 处 ①真 OOM —— PyTorch 已分配 78.23 GiB / 卡容量
    # 79.25 GiB。显存墙在**优化器态**：参数(bf16) 17.6 + 梯度(bf16) 17.6 =
    # 35.1 GiB（正是实测的 `after_backward=35.14GB`），`step()` 再建两份
    # AdamW 一阶/二阶矩（`zeros_like(param)` ⇒ bf16，4 B/参数 ≈35.1 GiB）。
    # ⇒ 省显存只能在优化器态上做，不是在激活上做。
    #
    # 语义：取 true 时，native 侧把 AdamW 的一阶/二阶矩**常驻 CPU**，每步把
    # 梯度 D2H、更新完的参数 H2D（与 ms-swift 侧 `zero2_offload` 同口径）。
    # 这是宪法 §3.3 的**预授权退路**：数值轨迹可能因 CPU 侧运算的舍入差异而
    # 轻微变化，且**必然变慢**——"变慢"不构成失败，但必须由用户在部署前显式
    # 声明接受，故默认关闭。默认值下行为与基线逐位一致。
    #
    # 消费点只有一处：`flow/adapters/models/qwen35_36/training_sft.py` 的
    # `_Qwen35SFTTrainingMethods._build_optimizer`（单一真相源，§1.4）。
    # 非法组合由 `validate_native_offload_combination` 在加载时 fail-closed 拒绝。
    offload_optimizer_state: bool = False
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
    # ── 多模态图像像素预算（ms-swift `--max_pixels`，单位：像素总数）──────
    # **F-3 实测新增（2026-09-18）**：不设上限时 ms-swift 对每张图按模型默认的
    # `image_max_token_num`（Qwen3.5 为 16384）预算编码——ELAM V5 的 1280×720 双目图
    # 两张即可顶到上限，单条样本编码长度直接超过 `max_length`，ms-swift 抛
    # `MaxLengthError` 后重试耗尽，报出与真因无关的
    # `ValueError: Failed to retrieve the dataset`（`dataset/utils.py:108`）。
    # ⇒ 必须把它做成**配置项**（§1.4 单一真相源），不能靠环境变量 `MAX_PIXELS`。
    # None = 不透传（交 ms-swift 默认值 = 不限制）。
    max_pixels: int | None = None
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


class EvalConfig(BaseModel):
    """效果评测链路配置（``graspo eval``）。ELAM V5 + vLLM，温度锁 0。

    **卡计划只能来自这里**：``gpus`` 必须显式给出（如 ``"0,1"``），取值限定在
    ``{0,1,2,3,4,5}``、卡数 ≤ 4；未显式给出时评测链路**拒绝启动**
    （fail-closed，见 ``eval.guard``）。不给默认值是有意的——目标 GPU 服务器上 GPU 6/7
    被生产 vLLM 占死，任何"顺手用个默认卡"的设计都会炸生产。

    ``temperature`` 字段**不存在**：温度在 ``eval.vllm_client`` 里锁成常量 0，
    连配置都不提供覆盖入口（宪法 §2.2 显式即防呆——不留隐式通道）。

    ``output_dir`` 必须位于 ``.local/`` 下（宪法 §16）：评测产物含宿主路径、
    GPU 编号等环境信息，不得进入已跟踪文件。
    """

    model_config = ConfigDict(extra="forbid")

    dataset_path: str = ""
    train_dataset_path: str | None = None
    #: 重叠分析只取训练集前 N 条（台账里的"train 前 100/20 条"）。
    #: ``None`` = 全量。**切片长度会被记进产物**（``EvalTrainSubset``）——不同切片
    #: 会给出不同的重叠集，不记下来事后无法解释两次评测的口径差异。
    train_limit: int | None = None
    dataset_split: str = "test"
    base_model_path: str = ""
    checkpoint_path: str | None = None
    role: Literal["base", "after", "sft", "other"] = "after"
    export_format: str | None = None
    merged_output_dir: str | None = None
    output_dir: str = ""
    endpoint: str = "http://127.0.0.1:18889"
    served_model_name: str = "graspo-eval"
    #: 显式卡列表，如 ``"0,1"``。空 = 未指定 = 拒绝启动（fail-closed）。
    gpus: str = ""
    max_workers: int = 8
    seed: int | None = 42
    graspo_threshold_pp: float = 20.0
    sft_threshold_percent: float = 50.0

    @model_validator(mode="after")
    def _validate_eval_contract(self) -> "EvalConfig":
        """加载即校验：必填路径 + 卡计划合法性 + 产物目录在 .local/ 下。

        本模型只由**评测入口**（``graspo eval --eval-config``）构造；训练配置里
        的 ``eval:`` 段是可选的，缺省时 ``GraspoConfig.eval`` 为 ``None``，本模型
        根本不会被构造，因此不触发任何校验（宪法 §2.2：None = 未提供）。
        """
        from graspo.eval.guard import GpuGuardError  # noqa: PLC0415

        missing = [
            name
            for name in ("dataset_path", "base_model_path", "output_dir")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"eval config missing required field(s): {', '.join(missing)}")
        if self.role != "base" and not self.checkpoint_path:
            raise ValueError(
                f"eval.role={self.role!r} requires eval.checkpoint_path; "
                "only role='base' may omit it"
            )
        # 卡计划：必须在加载时就校验，不能等到起容器时才发现写错了 6/7。
        try:
            from graspo.eval.guard import resolve_gpu_plan  # noqa: PLC0415

            resolve_gpu_plan(self.gpus)
        except GpuGuardError as exc:
            raise ValueError(f"eval.gpus invalid: {exc}") from None
        normalized = self.output_dir.replace("\\", "/")
        if not (normalized.startswith(".local/") or "/.local/" in normalized):
            raise ValueError(
                f"eval.output_dir must live under .local/ (got {self.output_dir!r}); "
                "eval artifacts contain host paths and GPU ids and must never be tracked"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """从独立的评测配置 YAML 加载（``graspo eval --eval-config``）。

        校验失败时输出人类可读的信息（哪个字段、期望什么），不抛原始 traceback。
        顶层未知键由 ``extra="forbid"`` 拒绝——不静默忽略拼错的字段。
        """
        import yaml
        from pydantic import ValidationError

        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise SystemExit(f"{config_path}: top level must be a mapping")
        # 允许两种写法：平铺的 eval 字段，或包在 `eval:` 段下（便于与训练配置共用文件）。
        if "eval" in data and isinstance(data["eval"], dict):
            data = dict(data["eval"])
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            _report_config_errors(exc, config_path, cls)
            raise SystemExit(1)


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

    train_method: TrainMethod = "graspo"
    backend: str = "native"
    native: GraspoFlowConfig = GraspoFlowConfig()
    msswift: MsSwiftConfig = MsSwiftConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    lora: LoRAConfig = LoRAConfig()
    export: ExportConfig = ExportConfig()
    eval: EvalConfig | None = None
    launch: LaunchConfig = LaunchConfig()
    reward: RewardConfig = RewardConfig()
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # ── CPT / OPD 的**算法级**配置（后端中立，§1.4）────────────────────────
    # 两个段都只有 `train_method` 取到对应值时才被消费；缺省是**全默认空段**，
    # 因此既有配置（不含这两个键）行为完全不变（向后兼容）。
    pretrain: PretrainConfig = PretrainConfig()
    distill: DistillConfig = DistillConfig()
    # ── 训练参数化模式（native / msswift 两个后端的唯一开关）────────────────
    # None = 未指定 ⇒ lora（向后兼容，v0.24 之前的所有配置都没有这个字段）。
    # 取 full = 全参（全量）微调：native 侧放开全部基座参数的 requires_grad，
    # msswift 侧映射为 `--tuner_type full`。
    # 消费点只读 ``effective_tuner_type``（归一后再用），不直接读原始字段，
    # 避免每个调用点各自判 None（宪法 §1.4 单一真相源）。
    tuner_type: TunerType | None = None

    @field_validator("eval", mode="before")
    @classmethod
    def _eval_section_is_optional(cls, value: Any) -> Any:
        """显式 ``eval: null`` 等价于"未提供该段"（与其它段 ``section: null`` 的约定一致）。

        缺省（键不存在）时 pydantic 直接用默认值 ``None``，不会走到这里。
        """
        if value is None:
            return None
        return value

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

    @model_validator(mode="after")
    def _validate_native_offload_combination(self) -> GraspoConfig:
        """native 优化器态 offload 的非法组合在加载时即拒绝（§2.3）。

        纯逻辑委托给 :func:`validate_native_offload_combination`，使该校验可在
        无 torch 的机器上单测。**只加不改**：``native.offload_optimizer_state``
        默认 false，既有配置不含该键 ⇒ 行为逐位不变。
        """
        validate_native_offload_combination(
            offload_optimizer_state=bool(self.native.offload_optimizer_state),
            backend=self.backend,
            native_adapter=self.native.adapter,
        )
        return self

    @model_validator(mode="after")
    def _validate_train_method_combination(self) -> GraspoConfig:
        """``train_method`` × ``backend`` × 教师配置的非法组合在加载时即拒绝（§2.3）。

        纯逻辑委托给 :func:`validate_train_method_combination`，使该校验可在无 torch
        的机器上单测。**只加不改**：既有的 ``graspo`` / ``sft`` 组合全部落在合法集合内，
        行为不变。
        """
        validate_train_method_combination(
            train_method=self.train_method,
            backend=self.backend,
            distill_teacher_model_path=self.distill.teacher_model_path,
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
        # 注意：`eval` **不在**这个列表里。它与其它段语义不同——是一段"可选功能"，
        # 缺省时必须是 None（未提供），不能归一成 {}（空段）。
        # 归一成 {} 会构造一个字段全空的 EvalConfig，触发它的必填校验，
        # 从而让**所有不含 eval: 的既有训练配置加载失败**（实测回归，已修）。
        for section in (
            "msswift",
            "model",
            "data",
            "lora",
            "export",
            "launch",
            "reward",
            "training",
            "pretrain",
            "distill",
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


def _unwrap_optional_model(annotation: Any) -> Any:
    """从 ``X | None`` 里取回 ``X``（用于错误信息里列字段名）。

    非 Union 时原样返回。取不到 BaseModel 时返回一个字段为空的占位类，
    保证错误报告路径**永远不会自己抛异常**——错误辅助函数崩掉会让用户
    看不到真正的原因（宪法 §13.1：异常必须转成可读信息）。
    """
    for candidate in getattr(annotation, "__args__", ()) or ():
        if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
            return candidate
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return annotation

    class _Unknown(BaseModel):
        model_config = ConfigDict(extra="forbid")

    return _Unknown


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
    # 注意：段可能是 Optional（如 `eval: EvalConfig | None`），需先解包再读 model_fields，
    # 否则 UnionType 上没有 model_fields，报错辅助函数自己会崩。
    config_section_fields: dict[str, list[str]] = {}
    for section in (
        "native",
        "msswift",
        "model",
        "data",
        "lora",
        "export",
        "eval",
        "launch",
        "reward",
        "training",
        "pretrain",
        "distill",
    ):
        if section not in config_cls.model_fields:
            config_section_fields[section] = []
            continue
        config_section_fields[section] = sorted(
            _unwrap_optional_model(config_cls.model_fields[section].annotation).model_fields
        )
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
