"""Layer 1 — TransformerAdapter: common adapter logic for all decoder-only transformers.

Extracted from the original Qwen adapter.  Every model family subclasses this and
only implements the model-specific parts (``_load_model``,
``generate_groups``, ``generate_sample_groups``, ``train_batch``,
``sequence_log_probs``, ``parse_completion``).
"""

import math
import datetime
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.adapters.base_graspo_flow_adapter import BaseGraspoFlowAdapter
from graspo.flow.logging import append_jsonl_segment, rank_metrics_filename, run_log_dir
from graspo.flow.parallel.placement_plan import (
    NativePlacementPlan,
    placement_summary,
)
from graspo.flow.parallel.state import GraspoFlowState, destroy_parallel_state
from graspo.flow.parallel.tensor_utils import (
    _cuda_memory_snapshot,
    _jsonable,
    _mean_present,
    _rollout_timing_summary,
    _scale_rollout_timings,
    _set_tensor_parallel_group,
)
from graspo.flow.runtime import NativeGeneration
from graspo.ripple.multimodal.rows import attach_rows

logger = logging.getLogger("graspo.flow")


def apply_config_optimizer_hyperparams(
    optimizer: Any,
    scheduler: Any,
    *,
    learning_rate: float,
    weight_decay: float,
) -> list[str]:
    """配置优先：resume 恢复的 optimizer/scheduler 状态可能携带旧训练超参。

    ``optimizer.load_state_dict`` 会把 checkpoint 里保存的 param_groups 整体
    替换为旧值（含 lr/weight_decay），constant 调度器（None）下没有任何机制
    再把它改回配置值——此前这是静默覆盖（配置改了不生效）。这里以配置为准
    覆盖旧值并返回被覆盖项的描述列表，由调用方 WARNING 告知（§3.2 透明退路）。

    scheduler（LambdaLR）的步进曲线锚定在 ``base_lrs`` 上，必须同步到配置值；
    ``last_epoch`` 保留 checkpoint 进度，保证调度曲线连续性。
    """
    overridden: list[str] = []
    if optimizer is not None:
        for group in optimizer.param_groups:
            group_lr = float(group.get("lr", 0.0))
            if group_lr != learning_rate:
                overridden.append(
                    f"learning_rate: checkpoint={group_lr:g} config={learning_rate:g}"
                )
                group["lr"] = learning_rate
            group_wd = float(group.get("weight_decay", 0.0))
            if group_wd != weight_decay:
                overridden.append(f"weight_decay: checkpoint={group_wd:g} config={weight_decay:g}")
                group["weight_decay"] = weight_decay
    if scheduler is not None and getattr(scheduler, "base_lrs", None):
        for i, base_lr in enumerate(scheduler.base_lrs):
            if float(base_lr) != learning_rate:
                scheduler.base_lrs[i] = learning_rate
        if getattr(scheduler, "_last_lr", None):
            # 刷新缓存的最近 lr，避免下一次 step 前读到 checkpoint 旧值
            scheduler._last_lr = [float(learning_rate)] * len(scheduler._last_lr)
    return overridden


def _normalize_optimizer_state_to_params(optimizer: Any) -> list[str]:
    """Relocate/cast every optimizer state tensor onto its owning parameter's
    device and dtype (defensive normalization).

    Each DP rank now loads its own checkpoint shard with ``map_location=self.device``,
    so optimizer state should already live on the local device.  Keep this as a
    boundary guard (§2.3): ``optimizer.load_state_dict`` relocation behavior has
    changed across PyTorch versions (modern torch moves ``exp_avg`` / ``exp_avg_sq``
    to the parameter device, but not always a non-fused ``step``), and any residual
    device/dtype mismatch would make ``AdamW.step`` fail with "Tensors of the same
    index must be on the same device and the same dtype".  This helper moves every
    state tensor to the parameter's device and casts it to the parameter's dtype
    (``step`` stays float32, the optimizer convention).  Returns the list of tensor keys
    normalized so callers can emit a transparent WARNING (§3.2 透明退路).
    """
    normalized: list[str] = []
    if optimizer is None:
        return normalized
    for param, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        target_device = param.device
        target_dtype = param.dtype
        for key, value in list(state.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if value.device == target_device and value.dtype == target_dtype:
                continue
            if key == "step":
                state[key] = value.to(device=target_device, dtype=torch.float32)
            else:
                state[key] = value.to(device=target_device, dtype=target_dtype)
            normalized.append(str(key))
    return normalized


#: `global_grad_norm_mean` 的**口径标签**（随值落盘，防止误读；2026-09-22 裁定）。
#: `global_grad_norm_l2` 的**分模式口径**（2026-09-22 离线重算执行者指出 + 指挥官裁定）：
#: 同一个 `sqrt(Σ‖g_r‖²)` 在**分片**与**DP 复制**下语义**不同**，必须如实标注，不许 over-claim。
GRAD_NORM_L2_CALIBER_SHARDED = (
    "per_rank_l2_composition（= sqrt(Σ‖g_r‖²)；PP/TP 各 rank 持**不同分片** ⇒ 等于拼接后的"
    "**全模型梯度范数**）"
)
GRAD_NORM_L2_CALIBER_DP = (
    "per_rank_l2_composition（= sqrt(Σ‖g_r‖²)；**DP 下各 rank 持同一份参数、梯度经 all-reduce 平均**"
    " ⇒ 该式是 ‖mean(g_r)‖ 的**上界**，**不等于**全模型梯度范数，也**不等于**更新步长；"
    "对照见 global_grad_norm_dp_mean）"
)
GRAD_NORM_L2_CALIBER_SINGLE = "single_rank（单卡/单进程 ⇒ 就是**全模型梯度范数**）"
GRAD_NORM_L2_CALIBER_UNDETERMINED = (
    "per_rank_l2_composition（**并行布局未判定**：既有分片又有复制，或字段缺失 ⇒ "
    "**不声明**它等于全模型范数）"
)


def grad_norm_l2_caliber(dp_size: int, tp_size: int, pp_size: int) -> str:
    """按并行布局给出 `global_grad_norm_l2` 的**如实口径**（不猜：判不了就说判不了）。

    - 纯 DP 复制（dp>1、tp=pp=1）⇒ 各 rank 同一份参数、梯度被平均 ⇒ L2 合成是**上界**；
    - 有分片（tp>1 或 pp>1）⇒ 各 rank 持不同参数 ⇒ L2 合成 = 拼接后的全模型范数；
    - 单卡（全 1）⇒ 平凡等于全模型范数；
    - 混合（dp>1 且 tp/pp>1）⇒ 两者叠加，无简单等式 ⇒ 标注"未判定"，**不 over-claim**。
    """
    if dp_size <= 1 and tp_size <= 1 and pp_size <= 1:
        return GRAD_NORM_L2_CALIBER_SINGLE
    if tp_size == 1 and pp_size == 1 and dp_size > 1:
        return GRAD_NORM_L2_CALIBER_DP
    if dp_size == 1 and (tp_size > 1 or pp_size > 1):
        return GRAD_NORM_L2_CALIBER_SHARDED
    return GRAD_NORM_L2_CALIBER_UNDETERMINED


GRAD_NORM_MEAN_CALIBER = (
    "per_rank_mean（= 各 rank 梯度 L2 范数的算术平均；"
    "**≠ 全模型梯度范数**（PP/TP 各 rank 只持部分参数，正确口径见 global_grad_norm_l2）、"
    "**≠ 用于更新的步长**（更新走裁剪后梯度，见 max_grad_norm））"
)


def _l2_norm_of_rank_norms(values) -> float | None:
    """各 rank 范数的 **L2 合成**：``sqrt(Σ‖g_r‖²)``——分片参数下的正确全局范数。

    为什么不是平均：PP/TP 下每个 rank 只持有部分参数的梯度，梯度向量在**参数维度**上
    相互正交地拼成全局向量 ⇒ 全局范数是各段范数的平方和开方，**不是**算术平均。
    只对 finite 且非 None 的读数求和；一个都没有 ⇒ 返回 ``None``（不猜，§2.2）。
    """
    total = 0.0
    seen = False
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            continue
        total += number * number
        seen = True
    return math.sqrt(total) if seen else None


class TransformerAdapter(BaseGraspoFlowAdapter):
    """Common adapter for all decoder-only transformer models.

    Provides:
    - Distributed initialization (``_setup_distributed``)
    - Tokenizer / Processor loading (``_load_tokenizer``)
    - Chat template formatting (``_format_messages``)
    - Checkpoint save/restore format (``save_checkpoint``, ``load_checkpoint``)
    - Training-loop helpers (``_shared_training_indices``, ``_aggregate_rank_metrics``)
    - Memory events (``_emit_rank_memory_event``)
    - Generation helpers (``_generation_from_sequences``, chunk-size helpers)

    Subclasses must implement:
    - ``_load_model()``
    - ``generate_groups()``
    - ``generate_sample_groups()``
    - ``train_batch()``
    - ``sequence_log_probs()``
    - ``parse_completion()``
    """

    # ── Initialization ──────────────────────────────────────────────────────

    def __init__(self, config: Any) -> None:
        self.config = config
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.tp_size = int(config.native.tp_size)
        self.tp_rank = 0
        self.dp_size = int(config.native.dp_size)
        self.dp_rank = 0
        self.pp_size = int(config.native.pp_size)
        self.pp_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tp_state: GraspoFlowState | None = None
        self.model: Any = None
        self.placement: NativePlacementPlan | None = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self._train_batch_call_index = 0

    # ── Setup (template method) ─────────────────────────────────────────────

    def setup(self) -> None:
        self._setup_distributed()
        self._patch_transformers_float8_import_compat()
        from transformers import AutoProcessor, AutoTokenizer

        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model.model_path does not exist: {model_path}")

        hf_config = self._load_native_qwen_config(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if bool(getattr(hf_config, "has_vision_config", False)):
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._load_model(hf_config, model_path)
        self._build_optimizer()
        self._emit_setup_event()

    @abstractmethod
    def _load_model(self, hf_config: Any, model_path: Path) -> None:
        """Load the model.  Subclass implements."""

    def _build_optimizer(self) -> None:
        from graspo.ripple.loss import GRASPORippleLoss

        self.loss_fn = GRASPORippleLoss(self.config.training.policy_ratio_clip_eps)
        # 优化器收参路径对两种模式**是同一条**：一律只收 ``requires_grad`` 的参数。
        # 区别在模型侧——lora 模式只有 ``lora_*`` 矩阵被放开，full 模式全部参数被放开
        # （``flow/adapters/models/*/model.py``）。因此这里不需要按模式分支，只需把
        # "全参却没收到参数"这种静默失败挡掉（宪法 §2.3 边界校验即防呆）。
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if self.config.effective_tuner_type == "full" and not trainable:
            raise RuntimeError(
                "tuner_type=full but no parameter has requires_grad=True — the native "
                "full-parameter entry did not take effect (optimizer would be None and "
                "training would silently do nothing)."
            )
        self.optimizer = (
            torch.optim.AdamW(
                trainable,
                lr=self.config.training.effective_learning_rate(dp_size=self.config.native.dp_size),
                weight_decay=self.config.training.weight_decay,
            )
            if trainable
            else None
        )
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> Any:
        """根据 ``lr_scheduler`` 配置构建学习率调度器，默认返回 None（恒定 LR）。"""
        import math

        sched_cfg = self.config.training.lr_scheduler
        if sched_cfg.type == "constant":
            return None

        base_lr = float(
            self.config.training.effective_learning_rate(dp_size=self.config.native.dp_size)
        )
        warmup_steps = max(0, int(sched_cfg.warmup_steps))
        min_lr = base_lr * float(sched_cfg.min_lr_ratio)

        # decay_steps 是纯调度参数（warmup 后的衰减跨度，optimizer-step 粒度）。
        # 训练长度只由 training.max_epochs 控制（v0.23.0 起 max_steps 已移除）——
        # 调度器不读任何训练长度配置；衰减完成后 lr 保持 min_lr（progress clamp）。
        decay_steps = max(0, int(sched_cfg.decay_steps))
        if decay_steps <= 0:
            raise ValueError(
                f"lr_scheduler.type 非 constant 时，lr_scheduler.decay_steps "
                f"必须 > 0，got {decay_steps}"
            )

        if sched_cfg.type == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = min(float(step - warmup_steps) / decay_steps, 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (min_lr + (base_lr - min_lr) * cosine) / base_lr

        elif sched_cfg.type == "linear":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = min(float(step - warmup_steps) / decay_steps, 1.0)
                return (min_lr + (base_lr - min_lr) * (1.0 - progress)) / base_lr

        else:
            raise ValueError(
                f"不支持的 lr_scheduler.type: {sched_cfg.type}，可选: constant, cosine, linear"
            )

        assert self.optimizer is not None, "optimizer must be built before scheduler"
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _emit_setup_event(self) -> None:
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self._emit_rank_memory_event(
            "setup_after",
            {
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "activation_checkpointing_enabled": bool(
                    getattr(self.model, "gradient_checkpointing", False)
                ),
                "lora_target_modules": sorted(self.model.lora_targets),
                "lora_target_signature": self.model.lora_target_signature(),
                "rollout_kv_cache_supported": bool(getattr(self.model, "supports_kv_cache", True)),
                "placement": placement_summary(self.placement) if self.placement else {},
                "micro_batch_size": self.config.native.micro_batch_size,
                "empty_cache_after_rollout_split": (
                    self.config.native.empty_cache_after_rollout_split
                ),
                "synchronize_cuda_timing": self.config.native.synchronize_cuda_timing,
            },
        )
        self._print_rank0(
            {
                "event": "adapter_ready",
                "rank": self.rank,
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "group_batch_semantics": (
                    "rollout_prompt_queue_batch_size prompts, each with rollout_group_size "
                    "completions per TP forward batch when budget permits"
                ),
                "activation_checkpointing_enabled": bool(
                    getattr(self.model, "gradient_checkpointing", False)
                ),
                "lora_target_modules": sorted(self.model.lora_targets),
                "lora_target_signature": self.model.lora_target_signature(),
                "rollout_kv_cache_supported": bool(getattr(self.model, "supports_kv_cache", True)),
                "placement": placement_summary(self.placement) if self.placement else {},
            }
        )

    # ── Distributed setup ───────────────────────────────────────────────────

    def _setup_distributed(self) -> None:
        state = GraspoFlowState.initialize(self.tp_size, self.pp_size, self.dp_size)
        self.tp_state = state
        self.rank = state.rank
        self.local_rank = state.local_rank
        self.world_size = state.world_size
        self.tp_rank = state.tp_rank
        self.dp_rank = state.dp_rank
        self.pp_rank = state.pp_rank
        self.device = state.device
        _set_tensor_parallel_group(state.tp_group, state.tp_size)

    def _load_native_qwen_config(self, model_path: Path) -> Any:
        from graspo.flow.adapters.models.common.model_builders import load_native_qwen_config

        return load_native_qwen_config(model_path)

    def _patch_transformers_float8_import_compat(self) -> None:
        # 运行时探测 PyTorch 版本以兼容 float8 类型（非接口探测，是外部库版本检测）
        if not hasattr(torch, "float8_e8m0fnu"):
            torch.float8_e8m0fnu = torch.uint8  # type: ignore[attr-defined]

    # ── Chat template ───────────────────────────────────────────────────────

    def format_messages(
        self,
        messages: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        return self._format_messages(messages, chat_template_kwargs, tools=tools)

    def _format_messages(
        self,
        messages: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__} is not set up; call setup() first")
        template_kwargs = dict(chat_template_kwargs or {})
        if tools is not None:
            template_kwargs["tools"] = tools
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        tools_text = ""
        if tools is not None:
            tools_text = "\n\ntools: " + json.dumps(tools, ensure_ascii=False)
        return (
            "\n\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in messages
            )
            + tools_text
        )

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        trainer_state: dict[str, Any] | None = None,
    ) -> None:
        self._require_ready()
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} is not set up; call setup() first")
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        full_param = self.config.effective_tuner_type == "full"
        payload = {
            "adapter": "native",
            "tuner_type": self.config.effective_tuner_type,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "placement": placement_summary(self.placement) if self.placement is not None else None,
            "lora_target_signature": self.model.lora_target_signature(),
            "lora_tensor_metadata": self.model.lora_tensor_metadata(),
            "lora_state_dict": self.model.lora_state_dict(),
            # 全参模式的训练状态不在 ``lora_state_dict`` 里（它只收 ``lora_`` 名字）。
            # 不补这一项的话 save/load 会"静默成功但什么都没恢复"——详见 load_checkpoint
            # 的同名分支。只存可训参数（optimizer 会改的就是这些）。
            "full_param_state_dict": (
                {
                    name: param.detach().cpu()
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                }
                if full_param
                else None
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
            "adapter_state": {
                "train_batch_call_index": self._train_batch_call_index,
            },
            "trainer_state": trainer_state,
            "config": self.config.model_dump(),
        }
        # DP: 每个 DP rank 都写自己的 shard（权重梯度已同步，LoRA/optimizer/RNG/调度器
        # 状态各 rank 独立保存）。resume 时各 rank 从共享文件系统读回自己的 shard，
        # 不再依赖 object-collective 广播——那会保留 sender 的 CUDA device index，
        # 把整套 optimizer state 反序列化到 cuda:0，造成 GPU0 显存偏高 + 每 rank 在
        # 非本卡上建 CUDA context（恰好是本 bug 的两个症状）。
        torch.save(
            payload,
            output / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}_pp_{self.pp_rank:02d}.pt",
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self._emit_rank_memory_event("checkpoint_after", {"checkpoint_dir": str(output)})
        if self.rank == 0 and self.dp_rank == 0:
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "native-lora" if not full_param else "native-full-param",
                        "tuner_type": self.config.effective_tuner_type,
                        "tp_size": self.tp_size,
                        "dp_size": self.dp_size,
                        "pp_size": self.pp_size,
                        "placement": (
                            placement_summary(self.placement)
                            if self.placement is not None
                            else None
                        ),
                        "lora_target_signature": self.model.lora_target_signature(),
                        "world_size": self.world_size,
                        "checkpoint_type": (
                            "recoverable_lora_training_state"
                            if not full_param
                            else "recoverable_full_param_training_state"
                        ),
                        "has_trainer_state": trainer_state is not None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None:
        self._require_ready()
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} is not set up; call setup() first")
        checkpoint_dir = Path(path)
        # DP: 每个 DP rank 直接从共享文件系统读自己的 shard。每个 rank 保存时都把
        # LoRA/optimizer/RNG/调度器状态放在自己的 device 上，load 用 map_location=
        # self.device 即可正确恢复。之前用 broadcast_object_list 把 rank0 的 CUDA
        # payload 广播到同组 rank，会保留 sender 的 device index，让所有接收 rank 的
        # optimizer state 停留在 cuda:0——这正是“显存不均 + 每 GPU 多进程”的根因。
        rank_path = (
            checkpoint_dir / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}_pp_{self.pp_rank:02d}.pt"
        )
        if not rank_path.exists():
            raise FileNotFoundError(
                "Missing current GRASPO checkpoint shard "
                f"for rank={self.rank} tp_rank={self.tp_rank} pp_rank={self.pp_rank}: {rank_path}"
            )
        try:
            payload = torch.load(rank_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(rank_path, map_location=self.device)
        if payload is None:
            raise RuntimeError(
                f"Failed to load checkpoint for rank={self.rank} "
                f"dp_rank={self.dp_rank} tp_rank={self.tp_rank} pp_rank={self.pp_rank}"
            )
        if int(payload.get("tp_size", self.tp_size)) != self.tp_size:
            raise ValueError(
                f"Checkpoint TP size {payload.get('tp_size')} does not match runtime "
                f"TP size {self.tp_size}"
            )
        if int(payload.get("dp_size", self.dp_size)) != self.dp_size:
            raise ValueError(
                f"Checkpoint DP size {payload.get('dp_size')} does not match runtime "
                f"DP size {self.dp_size}"
            )
        if int(payload.get("pp_size", self.pp_size)) != self.pp_size:
            raise ValueError(
                f"Checkpoint PP size {payload.get('pp_size')} does not match runtime "
                f"PP size {self.pp_size}"
            )
        if self.config.effective_tuner_type == "full":
            # 全参模式：权重在 ``full_param_state_dict`` 里。**必须** fail-closed 地
            # 校验它存在——旧格式（或误配）的 checkpoint 只有空的 lora_state_dict，
            # 走下面的 LoRA 分支会"load 成功但一个权重都没恢复"，那是静默训错。
            full_state = payload.get("full_param_state_dict")
            if full_state is None:
                raise RuntimeError(
                    "Checkpoint shard has no full-parameter state dict while "
                    "tuner_type=full: this shard was written by a LoRA run (or an "
                    "incompatible version). Refusing to resume, because loading it "
                    "would silently leave the base weights at initialization."
                )
            # strict=True：PP/TP 分片是按 rank 存的，本 rank 读自己的 shard，
            # 键集合必须逐字匹配；不匹配就是布局变了，当场报错。
            self.model.load_state_dict(full_state, strict=True)
        else:
            checkpoint_signature = payload.get("lora_target_signature")
            current_signature = self.model.lora_target_signature()
            if checkpoint_signature is not None and checkpoint_signature != current_signature:
                raise ValueError(
                    "Checkpoint LoRA target signature does not match runtime configuration: "
                    f"checkpoint={checkpoint_signature}, runtime={current_signature}"
                )
            missing, unexpected = self.model.load_state_dict(
                payload["lora_state_dict"], strict=False
            )
            unexpected_lora = [name for name in unexpected if "lora_" in name]
            if unexpected_lora:
                raise RuntimeError(f"Unexpected LoRA tensors in checkpoint: {unexpected_lora}")
            missing_lora = [name for name in missing if "lora_" in name]
            if missing_lora:
                raise RuntimeError(f"Missing LoRA tensors while loading checkpoint: {missing_lora}")
        optimizer_state = payload.get("optimizer_state_dict")
        normalized_optimizer_state: list[str] = []
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            # 防呆（§2.3 边界校验即防呆）：即便每个 rank 用 map_location=self.device
            # 读自己的 shard，仍可能因 torch 版本差异或旧格式残留 device/dtype 不一致，
            # 统一迁回参数 device/dtype，避免 optimizer.step 报 device/dtype 不一致。
            normalized_optimizer_state = _normalize_optimizer_state_to_params(self.optimizer)
            if normalized_optimizer_state:
                logger.warning(
                    "Normalized %d optimizer state tensor(s) to matching parameter "
                    "device/dtype after checkpoint load (DP broadcast preserves sender "
                    "device); checkpoint=%s",
                    len(normalized_optimizer_state),
                    checkpoint_dir.name,
                )
        elif self.optimizer is not None and optimizer_state is None:
            raise RuntimeError("Checkpoint shard is missing optimizer state for a trainable rank")
        scheduler_state = payload.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        # 配置优先（§2.2 显式优于隐式）：checkpoint 可能携带旧训练超参
        # （lr/weight_decay），load_state_dict 整体替换后静默生效——必须按
        # 当前配置覆盖，并 WARNING 告知用户（§3.2 透明退路，防静默坏退路）。
        overridden = apply_config_optimizer_hyperparams(
            self.optimizer,
            self.scheduler,
            learning_rate=float(
                self.config.training.effective_learning_rate(dp_size=self.config.native.dp_size)
            ),
            weight_decay=float(self.config.training.weight_decay),
        )
        if overridden:
            logger.warning(
                "Checkpoint hyperparameters overridden by config (config takes precedence; "
                "checkpoint=%s): %s",
                checkpoint_dir.name,
                "; ".join(overridden),
            )
        torch.set_rng_state(payload["torch_rng_state"].detach().cpu())
        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state.to("cpu"), self.device)
        adapter_state = payload.get("adapter_state") or {}
        self._train_batch_call_index = int(adapter_state.get("train_batch_call_index") or 0)
        self._emit_rank_memory_event(
            "checkpoint_loaded",
            {
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_rank_file": str(rank_path),
                "has_trainer_state": payload.get("trainer_state") is not None,
                "train_batch_call_index": self._train_batch_call_index,
                "overridden_hyperparams": overridden,
            },
        )
        trainer_state = payload.get("trainer_state")
        # 释放 checkpoint 载荷里的大对象引用：lora/optimizer/scheduler 状态已在
        # load_state_dict 时拷贝到模型/优化器（这里是载荷里的原始 CUDA tensor，不再需要）。
        # 随后 empty_cache 把缓存分配器保持的 reserved 高水位归还驱动，消除“resume
        # 广播缓冲区未释放”导致的显存统计偏高。checkpoint 加载是一次性启动事件，
        # 此处调用一次 empty_cache 成本可忽略（§3.1 同效退路）。
        payload.pop("optimizer_state_dict", None)
        payload.pop("lora_state_dict", None)
        payload.pop("scheduler_state_dict", None)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return trainer_state

    # ── Training helpers ────────────────────────────────────────────────────

    def _shared_training_indices(self, experience_count: int, *, optimize_round: int) -> list[int]:
        """Return shuffled training indices, deterministic per dp_rank.

        All ranks within the same dp_rank (same model replica) compute the same
        shuffle locally using the deterministic seed.  No cross-dp_rank
        communication is needed — each dp_rank trains on its own data.
        """
        import random

        indices = list(range(experience_count))
        seed = (
            int(self.config.training.seed)
            + (self._train_batch_call_index * 1_000_003)
            + int(optimize_round)
        )
        if dist.is_available() and dist.is_initialized():
            # 同一 dp_rank 内各 rank 用相同 seed 本地 shuffle，无需通信
            random.Random(seed + self.dp_rank).shuffle(indices)
            return indices
        random.Random(seed).shuffle(indices)
        return indices

    def _shared_generation_micro_batch_size(
        self,
        *,
        prompt_len: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        # User controls this directly via micro_batch_size, clamped to available rows.
        return max(
            1,
            min(
                int(self.config.native.micro_batch_size),
                int(rollout_group_size),
            ),
        )

    def _shared_rollout_prompt_chunk_size(
        self,
        *,
        prompt_len: int,
        requested_prompt_count: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        # Encode all queued samples together (no more auto-chunking).
        return max(1, int(requested_prompt_count))

    @staticmethod
    def _loss_bearing_ranks(ranks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        """只保留**真正产出 loss 的 rank**（PP 下即末 stage）；返回 ``(ranks, 口径标签)``。

        为什么必须这么做（P1 侦察 + 指挥官裁定，2026-09-22）：
        非末 PP stage 不计算 loss，其 ``loss_mean`` 是**结构性 0**；旧实现把它算进分母，
        导致 **SFT·PP 下 ``global_loss_mean`` 被系统性减半**（实测 T017 step1：
        真实 0.020874 → 报告 0.010437）。

        两级判据（**精度优先，回落必标注**）：

        1. 所有 rank 都带 ``pp_rank`` ⇒ 取 ``pp_rank == pp_size - 1``（**精确**）；
        2. 缺 ``pp_rank``（历史落盘）但 ``pp_size > 1`` ⇒ 取 ``loss_mean != 0`` 的 rank
           （结构性 0 的启发式），口径标签标 ``non_zero_loss_stage_fallback``——**如实标注**，
           不假装精确；
        3. 其余（单卡 / 拿不到 pp 信息）⇒ 全部 rank，标签 ``all_ranks``。

        单卡下三种路径等价（ranks 只有 1 个），行为不变。
        """
        if not ranks:
            return ranks, "all_ranks"
        pp_sizes = {item.get("pp_size") for item in ranks}
        if len(pp_sizes) == 1 and next(iter(pp_sizes)) is not None:
            pp_size = int(next(iter(pp_sizes)))
        else:
            pp_size = 0
        if pp_size > 1 and all(item.get("pp_rank") is not None for item in ranks):
            return [
                item for item in ranks if int(item.get("pp_rank") or 0) == pp_size - 1
            ] or ranks, "last_pp_stage_only"
        if pp_size > 1:
            non_zero = [
                item for item in ranks if float(item.get("loss_mean") or 0.0) != 0.0
            ]
            if non_zero:
                return non_zero, "non_zero_loss_stage_fallback"
        return ranks, "all_ranks"
    def _aggregate_rank_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        # ★ 带上 `pp_rank`（`_aggregate_rank_metrics` 精确聚合 loss 的前提）：它由本类
        #   自身持有，**无需改训练代码**。历史落盘没有该键 ⇒ 聚合退回启发式并标注口径。
        local = {
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            **metrics,
        }
        if not (dist.is_available() and dist.is_initialized()):
            # ★ 单进程 / 单卡（world_size==1）：**也必须产出 global_* 键**。
            #
            # 为什么（2026-09-20 实测缺陷）：``parallel/state.py`` 只在
            # ``world_size > 1`` 时 ``init_process_group``，所以单卡档走的就是这个
            # 分支。旧实现只回 ``rank_metrics``、**一个 global_* 键都不给**，于是：
            #   ① ``flow/trainer/sft_trainer.py`` 读 ``metrics["global_loss_mean"]``
            #      拿到 None ⇒ stdout 逐步写 ``loss: null grad_norm: null`` +
            #      ``loss_scope: "global"`` —— 一个自称"全局口径"的逐行 null，
            #      被 collector 如实识别为「数值不可得」⇒ A6 fail-closed 判否；
            #   ② 采集层读 ``global_loss_mean`` / ``global_grad_norm_mean`` /
            #      ``global_optimizer_steps_sum`` 全部取不到 ⇒ 即便 phase 名已认，
            #      单卡档的 A2/A6 依然是取证缺口。
            # 两者都是**伪否**：数据本来就在本地 metrics 里，只是键名随分支而变。
            #
            # 单卡下"局部 == 全局"在数学上平凡成立（ranks 只有 1 个，
            # ``_mean_present`` 退化为该值本身；sum 退化为该值），所以补键**不是**
            # 放松任何断言，只是把"键名取决于分支"这个隐式契约（§2.2）改成
            # "键名与分支无关"的显式契约（§1.4 单一真相源：全局读数的形状只有一份）。
            # PP 路径（dist 已初始化）的输出逐字不变。
            return {
                **metrics,
                "rank": self.rank,
                "tp_rank": self.tp_rank,
                "rank_metrics": [local],
                "global_optimizer_steps_sum": int(local.get("optimizer_steps") or 0),
                "global_nonzero_grad_count_sum": int(local.get("nonzero_grad_count") or 0),
                "global_loss_mean": local.get("loss_mean"),
                "global_grad_norm_mean": local.get("grad_norm_mean"),
                "global_grad_norm_l2": local.get("grad_norm_mean"),
                "global_grad_norm_l2_caliber": GRAD_NORM_L2_CALIBER_SINGLE,
                "global_grad_norm_dp_mean": local.get("grad_norm_mean"),
                "global_lora_norm_delta_mean": local.get("lora_norm_delta"),
                "global_trainable_norm_delta_mean": local.get("trainable_norm_delta"),
                "grad_count_metric": local.get("grad_count_metric"),
            }
        gathered: list[dict[str, Any] | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local)
        ranks = [item for item in gathered if item is not None]
        loss_ranks, loss_caliber = self._loss_bearing_ranks(ranks)
        return {
            **metrics,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "rank_metrics": ranks,
            "global_optimizer_steps_sum": sum(
                int(item.get("optimizer_steps") or 0) for item in ranks
            ),
            "global_nonzero_grad_count_sum": sum(
                int(item.get("nonzero_grad_count") or 0) for item in ranks
            ),
            # ★ 口径修正（2026-09-22）：loss 只在**真正产出 loss 的 stage** 上聚合；
            #   旧实现把非末 PP stage 的结构性 0 计入分母 ⇒ 系统性减半。
            "global_loss_mean": _mean_present(
                item.get("loss_mean") for item in loss_ranks
            ),
            "global_loss_caliber": loss_caliber,
            "global_loss_rank_count": len(loss_ranks),
            # ★ `global_grad_norm_mean` = **逐 rank 梯度范数的算术平均**（保留原名/原值以兼容
            #   既有消费方与历史可比性）。**它不是全模型梯度范数**（PP/TP 各 rank 只持部分
            #   参数；正确口径是 sqrt(Σ‖g_r‖²)），**也不是用于更新的量**（更新走裁剪后梯度）。
            #   口径标签随值一起落盘 ⇒ 读者不会把"49 → 4.18e9"误读成"梯度爆炸"。
            "global_grad_norm_mean": _mean_present(
                item.get("grad_norm_mean") for item in ranks
            ),
            "global_grad_norm_l2": _l2_norm_of_rank_norms(
                item.get("grad_norm_mean") for item in ranks
            ),
            "global_grad_norm_caliber": GRAD_NORM_MEAN_CALIBER,
            # ★ 分模式口径（同一个 sqrt(Σ‖g_r‖²) 在 DP 与 PP/TP 下语义不同）
            "global_grad_norm_l2_caliber": grad_norm_l2_caliber(
                self.dp_size, self.tp_size, self.pp_size
            ),
            # DP 下的对照值：逐 rank 范数的算术平均（读者可自己看出与 L2 合成差多少）
            "global_grad_norm_dp_mean": _mean_present(
                item.get("grad_norm_mean") for item in ranks
            ),
            # lora 模式：LoRA 权重范数变化；full 模式：该键为 None（无 lora 参数 ⇒
            # 指标不适用，_mean_present 会把 None 过滤掉，不会退化成 0）。
            "global_lora_norm_delta_mean": _mean_present(
                item.get("lora_norm_delta") for item in ranks
            ),
            # full 模式的等价指标（全部可训参数的权重范数变化）；lora 模式下为 None。
            "global_trainable_norm_delta_mean": _mean_present(
                item.get("trainable_norm_delta") for item in ranks
            ),
            # 指标定义标识（lora: nonzero_lora_grads / full: grad_populated_trainable_params）。
            "grad_count_metric": next(
                (item.get("grad_count_metric") for item in ranks if item.get("grad_count_metric")),
                None,
            ),
        }

    # ── Generation helpers ──────────────────────────────────────────────────

    def _generation_from_sequences(
        self,
        *,
        sequences: torch.Tensor,
        prompt_len: int,
        prompt_lens: list[int],
        pad_token_id: int,
        rollout_group_size: int,
        requested_prompt_queue_size: int,
        effective_prompt_queue_size: int,
        use_kv_cache: bool,
        generation_micro_batch_size: int,
        split_count: int,
        tokenize_sec: float,
        chunk_timings: list[dict[str, float | int]],
        timing_divisor: int,
        rollout_started_at: float,
        multimodal_rows: list[dict[str, Any]] | None = None,
    ) -> NativeGeneration:
        if self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__} is not set up; call setup() first")
        attention_mask = sequences.ne(pad_token_id)
        action_mask = torch.zeros(
            (sequences.shape[0], max(sequences.shape[1] - 1, 0)),
            dtype=torch.bool,
            device=self.device,
        )
        if sequences.shape[1] > prompt_len:
            action_mask[:, prompt_len - 1 :] = True
            action_mask &= attention_mask[:, 1:]
        completions = self.tokenizer.batch_decode(
            sequences[:, prompt_len:],
            skip_special_tokens=True,
        )
        metadata = attach_rows({}, multimodal_rows) if multimodal_rows else {}
        return NativeGeneration(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            completions=completions,
            prompt_len=prompt_len,
            metadata={
                **metadata,
                "adapter": "native",
                "rollout_group_size": rollout_group_size,
                "rollout_prompt_queue_batch_size": requested_prompt_queue_size,
                "rollout_prompt_queue_effective_size": effective_prompt_queue_size,
                "rollout_prompt_queue_fallback": effective_prompt_queue_size
                < requested_prompt_queue_size,
                "rollout_use_kv_cache": use_kv_cache,
                "rollout_generation_micro_batch_size": generation_micro_batch_size,
                "rollout_generation_split_count": split_count,
                "rollout_empty_cache_after_split": False,
                **_rollout_timing_summary(
                    tokenize_sec,
                    _scale_rollout_timings(chunk_timings, timing_divisor),
                ),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
                "prefill_len": prompt_len,
                "prompt_lens": prompt_lens,
                "generated_tokens_max": max(int(sequences.shape[1] - prompt_len), 0),
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
            },
        )

    def _pipeline_stage_timing(self) -> dict[str, float | int]:
        from graspo.flow.parallel.tensor_utils import _new_pipeline_stage_timing

        return _new_pipeline_stage_timing()

    def _add_pipeline_stage_timing(
        self, timing: dict[str, float | int], key: str, started_at: float
    ) -> None:
        from graspo.flow.parallel.tensor_utils import _add_pipeline_stage_timing

        _add_pipeline_stage_timing(timing, key, started_at)

    def _round_pipeline_stage_timing(
        self, timing: dict[str, float | int]
    ) -> dict[str, float | int]:
        from graspo.flow.parallel.tensor_utils import _round_pipeline_stage_timing

        return _round_pipeline_stage_timing(timing)

    # ── Utility ─────────────────────────────────────────────────────────────

    def generate_group(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> NativeGeneration:
        return self.generate_groups(
            message_batches=[messages],
            tool_batches=[tools],
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
        )[0]

    def is_primary(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        destroy_parallel_state()

    def _require_ready(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__} is not set up")

    def _print_rank0(self, payload: dict[str, Any]) -> None:
        if self.rank == 0:
            logging.getLogger("graspo.adapter").info(json.dumps(payload, ensure_ascii=False))

    def _sync_timing(self) -> None:
        if (
            bool(self.config.native.synchronize_cuda_timing)
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def _is_pipeline_parallel(self) -> bool:
        return bool(self.placement is not None and self.placement.is_pipeline)

    def _current_lr(self) -> float:
        """返回当前学习率（优先从 scheduler 读取，否则从 optimizer 读取）。"""
        if self.scheduler is not None:
            return float(self.scheduler.get_last_lr()[0])
        if self.optimizer is not None:
            return float(self.optimizer.param_groups[0]["lr"])
        return float(
            self.config.training.effective_learning_rate(dp_size=self.config.native.dp_size)
        )

    def _emit_rank_memory_event(self, phase: str, extra: dict[str, Any] | None = None) -> None:
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": "rank_memory",
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": phase,
            # §10.1：日志不得含不可复现的运行环境元数据（GPU ID/local_rank 已移除）。
            # rank/tp_rank/tp_size 是 config 决定的可复现执行拓扑；memory 快照
            # 属机器负载诊断数据，显式标记为非产物。
            "kind": "diagnostic",
            "reproducible": False,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "memory": _cuda_memory_snapshot(self.device),
        }
        if extra:
            payload.update(extra)
        path = run_log_dir(output_dir) / rank_metrics_filename(self.rank)
        append_jsonl_segment(path, _jsonable(payload))

    def _encode_multimodal_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from graspo.flow.adapters.multimodal_tensors import (
            _messages_from_multimodal_row,
            _processor_chat_messages,
            _tools_for_chat_template,
            _tools_from_multimodal_row,
        )

        if self.processor is None:
            raise RuntimeError(
                "This model did not expose an AutoProcessor; image/video samples cannot be encoded"
            )
        messages = [_processor_chat_messages(_messages_from_multimodal_row(row)) for row in rows]
        tool_batches = [_tools_from_multimodal_row(row) for row in rows]
        # 外部 HF processor 对象：部分 processor 直接暴露 apply_chat_template，
        # 部分通过 tokenizer 间接暴露，此处检测 processor 的能力边界
        if hasattr(self.processor, "apply_chat_template"):
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": add_generation_prompt,
                "return_dict": True,
                "return_tensors": "pt",
                **(chat_template_kwargs or {}),
            }
            tools_arg = _tools_for_chat_template(tool_batches)
            if tools_arg is not None:
                template_kwargs["tools"] = tools_arg
            try:
                encoded = self.processor.apply_chat_template(
                    messages,
                    processor_kwargs={"padding": True},
                    **template_kwargs,
                )
            except TypeError:
                encoded = self.processor.apply_chat_template(
                    messages,
                    padding=True,
                    **template_kwargs,
                )
        else:
            raise RuntimeError(
                "AutoProcessor does not implement apply_chat_template for multimodal samples"
            )
        return dict(encoded)

    def _multimodal_inputs_to_device(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        keys = (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
        )
        moved: dict[str, torch.Tensor] = {}
        for key in keys:
            value = encoded.get(key)
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
        return moved

    def _multimodal_inputs_from_metadata(
        self, metadata: Any | None, *, batch_size: int
    ) -> dict[str, torch.Tensor] | None:
        from graspo.flow.adapters.multimodal_tensors import _multimodal_rows_from_metadata
        from graspo.ripple.multimodal.rows import MULTIMODAL_ROWS_KEY

        rows = _multimodal_rows_from_metadata(metadata, expected_rows=batch_size)
        if not rows:
            # 防线：metadata 声明了 rows 键但解析为空 → 断链。
            # 静默返回 None 会丢失多模态（历史教训），必须硬失败。
            if isinstance(metadata, dict) and MULTIMODAL_ROWS_KEY in metadata:
                raise RuntimeError(
                    f"metadata contains {MULTIMODAL_ROWS_KEY!r} but resolved to empty rows; "
                    "multimodal inputs would be silently dropped"
                )
            return None
        encoded = self._encode_multimodal_rows(
            rows,
            add_generation_prompt=True,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        return self._multimodal_inputs_to_device(encoded)
