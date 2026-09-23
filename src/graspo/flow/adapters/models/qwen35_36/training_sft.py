"""Qwen3.5/3.6 adapter — SFT training methods (TP+DP+PP)."""

import logging
import time
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.adapters.models.common.grad_probe import (
    FAIL_CLOSED_PHASE,
    PP_EXCHANGE_BWD_RECEIVED,
    PP_EXCHANGE_BWD_SENT,
    PP_EXCHANGE_FWD_RECEIVED,
    PP_EXCHANGE_FWD_SENT,
    PP_NUMERIC_PROBE_PHASE,
    grad_fail_reason_text,
    grad_gate_verdict,
    pp_exchange_readings,
    reduced_grad_flags,
    step_index_one_based,
)
from graspo.flow.adapters.models.common.layers import _log_cuda_mem
from graspo.flow.adapters.models.qwen35_36.helpers import collate_sft_batch
from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.parallel.pipeline_comm import PipelineComm, wait_all
from graspo.flow.parallel.scheduling import build_scheduler
from graspo.flow.parallel.tensor_utils import (
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
)
from graspo.flow.progress_metrics import grad_count_event, training_norm_event
from graspo.ripple.data import SFTTokenized
from graspo.ripple.multimodal.contract import assert_sft_batch_has_multimodal

#: 旧的预授权开关名——**已删除**（§18.1 不留负债，v0.25.0 起）。
#:
#: 历史：非有限梯度"跳过并继续"曾是**环境变量** ``GRASPO_ALLOW_NONFINITE_GRAD_SKIP``
#: 通道。它违反 §7.1（环境变量不得承载配置）与 §10.1（决定产物的参数必须在 config
#: 里、且 config 备份必须能复现本 run），并且与 §3.3"部署前显式声明"不符——环境变量
#: 随启动命令消失，无从留痕。现已迁移为配置字段
#: ``native.allow_nonfinite_grad_skip``（默认 false，行为与基线逐位一致）。
#:
#: 本常量只为让"旧通道确实已断开"可被测试断言（防呆 §2.3：删掉的入口要能被验证
#: 删掉了）。**生产代码不得再读它。**
_REMOVED_NONFINITE_SKIP_ENV = "GRASPO_ALLOW_NONFINITE_GRAD_SKIP"


def _nonfinite_skip_preauthorized(allow_nonfinite_grad_skip: bool) -> bool:
    """是否被显式预授权"非有限梯度跳过并继续"（§3.3，默认关闭）。

    唯一真相源是配置字段 ``native.allow_nonfinite_grad_skip``（§1.4/§7.1），由调用点
    从 ``self.config`` 显式传入——本函数**不读进程环境**（防呆 §2.2：依赖显式）。
    """
    return bool(allow_nonfinite_grad_skip)


def _normalized_accumulation_weights(valid_token_counts: list[int]) -> list[float]:
    """把"各分块有效 token 数"归一为梯度累积权重（纯 math，§1.3 计算层可单测）。

    为什么必须是 token 数而不是样本数（F-13'，2026-09-19）：每个分块单独调
    :meth:`_Qwen35SFTTrainingMethods._compute_sft_loss`，而那**已经**是该分块内部的
    token 均值 ``m_i``。要让 C 个分块的加权和等于整批 token 均值
    ``Σ_i S_i / Σ_i L_i``（``S_i`` = 分块内 log-prob 之和，``L_i`` = 有效 token 数），
    唯一正确的累积权重是 **该分块有效 token 数占全批有效 token 数的比例**
    （``w_i = L_i / Σ_j L_j``）：

        ``Σ_i w_i · m_i = Σ_i (L_i / Σ_j L_j) · (S_i / L_i) = Σ_i S_i / Σ_j L_j``

    用**样本数**占比（``n_i / Σ_j n_j``）只有在"每个分块的有效 token 数都相等"时才
    恰好等价——micro-batch 里各样本 response 长度不同时并不成立，误差可达
    ``max(L_i) / min(L_i)``（长 chunk 被系统性高估）。

    权重和恒为 1.0（C 个分块各算一次），因此 C=1 时分块路径与整批路径**逐位等价**，
    也不再依赖调用方传入的 ``full_batch_size``。

    Args:
        valid_token_counts: 每个分块的有效 label token 数（非负；0 表示该分块无监督 token）。

    Returns:
        与输入等长的权重表；全批有效 token 数为 0 时返回**全 0**（由调用方按明确的
        边界行为处置——不得静默、不得除零）。

    Raises:
        ValueError: 任一计数为负（防呆：调用方统计口径写错时立即暴露）。
    """
    negatives = [count for count in valid_token_counts if count < 0]
    if negatives:
        raise ValueError(f"有效 token 数不得为负：{negatives}")
    total = sum(valid_token_counts)
    if total == 0:
        return [0.0 for _ in valid_token_counts]
    return [count / total for count in valid_token_counts]


def _count_sft_valid_tokens(labels: torch.Tensor) -> int:
    """micro-batch 的有效 label token 数（与 :meth:`_compute_sft_loss` 的分母同一口径）。

    :meth:`_compute_sft_loss` 内部做因果 shift（``labels[:, 1:]``）并只统计
    ``!= -100`` 的位置，这里必须用**同样的 shift 与同样的忽略值**统计，否则权重与
    分母不同源（§1.4 单一真相源）。
    """
    return int((labels[:, 1:] != -100).sum().item())



#: comm 侧诊断标签 → 读数方向（#1a）。**唯一真相源**：PipelineComm 的 `diag_sample` 用左列，
#: 本模块的钩子把它映射成语义方向（写进 rank_metrics 的 `pp_exchange`）。
_COMM_DIAG_LABEL_TO_DIRECTION = {
    "fwd_send": PP_EXCHANGE_FWD_SENT,
    "fwd_recv": PP_EXCHANGE_FWD_RECEIVED,
    "bwd_send": PP_EXCHANGE_BWD_SENT,
    "bwd_recv": PP_EXCHANGE_BWD_RECEIVED,
}

class _Qwen35SFTTrainingMethods:
    def _build_optimizer(self) -> None:
        """``_build_optimizer`` 扩展点：接线 native 优化器态 CPU offload（WP-X2）。

        **默认关闭**（``native.offload_optimizer_state=false``）⇒ 直接走基类实现，
        行为与基线逐位一致；只有显式开启时才把基类建好的 AdamW 换成 CPU-offload
        变体（宪法 §3.3 预授权退路）。基类的收参逻辑、``tuner_type=full`` 却收不到
        参数的 fail-closed、以及超参装配**一字未改**——这里只换"状态放在哪"。

        替换后必须**重建调度器**：基类是在建完优化器之后立刻建的 ``LambdaLR``，
        它持有的是那个已被丢弃的旧优化器；不重建的话 ``scheduler.step()`` 会去更新
        一个没人用的对象，学习率静默不变（§1.4 单一真相源）。
        """
        super()._build_optimizer()
        if not bool(self.config.native.offload_optimizer_state):
            return
        if self.optimizer is None:
            # 无可训参数时基类已经按 ``tuner_type`` 决定是报错还是返回 None；
            # 这里不重复判据，只保证不去 wrap 一个 None。
            return
        from graspo.flow.adapters.models.qwen35_36.optim_offload import (
            build_cpu_offloaded_adamw,
        )

        self.optimizer = build_cpu_offloaded_adamw(self.optimizer)
        self.scheduler = self._build_scheduler()

    def _compute_sft_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """从 hidden states 计算 SFT cross-entropy loss（设施层不持有算法分派）。

        复用 RL 的共享实现 ``masked_token_log_probs_from_hidden``（宪法 §1.4 单一真相源）：
        分块 logsumexp，不物化 (B,S,V) 全量 logits（v0.24.0 修复——此前 SFT 物化
        全词表 logits 导致显存不随 TP 分摊、TP=4 仍 OOM）。
        语义与 ``F.cross_entropy(ignore_index=-100, reduction="mean")`` 一致（**本分块内**的
        token 均值）。整批语义由调用方用**有效 token 数**做累积加权（见
        :func:`_normalized_accumulation_weights` 与 F-13'）——**不得**用样本数加权，
        否则多 micro-batch 且各 chunk 有效 token 数不等时与整批 token 均值不等价。
        无有效 token 时返回 0（由 ``mask.sum().clamp_min(1)`` 防除零）；调用方须显式
        处置该情形，不得让 0 权重块静默参与累积。
        """
        from graspo.ripple.loss import masked_token_log_probs_from_hidden

        norm = self.model.norm if hasattr(self.model, "norm") else None
        lm_head = self.model.lm_head if hasattr(self.model, "lm_head") else None
        if norm is None or lm_head is None:
            raise RuntimeError("SFT loss requires model.norm and model.lm_head")
        normalized = norm(hidden_states)
        # 因果语言模型 shift：位置 t 的 hidden 预测位置 t+1 的 token
        shift_hidden = normalized[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs = masked_token_log_probs_from_hidden(
            shift_hidden,  # 保持 bf16，函数内部分块处理，按需转换
            lm_head.weight,
            shift_labels,
            ignore_index=-100,
        )
        mask = shift_labels != -100
        return -(log_probs * mask).sum() / mask.sum().clamp_min(1)

    def _sync_grads_and_step(
        self,
        *,
        max_grad_norm: float,
        valid_micro_batches: int,
    ) -> tuple[float, int, int]:
        """统一的梯度同步 + clip + optimizer.step。

        防呆：DP/TP 梯度同步必须所有 rank 参与（集体操作），即使本 rank
        所有 micro-batch 都非有限（valid_micro_batches==0）也必须参加。
        若本 rank 无有效梯度，先 zero-fill 再参与 all_reduce。

        Returns:
            (grad_norm_sum, optimizer_steps, nonzero_grad_count)
        """
        from graspo.flow.lora.lora_linear import (
            _sync_dp_lora_grads,
            _sync_nonsharded_lora_grads,
        )
        from graspo.flow.parallel.tensor_utils import (
            _TENSOR_PARALLEL_GROUP,
            _TENSOR_PARALLEL_SIZE,
        )

        if valid_micro_batches == 0:
            for param in self.model.parameters():
                if param.requires_grad and param.grad is None:
                    param.grad = torch.zeros_like(param)

        # DP gradient sync: AVG across DP replicas（不同数据）
        if self.tp_state is not None and self.tp_state.dp_group is not None:
            _sync_dp_lora_grads(self.model, self.tp_state.dp_group)
        # TP gradient sync: SUM across TP ranks（同数据，部分梯度）
        if _TENSOR_PARALLEL_GROUP is not None and _TENSOR_PARALLEL_SIZE > 1:
            _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)

        if valid_micro_batches > 0:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                if trainable_params
                else torch.tensor(0.0)
            )
            grad_norm_sum = float(grad_norm.detach().float().cpu())
            self._sync_timing()
            _log_cuda_mem("before_optimizer_step")
            if self.optimizer is not None:
                self.optimizer.step()
            _log_cuda_mem("after_optimizer_step")
            self._sync_timing()
            if self.scheduler is not None:
                self.scheduler.step()
            optimizer_steps = 1
            nonzero_grad_count = self.model.training_progress_grad_count()
        else:
            grad_norm_sum = 0.0
            optimizer_steps = 0
            nonzero_grad_count = 0
        return grad_norm_sum, optimizer_steps, nonzero_grad_count

    def _build_sft_metrics(
        self,
        *,
        sft_batch_count: int,
        optimizer_steps: int,
        skipped_nonfinite: int,
        loss_sum: float,
        micro_batch_count: int,
        grad_norm_sum: float,
        nonzero_grad_count: int,
        norm_before: float,
        norm_after: float,
        train_batch_started_at: float,
        micro_batch_forward_sec: float,
        backward_sec: float,
        optimizer_step_sec: float,
        round_secs: list[float] | None = None,
        **pipeline_extras,
    ) -> dict[str, Any]:
        """构建统一的 SFT 训练指标字典。

        PP 路径通过 ``**pipeline_extras`` 注入 PP 特有字段（pp_size,
        pp_schedule, pipeline_stage_timing 等）。

        权重范数与梯度计数是**模式感知**的（``flow/progress_metrics.py``）：
        lora 模式键名/数值逐字不变；full 模式把结构性恒为 0 的 ``lora_norm_*``
        显式置 ``None``（不适用），并给出等价的 ``trainable_norm_*``。
        """
        tuner_type = self.config.effective_tuner_type
        metrics = {
            "optimized": optimizer_steps > 0,
            "sft_batch_count": sft_batch_count,
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / micro_batch_count if micro_batch_count else None,
            "grad_norm_mean": grad_norm_sum,
            **grad_count_event(tuner_type, count=nonzero_grad_count),
            **training_norm_event(tuner_type, before=norm_before, after=norm_after),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs or [],
            "optimize_round_sec_sum": sum(round_secs) if round_secs else 0.0,
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "current_lr": self._current_lr(),
            **pipeline_extras,
        }
        metrics = self._aggregate_rank_metrics(metrics)
        return metrics

    def _trainable_grad_norm(self) -> float:
        """本 rank 可训参数梯度的 L2 范数（诊断用；不含 clip，非有限值原样返回）。

        为什么需要它：非有限梯度被检测到时 ``_sync_grads_and_step`` 不会被调用，
        于是连"哪个 rank 的梯度是有限的"都没有读数（F-4 实测：stdout 只打了 rank0
        的有限局部值，全局 NaN 只存在于 rank_metrics 旁路）。这个探针只读、无集合
        通信，且**不做过滤**——NaN 必须原样透出，否则就重现了同一个观测缺陷。
        """
        total: torch.Tensor | None = None
        for param in self.model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            value = param.grad.detach().float().pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return 0.0
        return float(total.sqrt().cpu())

    def _record_nonfinite_skip(
        self,
        *,
        skipped: int,
        optimizer_steps: int,
        detail: str,
        rank_grad_norms: list[float] | None = None,
    ) -> None:
        """非有限梯度的事后处置：审计 WARNING +（非预授权时）硬失败。

        F-4（2026-09-18）实测的缺陷形态：SFT 路径既不告警也不中止，继续跑完全部
        步、写 ``final`` checkpoint、``exit_code=0``，于是"权重自第 2 步起完全冻结"
        的 run **看起来成功**。宪法 §3.4：边界校验拒绝非法数据是**防线**，不是退路
        ——"继续跑完并落盘"不是同效退路（它改变了结果且没有告知）。

        因此默认行为是 **raise**：进程非零退出、不写 final、判定层拿到显式标记。
        只有显式预授权（§3.3，配置字段 ``native.allow_nonfinite_grad_skip``，默认关闭）
        才允许"跳过并继续"，此时必须打出用户可读的声明，并把 ``skipped_nonfinite``
        一路带到 rank_metrics 与判定层（A2 会因此判不通过）。
        """
        logger = logging.getLogger("graspo.sft_trainer")
        self.nonfinite_grad_skips = int(getattr(self, "nonfinite_grad_skips", 0)) + skipped
        ranks = f"逐 rank grad_norm={rank_grad_norms} " if rank_grad_norms is not None else ""
        message = f"{ranks}{detail}；本步 optimizer_steps={optimizer_steps}。"
        if _nonfinite_skip_preauthorized(self.config.native.allow_nonfinite_grad_skip):
            logger.warning(
                "SFT 训练出现非有限梯度：%s【native.allow_nonfinite_grad_skip=true "
                "已显式预授权（§3.3）：跳过并继续；"
                "本 run 的后续步权重可能是冻结的，不得记为成功】",
                message,
            )
            return
        raise RuntimeError(
            "非有限梯度：SFT 训练硬失败（fail-closed，宪法 §3.4）。"
            f"SFT 训练出现非有限梯度：{message}"
            "权重已冻结，不得继续训练、不得落盘 final checkpoint；"
            "若确需跳过，必须显式预授权（§3.3，默认关闭）："
            "在 config 里设 native.allow_nonfinite_grad_skip=true。"
        )

    """Mixin: SFT training/batch optimization methods for Qwen35Adapter."""

    # ── TP-only SFT training ─────────────────────────────────────────────────

    def train_batch_sft(
        self,
        sft_batches: list[SFTTokenized],
        *,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """SFT 训练：对一批 ``SFTTokenized`` 样本执行 forward → cross-entropy loss → backward。

        支持梯度累积：当 ``micro_batch_size < len(sft_batches)`` 时，将外批拆分为多个
        micro-batch 逐个 forward/backward，梯度在 micro-batch 间累加，最后统一
        ``optimizer.step()``。有效 batch size = ``len(sft_batches)``。

        Args:
            sft_batches: ``sft_tokenize_text`` / ``sft_tokenize_multimodal`` 产出的
                ``SFTTokenized`` 列表。
            max_grad_norm: 梯度裁剪阈值
        """
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        if self._is_pipeline_parallel():
            return self._pipeline_train_batch_sft(
                sft_batches,
                max_grad_norm=max_grad_norm,
            )
        self.model.train()
        if bool(self.config.native.empty_cache_before_train) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        micro_batch_size = max(1, int(self.config.native.micro_batch_size))
        # gradient_accumulation_micro_batches 只决定拆几个 micro-batch；累积权重按
        # **有效 token 数**归一（F-13'），不按 micro-batch 个数。
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        norm_before = self.model.training_progress_norm()
        train_batch_started_at = time.monotonic()
        round_secs: list[float] = []
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        micro_batch_count = 0
        valid_micro_batches = 0  # 实际贡献梯度的 micro-batch 数
        # 先算**整批**的有效 token 分布，再据此归一累积权重（F-13'）。
        # 为什么单独先跑一遍：`collate_sft_batch` 可能截断（多模态的 max_seq_length），
        # 只有 collate 之后的 labels 才知道真实的有效 token 数；但**不能**把所有
        # micro-batch 同时留在内存里（旧路径是逐个 collate、逐个释放，多留一份
        # pixel_values/激活会实打实地抬高峰值显存）。因此第一遍只取计数、就地丢弃，
        # 第二遍重新 collate 做真正的 forward/backward（同一函数、同一参数 ⇒ 结果相同）。
        accumulation_weights = _normalized_accumulation_weights(
            [
                _count_sft_valid_tokens(
                    collate_sft_batch(
                        sft_batches[start : start + micro_batch_size],
                        self.device,
                        adapter=self,
                        max_seq_length=int(self.config.data.max_prompt_length),
                    )["labels"]
                )
                for start in range(0, len(sft_batches), micro_batch_size)
            ]
        )
        for start, accumulation_weight in zip(
            range(0, len(sft_batches), micro_batch_size), accumulation_weights, strict=True
        ):
            batch_items = sft_batches[start : start + micro_batch_size]
            micro_batch = collate_sft_batch(
                batch_items,
                self.device,
                adapter=self,
                max_seq_length=int(self.config.data.max_prompt_length),
            )
            self._sync_timing()
            forward_started_at = time.monotonic()
            multimodal_inputs = micro_batch.get("multimodal_inputs")
            # 防呆：样本含媒体但 batch 无 multimodal_inputs → 硬失败。
            # 与 RL 路径的 assert_rl_training_has_multimodal 对应，
            # 防止多模态样本静默走纯文本 forward（图像丢失）。
            assert_sft_batch_has_multimodal(
                any(item.deferred_multimodal is not None for item in batch_items),
                multimodal_inputs,
                context="SFT training forward",
            )
            if multimodal_inputs is not None:
                if not isinstance(self.model, Qwen35HybridTextModel):
                    raise ValueError("multimodal SFT batch for a non-multimodal model")
                hidden = self.model._forward_hidden(
                    micro_batch["input_ids"],
                    attention_mask=micro_batch["attention_mask"],
                    multimodal_inputs=multimodal_inputs,
                )
            else:
                hidden = self.model._forward_hidden(
                    micro_batch["input_ids"],
                    attention_mask=micro_batch["attention_mask"],
                )
            assert isinstance(hidden, torch.Tensor)
            self._sync_timing()
            micro_batch_forward_sec += time.monotonic() - forward_started_at
            loss = self._compute_sft_loss(hidden, micro_batch["labels"])
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue
            micro_batch_count += 1
            valid_micro_batches += 1
            loss_sum += float(loss.detach().cpu())
            # 梯度累积：按该 micro-batch 的有效 token 占比加权，使累加梯度与
            # "整批 token 均值"逐位等价（F-13'；此前是 / num_micro_batches，在各
            # micro-batch 有效 token 数不等时不等价）。weight=0（该块无监督 token）
            # ⇒ 梯度贡献为 0，backward 仍执行但无梯度，等价于该块不参与全局均值。
            scaled_loss = loss * accumulation_weight
            self._sync_timing()
            backward_started_at = time.monotonic()
            _log_cuda_mem("before_backward")
            scaled_loss.backward()
            _log_cuda_mem("after_backward")
            self._sync_timing()
            backward_sec += time.monotonic() - backward_started_at

        # 所有 micro-batch 的 backward 完成后，统一 sync / clip / step
        grad_norm_sum, optimizer_steps, nonzero_grad_count = self._sync_grads_and_step(
            max_grad_norm=max_grad_norm,
            valid_micro_batches=valid_micro_batches,
        )
        self._train_batch_call_index += 1
        if skipped_nonfinite > 0:
            self._record_nonfinite_skip(
                skipped=skipped_nonfinite,
                optimizer_steps=optimizer_steps,
                detail="train_batch_sft: 非有限 loss 的 micro-batch 已被跳过",
            )

        norm_after = self.model.training_progress_norm()
        metrics = self._build_sft_metrics(
            sft_batch_count=len(sft_batches),
            optimizer_steps=optimizer_steps,
            skipped_nonfinite=skipped_nonfinite,
            loss_sum=loss_sum,
            micro_batch_count=micro_batch_count,
            grad_norm_sum=grad_norm_sum,
            nonzero_grad_count=nonzero_grad_count,
            norm_before=norm_before,
            norm_after=norm_after,
            train_batch_started_at=train_batch_started_at,
            micro_batch_forward_sec=micro_batch_forward_sec,
            backward_sec=backward_sec,
            optimizer_step_sec=optimizer_step_sec,
            round_secs=round_secs,
        )
        self._emit_rank_memory_event("sft_train_batch_after", {"metrics": metrics})
        return metrics

    # ── PP SFT training (1F1B) ─────────────────────────────────────────────────

    def _pipeline_train_batch_sft(
        self,
        sft_batches: list[SFTTokenized],
        *,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """PP SFT 训练 — 1F1B 调度，梯度累积。

        将 ``sft_batches`` 按 ``micro_batch_size`` 拆分为 micro-batch，
        通过 1F1B fill/steady/drain 三阶段流水线执行 forward/backward。
        所有 micro-batch 的梯度累加后统一 ``optimizer.step()``。
        有效 batch size = ``len(sft_batches)``。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        self.model.train()
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        norm_before = self.model.training_progress_norm()
        micro_batch_size = max(1, int(self.config.native.micro_batch_size))
        train_batch_started_at = time.monotonic()
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        round_secs: list[float] = []
        micro_batch_count = 0
        stage_timing = _new_pipeline_stage_timing()
        fill_sec = 0.0
        steady_sec = 0.0
        drain_sec = 0.0

        # 预 collate 所有 micro-batch（1F1B 需要提前知道所有 chunk）
        chunk_batches: list[dict[str, Any]] = []
        for start in range(0, len(sft_batches), micro_batch_size):
            batch_items = sft_batches[start : start + micro_batch_size]
            chunk_batches.append(
                collate_sft_batch(
                    batch_items,
                    self.device,
                    adapter=self,
                    max_seq_length=int(self.config.data.max_prompt_length),
                )
            )

        chunk_count = len(chunk_batches)
        if chunk_count == 0:
            self._train_batch_call_index += 1
            return self._aggregate_rank_metrics({"optimized": False, "optimizer_steps": 0})

        # F-13'：每个 chunk 的 ``_compute_sft_loss`` 已是**该 chunk 内的 token 均值**，
        # 累积权重必须按**有效 token 数**归一（此前按样本数 mb["input_ids"].shape[0]
        # /full_batch_size，在 chunk_count>1 且各 chunk 有效 token 数不等时与整批 token
        # 均值不等价；micro_batch_size=1 时恰好退化为 1/chunk_count，仍不等价）。
        # 非末 stage 也各算一次，保证各 PP rank 的有效 token 总数一致（数据相同）；
        # 真正参与 loss/backward 的只有末 stage。
        valid_token_counts = [
            _count_sft_valid_tokens(micro_batch["labels"]) for micro_batch in chunk_batches
        ]
        accumulation_weights = _normalized_accumulation_weights(valid_token_counts)
        # 边界（防呆，§2.3）：整批无有效监督 token ⇒ 无 loss 可算。不得除零、不得静默
        # 产出 0 梯度却继续推进 optimizer（那会把"这一步没学到任何东西"伪装成成功）。
        if sum(valid_token_counts) == 0:
            raise RuntimeError(
                "SFT·PP 本批无任何有效 label token（labels[:, 1:] 全为 -100）："
                "整批 token 均值无定义，拒绝以 0 loss 静默推进 optimizer。"
                "请检查数据 tokenization / max_prompt_length 是否把 response 全部截掉。"
            )

        # 异步 P2P 通信管道（Flink 风格"调度与计算分离"）+ 可插拔调度策略
        comm = PipelineComm(
            device=self.device,
            fwd_group=self.tp_state.pp_group_fwd,
            bwd_group=self.tp_state.pp_group_bwd,
            max_inflight=int(self.config.native.pp_max_inflight_microbatches),
            chunk_count=chunk_count,
        )
        send_works: list[Any] = []
        records: list[dict[str, Any] | None] = [None for _ in range(chunk_count)]
        finite_flags = [True for _ in range(chunk_count)]
        loss_values = [0.0 for _ in range(chunk_count)]
        # ── 机制探针 #1（2026-09-23 裁定）：PP **交换张量**的只读读数 ────────────────
        # 为什么必须**在交换点就地取数**：`stage_input`/`grad_output` 在 backward 结束后即被
        # 释放（`records[chunk_idx] = None`），**事后取不到**。
        # 成本与纪律：每 chunk ≤4 个张量 × (isfinite + max|·|) 两次本地归约；**不改张量、
        # 不改数值、无任何集合通信**；读数**只进诊断行**（首步探针 / 失败步）⇒ 健康步
        # 不新增行、不新增键（见 tests/flow/trainer/test_grad_fail_closed.py 的钉子）。
        exchange_readings: list[dict[str, Any]] = []
        # #1b（2026-09-23 裁定）：**发送完成后**再读一次同一 buffer，用于分离
        # "我们的读数时刻不对（仪器问题甲）" 与 "通信真不保真（乙）"。
        # ⚠ 保守性（必须登记）：复读要**保留张量引用** ⇒ 会阻止 allocator 复用该内存，
        #   因此本测试对"释放后被复用"这一支是**保守**的（只可能少报、不会多报）。
        pending_recheck: list[tuple[str, str, Any]] = []
        post_wait_readings: list[dict[str, Any]] = []
        exchange_cap = 16
        # #1c：逐 bit 级证据（raw bytes sha256）开销大 ⇒ **只在首个 batch**那一次开
        probe_sha256 = int(self._train_batch_call_index) == 0

        # #1a（2026-09-23 裁定）：取数点移进 PipelineComm 的**真 send/recv 张量**处
        # （send 在 contiguous 之后、isend 之前；recv 在 wait 完成之后），并由 comm 在
        # 每个诊断步的**首个**取样点做**一次性** `torch.cuda.synchronize()`
        # （`diag_sync_once`）⇒ 读与写不再竞争。
        chunk_ref = {"idx": 0}

        def _on_comm_sample(comm_label: str, tensor: Any) -> None:
            """comm 诊断钩子：把真交换张量的标签映射成方向并落读数（None/未知 ⇒ 跳过）。"""
            direction = _COMM_DIAG_LABEL_TO_DIRECTION.get(str(comm_label))
            if direction is None or tensor is None or len(exchange_readings) >= exchange_cap:
                return
            name = f"chunk{chunk_ref['idx']}"
            exchange_readings.extend(
                pp_exchange_readings(
                    [(direction, name, tensor)],
                    max_items=exchange_cap - len(exchange_readings),
                    with_sha256=probe_sha256,
                )
            )
            if direction in (PP_EXCHANGE_FWD_SENT, PP_EXCHANGE_BWD_SENT):
                if len(pending_recheck) < exchange_cap:
                    pending_recheck.append((direction, name, tensor))

        comm.diag_hook = _on_comm_sample
        comm.diag_sync_once = True  # 每步首个取样点做一次设备同步（见 diag_begin_step）
        comm.diag_begin_step()  # 本步重新武装"一次性同步"（每步至多一次）

        # 梯度累积：zero_grad 只调一次，所有 chunk 的梯度累加后统一 step
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        def forward_chunk(chunk_idx: int) -> None:
            nonlocal micro_batch_forward_sec
            mb = chunk_batches[chunk_idx]
            self._sync_timing()
            t0 = time.monotonic()
            stage_output, stage_input, send_work = self._pipeline_forward_for_sft(
                mb["input_ids"],
                mb["attention_mask"],
                multimodal_inputs=mb.get("multimodal_inputs"),
                timing=stage_timing,
                comm=comm,
                tag=chunk_idx,
            )
            self._sync_timing()
            micro_batch_forward_sec += time.monotonic() - t0
            if send_work is not None:
                send_works.append(send_work)
            loss: torch.Tensor | None = None
            finite = True
            loss_value = 0.0
            if self.pp_rank == self.pp_size - 1:
                assert stage_output is not None
                chunk_loss = self._compute_sft_loss(stage_output, mb["labels"])
                finite = bool(torch.isfinite(chunk_loss).detach().cpu())
                # 梯度累积（F-13'）：该 chunk 的有效 token 数占全批有效 token 数的比例，
                # 使 Σ_i w_i ·（第 i chunk 的 token 均值）= 整批 token 均值。权重与
                # loss_value 用同一系数，保证报告口径与反传口径一致。
                weight = accumulation_weights[chunk_idx]
                loss = chunk_loss * weight if finite else None
                loss_value = float(chunk_loss.detach().cpu()) * weight if finite else 0.0
            chunk_ref["idx"] = chunk_idx  # comm 钩子按当前 chunk 标名（#1a）
            records[chunk_idx] = {
                "stage_output": stage_output,
                "stage_input": stage_input,
                "loss": loss,
            }
            finite_flags[chunk_idx] = finite
            loss_values[chunk_idx] = loss_value

        def backward_chunk(chunk_idx: int) -> None:
            nonlocal backward_sec
            record = records[chunk_idx]
            if record is None:
                raise RuntimeError("1F1B attempted backward before forward")
            self._sync_timing()
            t0 = time.monotonic()
            if self.pp_rank == self.pp_size - 1:
                stage_input = record["stage_input"]
                loss = record["loss"]
                if loss is not None:
                    _log_cuda_mem("before_backward_pp")
                    loss.backward()
                    _log_cuda_mem("after_backward_pp")
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
            else:
                stage_output = record["stage_output"]
                assert stage_output is not None
                grad_output = torch.empty_like(stage_output)
                recv_work = comm.bwd_recv(
                    grad_output,
                    src=int(self.tp_state.next_pp_rank or 0),
                    tag=chunk_count + chunk_idx,
                )
                comm.wait(recv_work)  # 阻塞直到梯度到达（上游异步 send，不会死锁）
                chunk_ref["idx"] = chunk_idx  # （recv 侧取样由 comm 钩子在 wait 完成后做）
                stage_output.backward(grad_output)
                stage_input = record["stage_input"]
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
            self._sync_timing()
            backward_sec += time.monotonic() - t0
            records[chunk_idx] = None

        # 可插拔调度策略：默认 1F1B（fill → steady → drain）
        scheduler = build_scheduler(
            self.config.native.pp_scheduler,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            num_chunks=chunk_count,
            forward=forward_chunk,
            backward=backward_chunk,
        )
        sched_stats = scheduler.run()
        fill_sec = float(sched_stats.get("pipeline_fill_sec", 0.0))
        steady_sec = float(sched_stats.get("pipeline_steady_sec", 0.0))
        drain_sec = float(sched_stats.get("pipeline_drain_sec", 0.0))
        # 同步所有异步发送（数据已被下游消费），保证 buffer 生命周期安全
        wait_all(send_works)
        # ── #1b：发送完成后复读同一 buffer（仅发送侧张量；见 pending_recheck 的保守性说明）──
        for direction, name, tensor in pending_recheck:
            post_wait_readings.extend(
                pp_exchange_readings(
                    [(direction, name, tensor)],
                    max_items=exchange_cap,
                    with_sha256=probe_sha256,
                )
            )

        all_finite = all(finite_flags)
        finite_tensor = torch.tensor([all_finite], dtype=torch.int, device=self.device)
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        all_finite = bool(finite_tensor.item())

        # ── R1（2026-09-23）：梯度有限性 / 填充度判据 ─────────────────────────────
        # 为什么必须在这里：上面的 ``all_finite`` 只反映**末 stage 的 loss 是否有限**
        # （``finite`` 仅在 ``pp_rank == pp_size-1`` 上赋值，其余 rank 恒 True），
        # 而 T018 逐 rank 实测 rank2 在 step3 已有 ``grad_norm_mean=nan`` 却
        # ``optimizer_steps=1``/``skipped_nonfinite=0`` ⇒ NaN 梯度被照常写进权重。
        # 判据是纯函数（``models/common/grad_probe.py::grad_gate_verdict``），可零 GPU 单测。
        local_grad = self.model.gradient_finiteness_report()
        grad_nonfinite_any, grad_unpopulated_any = reduced_grad_flags(
            local_report=local_grad, distributed=dist, device=self.device
        )
        grad_failed, grad_fail_reason = grad_gate_verdict(
            loss_all_finite=all_finite,
            grad_nonfinite_any=grad_nonfinite_any,
            grad_unpopulated_any=grad_unpopulated_any,
        )
        step_ok = not grad_failed
        # 0-based 批序号（自增前的值）；落盘一律经 ``step_index_one_based``
        # 转 1-based —— P16 统一口径（2026-09-23 裁定）
        batch_index_0based = int(self._train_batch_call_index)
        is_first_train_batch = batch_index_0based == 0
        # 诊断步（首步 / 本步要拦）才做第二遍 `max|·|` 归约：让落盘行带上
        # ``grad_max_abs`` 与 argmax 张量名（"范数是否被单个元素支配"的区分钥匙）。
        # 条件由 WORLD 归约后的标志决定 ⇒ 各 rank 一致，不会造成集合错配。
        want_detail = (not step_ok) or is_first_train_batch
        local_grad_detail = (
            self.model.gradient_finiteness_report(with_max_abs=True) if want_detail else local_grad
        )

        # 逐 rank 梯度范数/探针：所有 rank 都算（含 nan），再做集合汇总，得到"全局"旁证。
        # 只在需要审计时才计算（本步要拦 / 预授权模式），避免正常路径开销。
        # ★ 2026-09-23 裁定 5(B)：**首步不再并入本条件**。理由：本装置的目的是"把错误暴露出来"，
        #   不该反过来给训练路径加集合通信——**被测对象不得被观测动作扰动**；而 NaN 缺陷的候选
        #   机制之一正是 PP 边界的流/时序纪律（`record_stream` 零命中），在首步插一次集合通信
        #   可能正好扰动要观测的对象。代价：健康首步的 `pp_numeric_probe` 行 `rank_grad_norms=[]`
        #   （其余字段不变——`grad_max_abs`/填充计数等由本地探针给出，不经集合通信）。
        # 注意：条件 ``not step_ok`` 由 WORLD 归约后的标志决定 ⇒ 各 rank **一致**，
        # 因此这里的集合通信不会错配（§2.1 契约）。
        rank_grad_norms: list[float] = []
        rank_grad_reports: list[dict[str, Any]] = []
        if not step_ok or _nonfinite_skip_preauthorized(
            self.config.native.allow_nonfinite_grad_skip
        ):
            local_grad_norm = self._trainable_grad_norm()
            gathered_norms: list[float | None] = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_norms, local_grad_norm)
            rank_grad_norms = [
                float(value) if value is not None else float("nan") for value in gathered_norms
            ]
            self.rank_grad_norms_last = rank_grad_norms
            # 逐 rank 探针表（诊断行用）：表很小（计数 + 前 N 个张量名 + max|g|）。
            gathered_reports: list[dict[str, Any] | None] = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_reports, local_grad_detail)
            rank_grad_reports = [item for item in gathered_reports if isinstance(item, dict)]

        if step_ok:
            grad_norm_sum, optimizer_steps, nonzero_grad_count = self._sync_grads_and_step(
                max_grad_norm=max_grad_norm,
                valid_micro_batches=chunk_count,
            )
            loss_sum = sum(loss_values)
            micro_batch_count = chunk_count
        else:
            skipped_nonfinite = chunk_count

        self._train_batch_call_index += 1
        if is_first_train_batch:
            # R3（2026-09-23）：首步逐 rank 数值探针——**只增行、不新增集合通信、
            # 不参与判据**。明细（``grad_max_abs`` / argmax 张量名）由本 rank 另算一次
            # （只发生在首步/诊断步），用于回答"范数是否被单个元素支配"。
            self._emit_rank_memory_event(
                PP_NUMERIC_PROBE_PHASE,
                {
                    PP_NUMERIC_PROBE_PHASE: {
                        "step": step_index_one_based(batch_index_0based),
                        "pp_rank": self.pp_rank,
                        "pp_size": self.pp_size,
                        "loss_all_finite": all_finite,
                        "finite_flags": [bool(flag) for flag in finite_flags],
                        "loss_values": [float(value) for value in loss_values],
                        "optimizer_stepped": bool(step_ok),
                        "fail_reason": grad_fail_reason,
                        "rank_grad_norms": [float(value) for value in rank_grad_norms],
                        "pp_exchange": exchange_readings,
                        "pp_exchange_post_wait": post_wait_readings,
                        **local_grad_detail,
                    }
                },
            )
        if skipped_nonfinite > 0:
            # ★ F-4 P0 修法①（宪法 §3.4：这是防线，不是退路）：首个非有限梯度
            #   **不得**被静默跳过。此前路径继续跑完全部步、写 final checkpoint、
            #   exit=0，于是"权重从第 2 步起完全冻结"的 run 看起来是成功的。
            #   ⇒ 这里打审计 WARNING（列出逐 rank 读数），并在**没有**显式预授权
            #   （§3.3，默认关闭）时直接硬失败——不写 final、exit≠0。
            # ★ R1（2026-09-23）：**先落盘诊断行、再 raise**。此前 raise 早于
            #   ``_build_sft_metrics`` ⇒ 触发失败的那一步**没有任何 rank_metrics 行**
            #   （T018 实测：4 步有读数、崩溃的第 5 步零行 ⇒ 取证只能靠推理）。
            fail_closed_metrics = {
                "step": step_index_one_based(batch_index_0based),
                "phase_kind": FAIL_CLOSED_PHASE,
                "fail_reason": grad_fail_reason,
                "fail_reason_text": grad_fail_reason_text(grad_fail_reason),
                "optimizer_steps": 0,  # 本步未 step（步进被本判据拦下）
                "optimizer_stepped": False,
                "skipped_nonfinite": int(skipped_nonfinite),
                "loss_all_finite": bool(all_finite),
                "finite_flags": [bool(flag) for flag in finite_flags],
                "loss_values": [float(value) for value in loss_values],
                "pp_rank": self.pp_rank,
                "pp_size": self.pp_size,
                "grad_nonfinite_any": bool(grad_nonfinite_any),
                "grad_unpopulated_any": bool(grad_unpopulated_any),
                "rank_grad_norms": [float(value) for value in rank_grad_norms],
                "rank_grad_reports": rank_grad_reports,
                "pp_exchange": exchange_readings,
                "pp_exchange_post_wait": post_wait_readings,
                **local_grad_detail,
            }
            self._emit_rank_memory_event(FAIL_CLOSED_PHASE, {"metrics": fail_closed_metrics})
            nonfinite_names = (
                ", ".join(
                    f"{item['name']}(max|g|={item['max_abs']:.6g})"
                    for item in local_grad_detail["first_nonfinite_grad_names"]
                )
                or "无（本 rank 梯度全有限）"
            )
            self._record_nonfinite_skip(
                skipped=skipped_nonfinite,
                optimizer_steps=optimizer_steps,
                detail=(
                    f"pipeline SFT: 本步判据={grad_fail_reason}"
                    f"（{grad_fail_reason_text(grad_fail_reason)}）⇒ 已跳过 optimizer.step()"
                    f"（逐 rank grad_norm={[float(value) for value in rank_grad_norms]}，"
                    f"逐 chunk finite={finite_flags}，"
                    f"本 rank grad_populated_count={local_grad_detail['grad_populated_count']}"
                    f"/{local_grad_detail['trainable_tensor_count']}，"
                    f"非有限梯度张量={nonfinite_names}）"
                ),
                rank_grad_norms=rank_grad_norms,
            )

        norm_after = self.model.training_progress_norm()
        metrics = self._build_sft_metrics(
            sft_batch_count=len(sft_batches),
            optimizer_steps=optimizer_steps,
            skipped_nonfinite=skipped_nonfinite,
            loss_sum=loss_sum,
            micro_batch_count=micro_batch_count,
            grad_norm_sum=grad_norm_sum,
            nonzero_grad_count=nonzero_grad_count,
            norm_before=norm_before,
            norm_after=norm_after,
            train_batch_started_at=train_batch_started_at,
            micro_batch_forward_sec=micro_batch_forward_sec,
            backward_sec=backward_sec,
            optimizer_step_sec=optimizer_step_sec,
            round_secs=round_secs,
            pp_size=self.pp_size,
            pp_schedule=sched_stats.get("pp_schedule", "one_f_one_b"),
            pipeline_stage_timing=_round_pipeline_stage_timing(stage_timing),
            pipeline_fill_sec=fill_sec,
            pipeline_steady_sec=steady_sec,
            pipeline_drain_sec=drain_sec,
            pipeline_forward_sec=float(sched_stats.get("pipeline_forward_sec", 0.0)),
            pipeline_backward_sec=float(sched_stats.get("pipeline_backward_sec", 0.0)),
        )
        self._emit_rank_memory_event("pipeline_sft_train_batch_after", {"metrics": metrics})
        return metrics

    def _pipeline_forward_for_sft(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        timing: dict[str, float | int] | None = None,
        comm: PipelineComm | None = None,
        tag: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Any | None]:
        """PP forward pass for SFT — delegates to the unified PP forward.

        SP / async-P2P / position_ids / tag 语义由 :meth:`_pipeline_forward_hidden`
        （pipeline_forward.py）集中维护，避免多套 PP forward 漂移。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        output, _present, stage_input, send_work = self._pipeline_forward_hidden(
            input_ids=input_ids,
            hidden_states=None,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            multimodal_inputs=multimodal_inputs,
            position_input_ids=input_ids,
            apply_lm_head=False,
            timing=timing,
            comm=comm,
            tag=tag,
            debug_label="fwd",
        )
        return output, stage_input, send_work


# ── SFT batch collation ──────────────────────────────────────────────────────
# collate_sft_batch / collate_sft_text_batch / collate_sft_multimodal_batch /
# count_media_types_from_messages 已迁入本目录 helpers.py（纯函数，受 mypy 检查）。
