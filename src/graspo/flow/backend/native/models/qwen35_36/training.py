"""Qwen3.5/3.6 adapter — RL training methods (TP-only, PP 1F1B)."""

import os
import time
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.backend.native.models.common.grad_probe import (
    FAIL_CLOSED_PHASE,
    PP_NUMERIC_PROBE_PHASE,
    grad_fail_reason_text,
    grad_gate_verdict,
    reduced_grad_flags,
    step_index_one_based,
)
from graspo.flow.backend.native.models.common.layers import _log_cuda_mem
from graspo.flow.backend.native.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.backend.native.parallel.pipeline_comm import PipelineComm, wait_all
from graspo.flow.backend.native.parallel.scheduling import build_scheduler
from graspo.flow.backend.native.parallel.tensor_utils import (
    _add_pipeline_stage_timing,
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
    collate_experiences,
)
from graspo.flow.progress_metrics import grad_count_event, training_norm_event
from graspo.ripple.buffer import Experience
from graspo.ripple.multimodal.contract import assert_rl_training_has_multimodal


# ── K-会合（step 内分段梯度同步；K 默认 4）──────────────────────────────────
# ★ 语义（证明见 `task-d3-escalation-design/viability.md` §1.2）：
#   把「每步一次 DP 梯度同步（R-B）」拆成 K 段，段末对**同一个累积缓冲**
#   （``param.grad`` 本身）做一次 ``all_reduce(AVG)``。AVG 是线性算子，且段间
#   **不清零** ⇒ K 次归约与 1 次归约在实数算术下逐元素相等：
#       Σ_k AVG(L_k) == AVG(Σ_k L_k)
#   五个必要条件（违反任一条即落入 viability.md §1.3 的反例 A–E，且判据看不见）：
#     ① 段间不清零/不重置  —— 本实现只在循环**前** ``zero_grad`` 一次；
#     ② 不再除以 K        —— ``loss / num_micro_batches`` 的分母恒为**全步**条数；
#     ③ 缩放恒为 1/n_total —— 同上，不按段改分母；
#     ④ 裁剪只在末段之后一次 —— ``clip_grad_norm_`` 仍在 R-B 之后、只执行一次；
#     ⑤ 归约算子必须是 AVG —— ``lora_linear.py::_sync_dp_lora_grads``（未改动）。
#   K=1 ⇒ 切点为空 ⇒ 与未分段版本**行为等价**（零新增集合、数值一致）。
#: 段数的配置来源（环境变量）：>=1 的整数；未设置时用默认值。
DP_SYNC_SEGMENTS_ENV = "GRASPO_DP_SYNC_SEGMENTS"
#: 默认段数 K = 4（用户 2026-10-09 批准实施）。
DEFAULT_DP_SYNC_SEGMENTS = 4


def _resolve_dp_sync_segments(env: "Mapping[str, str] | None" = None) -> int:
    """解析**生效的段数 K**：配置优先，未配置回落默认值；非法取值 fail-loud。

    ⛔ 不静默回落：一个手误的配置会让"每步同步几次"不可预期（宪法 §2.4 操作防呆）。
    """
    source = os.environ if env is None else env
    raw = source.get(DP_SYNC_SEGMENTS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_DP_SYNC_SEGMENTS
    text = raw.strip()
    if not text.isdigit() or int(text) < 1:
        raise ValueError(
            f"{DP_SYNC_SEGMENTS_ENV}={raw!r} 非法：必须是 >= 1 的整数"
            "（1 = 不分段，与未分段版本行为等价）"
        )
    return int(text)


def _dp_sync_segment_cuts(n_total: int, segments: int) -> tuple[int, ...]:
    """返回本步需要在哪些 micro-batch **迭代序号之前**做段末同步（K−1 个切点）。

    切法（与设计件一致）：按**条数**均分，余数给**前段**——
    ``base, rem = divmod(n_total, k_eff)``，前 ``rem`` 段各多 1 条。
    逐 rank 同切法的前提是 ``n_total`` 逐 rank 相等（route 3′ 的 MIN 对齐已保证）。
    ``k_eff = min(segments, n_total)``：条数少于段数时退化为"每条一次"，数学上仍正确，
    且切点随 ``n_total`` 确定性生成 ⇒ 逐 rank 同进同出。
    """
    if segments < 1:
        raise ValueError(f"segments 必须 >= 1，收到 {segments}")
    if n_total <= 0:
        return ()
    k_eff = min(int(segments), int(n_total))
    base, rem = divmod(int(n_total), k_eff)
    cuts: list[int] = []
    acc = 0
    for idx in range(k_eff - 1):
        acc += base + (1 if idx < rem else 0)
        if 0 < acc < n_total:
            cuts.append(acc)
    return tuple(cuts)


def _dp_k_meeting_sync_dp_grads(
    model: Any,
    tp_state: Any,
    *,
    iter_index: int,
    cuts: tuple[int, ...],
    n_total: int,
) -> bool:
    """K-会合：若 ``iter_index`` 命中段边界，做一次 **in-place** DP 梯度同步。

    ⛔ **无条件**执行（不看 ``torch.isfinite(loss)``、不看 ``valid_micro_batches``）：
       段边界只由**迭代序号**决定 ⇒ 逐 rank 同进同出，不产生"条件集合"失配。
       因此本调用必须放在循环体的**最开头**（``continue`` 跳过不了它）。
    ⛔ 段间**不清零**：本函数只做 ``all_reduce(AVG)``，累积在既有 ``param.grad`` 上。
    ⛔ 守卫与 R-B 同域，并额外要求 ``dp_size > 1``：``dp_size == 1``（PP/单卡档，
       含 ``dp_group`` 为**1 成员组**的 T035/T036 形态）⇒ **0 次新增集合**。

    :returns: 是否真的做了一次同步（供自证计数）。
    """
    if iter_index not in cuts:
        return False
    if tp_state is None or getattr(tp_state, "dp_group", None) is None:
        return False
    if int(getattr(tp_state, "dp_size", 1)) <= 1:
        return False
    from graspo.flow.backend.native.lora.lora_linear import _sync_dp_lora_grads  # noqa: PLC0415

    # ★ 只读探针（诊断跑）：段末同步是否发生、在第几条 micro-batch 之后。
    #   ⛔ 取 rank 必须用 `dist.get_rank()`（硬规矩 15），不得用分维度 rank 字段冒充（E-97）。
    print(
        f"[DPSYNC] site=dp_seg_sync rank={dist.get_rank() if dist.is_initialized() else 0} "
        f"t={time.monotonic():.3f} seg_after={iter_index} n_micro={n_total}",
        flush=True,
    )
    _sync_dp_lora_grads(model, tp_state.dp_group)
    return True


class _Qwen35TrainingMethods:
    """Mixin: RL training/batch optimization methods for Qwen35ModelAdapter."""

    # ── TP-only RL training ──────────────────────────────────────────────────

    def train_batch(
        self,
        *,
        experiences: list[Experience],
        optimizer_steps: int = 1,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """RL 训练：对一批 ``Experience`` 执行 forward → GRPO loss → backward。

        支持梯度累积：每个 micro_batch 处理 ``micro_batch_size`` 个 experience，
        所有 micro-batch 的梯度累加后统一 ``optimizer.step()``。
        有效 batch size = ``micro_batch_size × gradient_accumulation_micro_batches``。

        ★ 口径注（E-80/E-84，2026-10-05）：上述"累积后统一 ``optimizer.step()``"的描述
        **修复前仅对非 PP 路径成立**；PP 路径原先每处理 1 条 experience 即 clip+step 一次
        （T035 实测 ``optimizer_steps_per_rank`` = 31/53/55）。**E-80 起 PP 路径亦按
        ``gradient_accumulation_micro_batches`` 分组累积**（``G=1`` 时为恒等变换，见 E-80 补注）；
        **E-84 另将 PP 的梯度裁剪改为跨 PP rank 归约的全局范数**。详见 ``CORRECTIONS.md``
        E-77/E-78/E-80/E-84。
        """
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        if self._is_pipeline_parallel():
            return self._pipeline_train_batch(
                experiences,
                policy_ratio_clip_eps=policy_ratio_clip_eps,
                max_grad_norm=max_grad_norm,
            )
        self.loss_fn.policy_ratio_clip_eps = policy_ratio_clip_eps
        self.model.train()
        if bool(self.config.native.empty_cache_before_train) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        norm_before = self.model.training_progress_norm()
        batch_size = int(self.config.native.micro_batch_size)
        train_batch_started_at = time.monotonic()
        round_secs: list[float] = []
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        micro_batch_count = 0
        # Single pass — no repeated iterations (avoids stale old_log_probs).
        optimize_round = 0
        round_started_at = time.monotonic()
        indices = self._shared_training_indices(len(experiences), optimize_round=optimize_round)
        # 梯度累积：zero_grad 只调一次，所有 micro-batch 的梯度累加后统一 step。
        num_micro_batches = max(1, len(indices) // batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        valid_micro_batches = 0
        # ★ K-会合：把本步的 num_micro_batches 条 micro-batch 切成 K 段（余数给前段），
        #   除**末段**外的每个段末在**下一条的循环开头**做一次 in-place DP 同步；
        #   末段由循环之后的 R-B 承担 ⇒ 每步恰好 K 次 DP 集合（K=4 ⇒ 4×220）。
        #   切点只由本 rank 的条数算出；route 3′ 保证逐 rank 条数相等 ⇒ 同进同出。
        _dp_k_cuts = _dp_sync_segment_cuts(num_micro_batches, _resolve_dp_sync_segments())
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            # ★ 段末同步（K−1 个切点）——**无条件**：不看 isfinite(loss)、不看
            #   valid_micro_batches。段边界只由迭代序号决定；放在循环体最开头，
            #   故上面 `continue` 也跳过不了它 ⇒ 逐 rank 同进同出、无集合失配。
            _dp_k_meeting_sync_dp_grads(
                self.model,
                self.tp_state,
                iter_index=start // batch_size,
                cuts=_dp_k_cuts,
                n_total=num_micro_batches,
            )
            batch_indices = indices[start : start + batch_size]
            batch = collate_experiences([experiences[idx] for idx in batch_indices], self.device)
            self._sync_timing()
            forward_started_at = time.monotonic()
            multimodal_inputs = self._multimodal_inputs_from_metadata(
                batch.metadata,
                batch_size=int(batch.sequences.shape[0]),
            )
            # 防呆：sequences 含图像 token 但 metadata 无 rows → 硬失败。
            # 修复前的断链正是静默走到 else 分支（图像占位符按纯文本嵌入），
            # 训练 19.5 小时视觉 LoRA 从未收到梯度（历史教训）。
            assert_rl_training_has_multimodal(
                batch.metadata,
                batch.sequences,
                image_token_id=getattr(getattr(self.model, "config", None), "image_token_id", None),
                expected_rows=int(batch.sequences.shape[0]),
            )
            if multimodal_inputs is not None:
                if not isinstance(self.model, Qwen35HybridTextModel):
                    raise ValueError("multimodal batch metadata for a non-multimodal model")
                log_probs = self.model.sequence_log_probs(
                    batch.sequences,
                    batch.attention_mask,
                    multimodal_inputs=multimodal_inputs,
                )
            else:
                log_probs = self.model.sequence_log_probs(batch.sequences, batch.attention_mask)
            self._sync_timing()
            micro_batch_forward_sec += time.monotonic() - forward_started_at
            loss = self.loss_fn(
                log_probs,
                batch.old_log_probs,
                batch.advantages,
                batch.action_mask,
            )
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue
            micro_batch_count += 1
            valid_micro_batches += 1
            loss_sum += float(loss.detach().cpu())
            # 梯度累积：loss 除以 num_micro_batches 使累加梯度等价于大 batch
            scaled_loss = loss / num_micro_batches
            self._sync_timing()
            backward_started_at = time.monotonic()
            _log_cuda_mem("before_backward")
            scaled_loss.backward()
            _log_cuda_mem("after_backward")
            self._sync_timing()
            backward_sec += time.monotonic() - backward_started_at
            # ★ 只读探针 P4（诊断跑）：逐 micro-batch 墙钟 + 序列长度。
            #   ⚠ 本行只在 `torch.isfinite(loss)` 为真时执行（`:132-134` 的 `continue` 跳过它）
            #     ⇒ **行数 = `valid_micro_batches` ≤ `n_micro`**；做"单条耗时"分母时必须用它自己的行数
            #     （`finite_idx=` 即自证），⛔ 不得默认等于 `usable`。
            print(
                f"[DPSYNC] site=micro rank={dist.get_rank() if dist.is_initialized() else 0} t={time.monotonic():.3f} "
                f"idx={start} finite_idx={valid_micro_batches} "
                f"seqlen={int(batch.sequences.shape[1])}",
                flush=True,
            )

        # 所有 micro-batch 的 backward 完成后，统一 sync / clip / step
        # 防呆：DP/TP 梯度同步必须所有 rank 参与（集体操作），即使本 rank
        # 所有 micro-batch 都非有限（valid_micro_batches==0）也必须参加。
        # 若本 rank 无有效梯度，先 zero-fill 再参与 all_reduce。
        if valid_micro_batches == 0:
            for param in self.model.parameters():
                if param.requires_grad and param.grad is None:
                    param.grad = torch.zeros_like(param)
        from graspo.flow.backend.native.lora.lora_linear import (
            _sync_dp_lora_grads,
            _sync_nonsharded_lora_grads,
        )
        from graspo.flow.backend.native.parallel.tensor_utils import (
            _TENSOR_PARALLEL_GROUP,
            _TENSOR_PARALLEL_SIZE,
        )

        # DP gradient sync: AVG across DP replicas（不同数据）
        # ★ K-会合：本处即**末段**同步（段内的 K−1 次已在上面的循环里做过）
        #   ⇒ 每步 DP 集合总数 = K × 220；K=1 时切点为空，本处即唯一一次（行为等价）。
        if self.tp_state is not None and self.tp_state.dp_group is not None:
            # ★ 只读探针 P2（诊断跑）：R-B 的到达时刻 + 本 rank 的 micro-batch 条数。
            print(
                f"[DPSYNC] site=r_b_enter rank={dist.get_rank() if dist.is_initialized() else 0} t={time.monotonic():.3f} "
                f"n_micro={num_micro_batches} len_indices={len(indices)} "
                f"valid_micro={valid_micro_batches}",
                flush=True,
            )
            _sync_dp_lora_grads(self.model, self.tp_state.dp_group)
            # ★ 只读探针 P3（诊断跑）：R-B 退出时刻（与 P2 配对算 R-B 实际耗时）。
            print(
                f"[DPSYNC] site=r_b_exit rank={dist.get_rank() if dist.is_initialized() else 0} t={time.monotonic():.3f}",
                flush=True,
            )
        # TP gradient sync: SUM across TP ranks（同数据，部分梯度）
        if _TENSOR_PARALLEL_GROUP is not None and _TENSOR_PARALLEL_SIZE > 1:
            _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)
        if valid_micro_batches > 0:
            # ★ K-会合条件④：裁剪**只在末段同步之后做一次**（⛔ 绝不每段裁一次——
            #   clip 是非线性的，逐段 clip ≠ 整体 clip，且判据③ 看不见）。
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in self.model.parameters() if param.requires_grad],
                max_grad_norm,
            )
            grad_norm_sum = float(grad_norm.detach().float().cpu())
            self._sync_timing()
            optimizer_started_at = time.monotonic()
            _log_cuda_mem("before_optimizer_step")
            self.optimizer.step()
            _log_cuda_mem("after_optimizer_step")
            self._sync_timing()
            optimizer_step_sec = time.monotonic() - optimizer_started_at
            if self.scheduler is not None:
                self.scheduler.step()
            optimizer_steps = 1
            nonzero_grad_count = self.model.training_progress_grad_count()
        else:
            grad_norm_sum = 0.0
        round_secs.append(time.monotonic() - round_started_at)
        self._train_batch_call_index += 1

        norm_after = self.model.training_progress_norm()
        metrics = {
            "optimized": optimizer_steps > 0,
            "replay_buffer_trainable_completion_count": len(experiences),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / micro_batch_count if micro_batch_count else None,
            "grad_norm_mean": grad_norm_sum,
            **grad_count_event(self.config.effective_tuner_type, count=nonzero_grad_count),
            **training_norm_event(
                self.config.effective_tuner_type, before=norm_before, after=norm_after
            ),
            "activation_checkpointing_enabled": bool(
                getattr(self.model, "gradient_checkpointing", False)
            ),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "synchronize_cuda_timing": bool(self.config.native.synchronize_cuda_timing),
            "current_lr": self._current_lr(),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("train_batch_after", {"metrics": metrics})
        return metrics

    # ── PP RL training dispatch ──────────────────────────────────────────────

    def _pipeline_train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """PP training — 1F1B 调度，梯度累积。

        Flow 层统一架构：TP + PP(1F1B)。PP=1 时走 TP-only 路径（``train_batch``），
        PP>1 时走 1F1B 流水线（本方法）。不存在其他 PP 调度策略。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        return self._pipeline_train_batch_one_f_one_b(
            experiences,
            policy_ratio_clip_eps=policy_ratio_clip_eps,
            max_grad_norm=max_grad_norm,
        )

    # ── PP forward (shared) ──────────────────────────────────────────────────

    def _pipeline_forward_for_training(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        metadata: Any | None = None,
        timing: dict[str, float | int] | None = None,
        comm: PipelineComm | None = None,
        tag: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Any | None]:
        """PP forward for RL training — delegates to the unified PP forward.

        SP / async-P2P / position_ids / tag 语义由 :meth:`_pipeline_forward_hidden`
        （pipeline_forward.py）集中维护，避免多套 PP forward 漂移。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        batch = int(sequences.shape[0])
        multimodal_inputs = self._multimodal_inputs_from_metadata(metadata, batch_size=batch)
        output, _present, stage_input, send_work = self._pipeline_forward_hidden(
            input_ids=sequences,
            hidden_states=None,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            multimodal_inputs=multimodal_inputs,
            position_input_ids=sequences,
            apply_lm_head=False,
            timing=timing,
            comm=comm,
            tag=tag,
            debug_label="fwd",
        )
        return output, stage_input, send_work

    # ── PP 1F1B schedule ─────────────────────────────────────────────────────

    def _pipeline_train_batch_one_f_one_b(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        self.loss_fn.policy_ratio_clip_eps = policy_ratio_clip_eps
        self.model.train()
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        norm_before = self.model.training_progress_norm()
        batch_size = int(self.config.native.micro_batch_size)
        pipeline_micro_batch_size = max(1, int(self.config.native.pp_micro_batch_size))
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
        max_chunks_per_optimizer_step = 0
        configured_inflight = int(self.config.native.pp_max_inflight_microbatches)
        _pp_schedule = "one_f_one_b"  # 实际调度策略名（由 scheduler 返回更新）
        pipeline_forward_sec = 0.0
        pipeline_backward_sec = 0.0
        # Single pass — no repeated iterations.
        optimize_round = 0
        round_started_at = time.monotonic()
        #: R3（2026-09-23）：首步逐 rank 数值探针**只落一行**的幂等标记（每个 rank 各自判）。
        probe_emitted = False
        indices = self._shared_training_indices(len(experiences), optimize_round=optimize_round)
        # ★ E-80（(D) 最小侵入式修复，2026-10-05 指挥官批准）：
        #   PP 路径原**完全不消费** ``gradient_accumulation_micro_batches`` ⇒ 每处理 1 条
        #   experience 就 clip+step 一次（实测该批 31–55 条 ⇒ 等效 LR 放大约 31–55×）⇒ 与本包
        #   非 PP 路径（累积后统一 step；T034 实测 optimizer_steps=1）语义不一致。
        #   此处引入「组」：按该字段分组、组内累积、组末才 clip+step；
        #   **门控与 ``reduced_grad_flags`` 仍每轮执行**（保全 PP rank 的集合调用次序）。
        grad_accum_group = max(1, int(self.config.training.gradient_accumulation_micro_batches))
        _starts = list(range(0, len(indices) - batch_size + 1, batch_size))
        _group_poisoned = False
        _metric_iters = 0
        for _i, start in enumerate(_starts):
            do_step = ((_i + 1) % grad_accum_group == 0) or (_i == len(_starts) - 1)
            if _i % grad_accum_group == 0:
                _group_poisoned = False
            batch_indices = indices[start : start + batch_size]
            chunk_batches = [
                collate_experiences(
                    [
                        experiences[idx]
                        for idx in batch_indices[
                            chunk_start : chunk_start + pipeline_micro_batch_size
                        ]
                    ],
                    self.device,
                )
                for chunk_start in range(0, len(batch_indices), pipeline_micro_batch_size)
            ]
            if not chunk_batches:
                continue
            max_chunks_per_optimizer_step = max(max_chunks_per_optimizer_step, len(chunk_batches))
            # ★ E-80：只在**组首**清零 ⇒ 组内梯度累积（⛔ 非组首不清零）
            if self.optimizer is not None and _i % grad_accum_group == 0:
                self.optimizer.zero_grad(set_to_none=True)
            result = self._pipeline_one_f_one_b_optimizer_step(
                chunk_batches,
                full_batch_size=len(batch_indices),
                timing=stage_timing,
                max_inflight=configured_inflight,
            )
            micro_batch_forward_sec += result["forward_sec"]
            backward_sec += result["backward_sec"]
            fill_sec += result["fill_sec"]
            steady_sec += result["steady_sec"]
            drain_sec += result["drain_sec"]
            _pp_schedule = result.get("pp_schedule", "one_f_one_b")
            pipeline_forward_sec += float(result.get("pipeline_forward_sec", 0.0))
            pipeline_backward_sec += float(result.get("pipeline_backward_sec", 0.0))
            micro_batch_count += len(chunk_batches)
            if not result["finite"]:
                # ★ E-80（组内非有限策略，指挥官 2026-10-05 定）：丢弃整组已累积梯度 ⇒
                #   zero_grad；skipped_nonfinite 按**组**计 1 次；本组不 step。
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                if not _group_poisoned:
                    _group_poisoned = True
                    skipped_nonfinite += 1
                continue
            # ── R1（2026-09-23）：梯度有限性 / 填充度判据 ─────────────────────────
            # ``result["finite"]`` 只反映末 stage 的 loss 是否有限；梯度是否有限此前
            # 完全不判 ⇒ NaN 梯度会被下面的 ``optimizer.step()`` 写进权重（T018 实测：
            # rank2 在 loss 仍有限时梯度已 NaN，照常 step）。判据是纯函数，见
            # ``models/common/grad_probe.py::grad_gate_verdict``。
            local_grad = self.model.gradient_finiteness_report()
            grad_nonfinite_any, grad_unpopulated_any = reduced_grad_flags(
                local_report=local_grad, distributed=dist, device=self.device
            )
            grad_failed, grad_fail_reason = grad_gate_verdict(
                loss_all_finite=True,  # 已过上面的 loss 门
                grad_nonfinite_any=grad_nonfinite_any,
                grad_unpopulated_any=grad_unpopulated_any,
            )
            if grad_failed:
                # 与上面 loss 门同构：zero_grad + 计数跳过（**不做 optimizer.step()**）。
                # 本包的边界：RL 的"硬失败"仍由既有判定层（A7/A2 读 skipped_nonfinite）
                # 负责，这里不新增 raise（不改变 RL 的失败语义；见实施报告"范围与边界"）。
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                if not _group_poisoned:
                    _group_poisoned = True
                    skipped_nonfinite += 1
                # 诊断明细只算一次（失败步才做第二遍 max|·| 归约）
                grad_detail = self.model.gradient_finiteness_report(with_max_abs=True)
                if self._train_batch_call_index == 0 and not probe_emitted:
                    probe_emitted = True
                    self._emit_rank_memory_event(
                        PP_NUMERIC_PROBE_PHASE,
                        {
                            PP_NUMERIC_PROBE_PHASE: {
                                "step": step_index_one_based(self._train_batch_call_index),
                                "pp_rank": self.pp_rank,
                                "pp_size": self.pp_size,
                                "loss_all_finite": True,
                                "optimizer_stepped": False,
                                "fail_reason": grad_fail_reason,
                                **grad_detail,
                            }
                        },
                    )
                self._emit_rank_memory_event(
                    FAIL_CLOSED_PHASE,
                    {
                        "metrics": {
                            "step": step_index_one_based(self._train_batch_call_index),
                            "phase_kind": FAIL_CLOSED_PHASE,
                            "fail_reason": grad_fail_reason,
                            "fail_reason_text": grad_fail_reason_text(grad_fail_reason),
                            "optimizer_steps": 0,
                            "optimizer_stepped": False,
                            "skipped_nonfinite": int(skipped_nonfinite),
                            "loss_all_finite": True,
                            "pp_rank": self.pp_rank,
                            "pp_size": self.pp_size,
                            **grad_detail,
                        }
                    },
                )
                continue
            if self._train_batch_call_index == 0 and not probe_emitted:
                # R3（2026-09-23）：首步逐 rank 数值探针（只增行、不参与判据）。
                probe_emitted = True
                self._emit_rank_memory_event(
                    PP_NUMERIC_PROBE_PHASE,
                    {
                        PP_NUMERIC_PROBE_PHASE: {
                            "step": step_index_one_based(self._train_batch_call_index),
                            "pp_rank": self.pp_rank,
                            "pp_size": self.pp_size,
                            "loss_all_finite": True,
                            "optimizer_stepped": True,
                            **self.model.gradient_finiteness_report(with_max_abs=True),
                        }
                    },
                )
            # DP/TP gradient sync: 跨 rank 同步梯度后再 clip + step
            from graspo.flow.backend.native.lora.lora_linear import (
                _sync_dp_lora_grads,
                _sync_nonsharded_lora_grads,
            )
            from graspo.flow.backend.native.parallel.tensor_utils import (
                _TENSOR_PARALLEL_GROUP,
                _TENSOR_PARALLEL_SIZE,
            )

            if self.tp_state is not None and self.tp_state.dp_group is not None:
                _sync_dp_lora_grads(self.model, self.tp_state.dp_group)
            if _TENSOR_PARALLEL_GROUP is not None and _TENSOR_PARALLEL_SIZE > 1:
                _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            grad_clip_started_at = time.monotonic()
            # ★ E-84（候选 A：PP 下按【全局】梯度范数裁剪，与 1 卡语义对齐）
            #   逐 stage 各自的 clip_grad_norm_ 只覆盖本 rank 分片 ⇒ 同一个 max_grad_norm
            #   作用在两个不同口径的范数上 ⇒ 与 1 卡（全模型范数）语义不一致。
            #   先求本 rank 分片的范数平方 ⇒ 跨 PP rank 归约(SUM) ⇒ 开方得全模型范数。
            #   ★ 集合调用次序保全：下述 all_reduce 无条件执行（各 rank 每轮都跑）。
            local_sq = torch.zeros(1, device=self.device)
            for _p in trainable_params:
                if _p.grad is not None:
                    local_sq += _p.grad.detach().float().pow(2).sum()
            _pp_grp = self.tp_state.pp_group if self.tp_state is not None else None
            if _pp_grp is not None and self.pp_size > 1:
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=_pp_grp)
            global_norm = float(local_sq.sqrt().item())
            if global_norm > 0.0:
                _clip_coef = min(1.0, float(max_grad_norm) / (global_norm + 1e-6))
                for _p in trainable_params:
                    if _p.grad is not None:
                        _p.grad.mul_(_clip_coef)
            grad_norm = torch.tensor(global_norm, device=self.device)
            _add_pipeline_stage_timing(stage_timing, "pipeline_grad_clip_sec", grad_clip_started_at)
            self._sync_timing()
            optimizer_started_at = time.monotonic()
            if self.optimizer is not None and do_step:
                _log_cuda_mem("before_optimizer_step_pp")
                self.optimizer.step()
                _log_cuda_mem("after_optimizer_step_pp")
            self._sync_timing()
            _add_pipeline_stage_timing(
                stage_timing, "pipeline_optimizer_step_sec", optimizer_started_at
            )
            optimizer_step_sec += time.monotonic() - optimizer_started_at
            if do_step and self.scheduler is not None and self.optimizer is not None:
                self.scheduler.step()
            if do_step:
                optimizer_steps += 1
            _metric_iters += 1
            loss_tensor = torch.tensor(
                [float(result["loss_value"])], dtype=torch.float, device=self.device
            )
            # 只有末 PP stage 计算真实 loss（见 _pipeline_one_f_one_b_optimizer_step：
            # 非末 stage 的 loss_value 恒为 0.0）。因此用 SUM 聚合，MIN 会把末 stage
            # 的非零 loss 压成 0（因为非末 stage 贡献 0.0），导致 loss 恒为 0。
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_sum += float(loss_tensor.item())
            grad_norm_sum += float(grad_norm.detach().float().cpu())
            nonzero_grad_count += self.model.training_progress_grad_count()
        round_secs.append(time.monotonic() - round_started_at)
        self._train_batch_call_index += 1
        norm_after = self.model.training_progress_norm()
        effective_inflight = max_chunks_per_optimizer_step
        if configured_inflight > 0:
            effective_inflight = min(effective_inflight, configured_inflight)
        metrics = {
            "optimized": optimizer_steps > 0,
            "replay_buffer_trainable_completion_count": len(experiences),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / _metric_iters if _metric_iters else None,
            "grad_norm_mean": grad_norm_sum / _metric_iters if _metric_iters else None,
            **grad_count_event(self.config.effective_tuner_type, count=nonzero_grad_count),
            **training_norm_event(
                self.config.effective_tuner_type, before=norm_before, after=norm_after
            ),
            "activation_checkpointing_enabled": bool(
                getattr(self.model, "gradient_checkpointing", False)
            ),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "pp_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": (self.placement.strategy if self.placement is not None else None),
            "pp_schedule": _pp_schedule,
            "pipeline_pp_micro_batch_size": pipeline_micro_batch_size,
            "pipeline_chunks_per_optimizer_step": max_chunks_per_optimizer_step,
            "pp_max_inflight_microbatches": effective_inflight,
            "pipeline_inflight_bound_source": "optimizer_step_chunks",
            "pipeline_fill_sec": fill_sec,
            "pipeline_steady_sec": steady_sec,
            "pipeline_drain_sec": drain_sec,
            "pipeline_forward_sec": pipeline_forward_sec,
            "pipeline_backward_sec": pipeline_backward_sec,
            "pipeline_backpressure_wait_sec": float(stage_timing.get("pipeline_recv_sec") or 0.0)
            + float(stage_timing.get("pipeline_send_sec") or 0.0)
            + float(stage_timing.get("pipeline_grad_recv_sec") or 0.0)
            + float(stage_timing.get("pipeline_grad_send_sec") or 0.0),
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            "synchronize_cuda_timing": bool(self.config.native.synchronize_cuda_timing),
            "current_lr": self._current_lr(),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("pipeline_train_batch_after", {"metrics": metrics})
        return metrics

    def _pipeline_one_f_one_b_optimizer_step(
        self,
        chunk_batches: list[Any],
        *,
        full_batch_size: int,
        timing: dict[str, float | int],
        max_inflight: int,
    ) -> dict[str, Any]:
        # max_inflight 用于 PipelineComm 背压（有界在途 send work），此处传参以复用。
        chunk_count = len(chunk_batches)
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
        forward_sec = 0.0
        backward_sec = 0.0
        fill_sec = 0.0
        steady_sec = 0.0
        drain_sec = 0.0

        def forward_chunk(chunk_idx: int) -> None:
            nonlocal forward_sec
            batch = chunk_batches[chunk_idx]
            self._sync_timing()
            forward_started_at = time.monotonic()
            stage_output, stage_input, send_work = self._pipeline_forward_for_training(
                batch.sequences,
                batch.attention_mask,
                metadata=batch.metadata,
                timing=timing,
                comm=comm,
                tag=chunk_idx,
            )
            self._sync_timing()
            forward_sec += time.monotonic() - forward_started_at
            if send_work is not None:
                send_works.append(send_work)
            loss: torch.Tensor | None = None
            finite = True
            loss_value = 0.0
            if self.pp_rank == self.pp_size - 1:
                assert stage_output is not None
                assert self.model is not None
                assert isinstance(self.model, Qwen35HybridTextModel)
                assert self.model.norm is not None and self.model.lm_head is not None
                norm_started_at = time.monotonic()
                hidden = self.model.norm(stage_output)
                _add_pipeline_stage_timing(timing, "pipeline_norm_sec", norm_started_at)
                lm_head_started_at = time.monotonic()
                log_probs = _selected_token_log_probs_from_hidden(
                    hidden[:, :-1].float(),
                    self.model.lm_head.weight.float(),
                    batch.sequences[:, 1:],
                )
                _add_pipeline_stage_timing(timing, "pipeline_lm_head_sec", lm_head_started_at)
                loss_started_at = time.monotonic()
                chunk_loss = self.loss_fn(
                    log_probs,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.action_mask,
                )
                _add_pipeline_stage_timing(timing, "pipeline_loss_sec", loss_started_at)
                finite = bool(torch.isfinite(chunk_loss).detach().cpu())
                # 累积权重按**样本数**占比：与 RL loss 自身的口径一致，故正确。
                # ``GRASPORippleLoss`` 是"batch 内逐样本 token 均值再取未加权平均"
                # （``masked_mean(..., dim=-1).mean()``），即每个样本权重恒为 1/N，
                # 与 chunk 内样本数无关 ⇒ 逐 chunk 权重 n_i/N 恰好合成同一口径
                # （Σ_i (L_i·mean_i)·n_i/N == mean over all samples）。
                # ⚠️ 不要照抄 SFT·PP 的"按有效 token 数加权"（F-13'）——那里的
                # chunk_loss 是**该 chunk 的 token 均值**，两者口径不同不可互换。
                weight = float(batch.sequences.shape[0]) / max(1, int(full_batch_size))
                loss = chunk_loss * weight if finite else None
                loss_value = float(chunk_loss.detach().cpu()) * weight if finite else 0.0
            records[chunk_idx] = {
                "stage_output": stage_output,
                "stage_input": stage_input,
                "loss": loss,
                "batch": batch,
            }
            finite_flags[chunk_idx] = finite
            loss_values[chunk_idx] = loss_value

        def backward_chunk(chunk_idx: int) -> None:
            nonlocal backward_sec
            record = records[chunk_idx]
            if record is None:
                raise RuntimeError("1F1B attempted backward before forward")
            self._sync_timing()
            backward_started_at = time.monotonic()
            assert self.tp_state is not None
            if self.pp_rank == self.pp_size - 1:
                stage_input = record["stage_input"]
                loss = record["loss"]
                if loss is not None:
                    autograd_started_at = time.monotonic()
                    _log_cuda_mem("before_backward_pp")
                    loss.backward()
                    _log_cuda_mem("after_backward_pp")
                    _add_pipeline_stage_timing(
                        timing, "pipeline_backward_autograd_sec", autograd_started_at
                    )
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    grad_send_started_at = time.monotonic()
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            else:
                stage_output = record["stage_output"]
                assert stage_output is not None
                grad_output = torch.empty_like(stage_output)
                grad_recv_started_at = time.monotonic()
                recv_work = comm.bwd_recv(
                    grad_output,
                    src=int(self.tp_state.next_pp_rank or 0),
                    tag=chunk_count + chunk_idx,
                )
                comm.wait(recv_work)  # 阻塞直到梯度到达（上游异步 send，不会死锁）
                _add_pipeline_stage_timing(timing, "pipeline_grad_recv_sec", grad_recv_started_at)
                autograd_started_at = time.monotonic()
                stage_output.backward(grad_output)
                _add_pipeline_stage_timing(
                    timing, "pipeline_backward_autograd_sec", autograd_started_at
                )
                stage_input = record["stage_input"]
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    grad_send_started_at = time.monotonic()
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            self._sync_timing()
            backward_sec += time.monotonic() - backward_started_at
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
        # 同步所有异步发送（梯度已被上游消费），保证 buffer 生命周期安全
        wait_all(send_works)

        all_finite = all(finite_flags)
        finite_tensor = torch.tensor([all_finite], dtype=torch.int, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        all_finite = bool(finite_tensor.item())
        return {
            "finite": all_finite,
            "loss_value": sum(loss_values),
            "forward_sec": forward_sec,
            "backward_sec": backward_sec,
            "fill_sec": fill_sec,
            "steady_sec": steady_sec,
            "drain_sec": drain_sec,
            "pp_schedule": sched_stats.get("pp_schedule", "one_f_one_b"),
            "pipeline_forward_sec": float(sched_stats.get("pipeline_forward_sec", 0.0)),
            "pipeline_backward_sec": float(sched_stats.get("pipeline_backward_sec", 0.0)),
        }
