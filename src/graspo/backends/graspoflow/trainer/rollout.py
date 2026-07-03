"""GraspoFlowTrainer rollout 生成与评分的 mixin。"""

from __future__ import annotations

import logging
import time
from typing import Any

from graspo.backends.graspoflow.trainer.helpers import (
    expand_advantages_like,
    experience_metadata_for_row,
    generated_token_counts,
    group_stats,
    public_generation_metadata,
    reward_detail,
    safe_sample_metadata,
)
from graspo.backends.graspoflow.trainer.stats import _AttemptRecord, _QueuedSample
from graspo.backends.graspoflow.trainer.summary import (
    monitor_group,
    scalar_generation_timing,
)
from graspo.core.buffer import Experience
from graspo.core.completion import raw_parsed_completion
from graspo.core.graspo_parity import classify_group, has_reward_variance


class RolloutMixin:
    """Rollout 生成、评分、决策、replay buffer 追加的 mixin。"""

    config: Any
    runtime: Any
    reward: Any
    replay_buffer: Any
    stats: Any
    current_epoch_stats: Any
    recent_groups: Any
    pending_batch_attempts: Any
    pending_batch_timings: Any
    logger: Any
    sample_index: int
    global_step: int

    # ── 生成 ──────────────────────────────────────────────────────────────────

    def _generate_groups(self, samples: list[Any]) -> list[Any]:
        """为 samples 生成 rollout groups。"""
        message_batches = [sample.messages for sample in samples]
        tool_batches = [sample.tools for sample in samples]
        generate_groups = getattr(self.runtime, "generate_groups", None)
        if callable(generate_groups):
            generations = generate_groups(
                message_batches=message_batches,
                tool_batches=tool_batches,
                rollout_group_size=self.config.training.rollout_group_size,
                max_new_tokens=self.config.training.max_new_tokens,
                max_prompt_length=self.config.data.max_prompt_length,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            if len(generations) != len(message_batches):
                raise RuntimeError(
                    f"graspoflow generate_groups returned {len(generations)} "
                    f"groups for {len(message_batches)} prompts"
                )
            return generations
        return [
            self.runtime.generate_group(
                messages=messages,
                tools=tools,
                rollout_group_size=self.config.training.rollout_group_size,
                max_new_tokens=self.config.training.max_new_tokens,
                max_prompt_length=self.config.data.max_prompt_length,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            for messages, tools in zip(message_batches, tool_batches, strict=True)
        ]

    def _generate_sample_groups(self, samples: list[Any]) -> list[Any]:
        """为 samples 生成 rollout groups（支持多模态）。"""
        if not any(sample.media for sample in samples):
            return self._generate_groups(samples)
        generate_sample_groups = getattr(self.runtime, "generate_sample_groups", None)
        if not callable(generate_sample_groups):
            raise RuntimeError(
                "Input samples contain image/video media, but the runtime "
                "does not implement multimodal generate_sample_groups"
            )
        generations = generate_sample_groups(
            samples=samples,
            rollout_group_size=self.config.training.rollout_group_size,
            max_new_tokens=self.config.training.max_new_tokens,
            max_prompt_length=self.config.data.max_prompt_length,
            temperature=self.config.training.temperature,
            top_p=self.config.training.top_p,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        if len(generations) != len(samples):
            raise RuntimeError(
                f"graspoflow generate_sample_groups returned {len(generations)} "
                f"groups for {len(samples)} samples"
            )
        return generations

    # ── 解析 completion ───────────────────────────────────────────────────────

    def _parse_completion(self, completion: str, sample: Any) -> Any:
        """解析单条 completion 为 ParsedCompletion。"""
        parse_completion = getattr(self.runtime, "parse_completion", None)
        if callable(parse_completion):
            return parse_completion(completion, sample)
        return raw_parsed_completion(completion)

    def _completion_parser_name(self) -> str:
        """获取 completion 解析器名称。"""
        adapter = getattr(self.runtime, "_adapter", None)
        if adapter is not None:
            parser_name = getattr(adapter, "completion_parser_name", None)
            if parser_name:
                return str(parser_name)
            return adapter.__class__.__name__
        return "raw"

    # ── rollout 批次 ──────────────────────────────────────────────────────────

    def _rollout_queue_attempt(
        self, active: list[_QueuedSample], *, epoch: int
    ) -> list[_AttemptRecord]:
        """对一组 active samples 执行一次 rollout 尝试。"""
        rollout_started_at = time.monotonic()
        generations = self._generate_sample_groups([state.sample for state in active])
        rollout_sec = time.monotonic() - rollout_started_at
        rollout_sec_per_prompt = rollout_sec / max(len(generations), 1)
        records: list[_AttemptRecord] = []
        for state, generation in zip(active, generations, strict=True):
            attempt_started_at = time.monotonic()
            reward_started_at = time.monotonic()
            parsed_completions = [
                self._parse_completion(text, state.sample) for text in generation.completions
            ]
            results = [
                self.reward.score_parsed(
                    parsed,
                    state.sample.targets,
                    is_tool_call=state.sample.expects_tool_calls,
                )
                for parsed in parsed_completions
            ]
            reward_cpu_sec = time.monotonic() - reward_started_at
            rewards = [float(result.reward) for result in results]
            content_scores = [float(result.content_score) for result in results]
            all_right = [bool(result.all_right) for result in results]
            reward_details = [reward_detail(result) for result in results]
            decision_started_at = time.monotonic()
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            if best_idx < len(parsed_completions):
                best_parsed = parsed_completions[best_idx]
                has_parse_error = bool(best_parsed.parse_errors)
                has_count_mismatch = False
                if state.sample.expects_tool_calls and state.sample.targets and not has_parse_error:
                    first_target = state.sample.targets[0]
                    tc = first_target.get("output", {}).get("tool_calls")
                    expected_count = len(tc) if isinstance(tc, list) else 0
                    actual_count = len(best_parsed.tool_calls)
                    has_count_mismatch = expected_count > 0 and actual_count != expected_count
                best_has_parse_error = has_parse_error or has_count_mismatch
            else:
                best_has_parse_error = False
            decision = classify_group(
                rewards,
                content_scores,
                retry_count=state.retry_count,
                rollout_max_retries=self.config.training.rollout_max_retries,
                perfect_skip_reward_threshold=self.config.training.perfect_skip_reward_threshold,
                best_completion_has_parse_error=best_has_parse_error,
                reject_unparseable_groups=self.config.training.reject_unparseable_groups,
            )
            decision_sec = time.monotonic() - decision_started_at
            self.stats.total_groups += 1
            if decision.should_retry:
                self.stats.retries += 1

            readable = self._group_payload(
                sample=state.sample,
                epoch=epoch,
                generation=generation,
                rewards=rewards,
                content_scores=content_scores,
                all_right=all_right,
                reward_details=reward_details,
                parsed_completions=parsed_completions,
                decision=decision,
                retry_count=state.retry_count,
            )
            if self._is_primary():
                self.logger.write_readable(readable)
            timing = {
                "rollout_sec": rollout_sec_per_prompt,
                "reward_cpu_sec": reward_cpu_sec,
                "decision_sec": decision_sec,
                "old_logprob_sec": 0.0,
                "replay_append_sec": 0.0,
                "attempt_total_sec": time.monotonic() - attempt_started_at,
                "decision": decision.decision.value,
                "retry_count": state.retry_count,
                "reward_max_median_gap": decision.reward_max_median_gap,
                "completion_count": len(generation.completions),
                "sequence_len": int(getattr(generation.sequences, "shape", [0, 0])[1]),
                "generated_tokens_max": max(generated_token_counts(generation), default=0),
                **scalar_generation_timing(generation.metadata or {}),
            }

            records.append(
                _AttemptRecord(
                    sample=state.sample,
                    generation=generation,
                    parsed_completions=parsed_completions,
                    rewards=rewards,
                    content_scores=content_scores,
                    all_right=all_right,
                    reward_details=reward_details,
                    decision=decision,
                    retry_count=state.retry_count,
                    readable=readable,
                    timing=timing,
                )
            )
        return records

    # ── 群体 payload 构建 ─────────────────────────────────────────────────────

    def _group_payload(
        self,
        *,
        sample: Any,
        epoch: int,
        generation: Any,
        rewards: list[float],
        content_scores: list[float],
        all_right: list[bool],
        reward_details: list[dict[str, Any]],
        parsed_completions: list[Any],
        decision: Any,
        retry_count: int,
    ) -> dict[str, Any]:
        """构建单个 rollout group 的完整 payload。"""
        payload = {
            "event": "graspo_group",
            "backend": self.backend_name,
            "epoch": epoch,
            "step": self.global_step,
            "sample_index": self.sample_index,
            "messages": sample.messages,
            "tools": sample.tools,
            "prompt_preview": sample.prompt_preview,
            "targets": sample.targets,
            "metadata": safe_sample_metadata(sample),
            "completions": generation.completions,
            "parsed_completions": [parsed.to_dict() for parsed in parsed_completions],
            "rewards": rewards,
            "content_scores": content_scores,
            "all_right": all_right,
            "reward_details": reward_details,
            "generated_tokens": generated_token_counts(generation),
            "decision": decision.decision.value,
            "attempt_index": retry_count + 1,
            "max_attempts": self.config.training.rollout_max_retries + 1,
            "retry_count": retry_count,
            "group_stats": group_stats(rewards),
            "reward_max_median_gap": decision.reward_max_median_gap,
            "generation_metadata": public_generation_metadata(generation.metadata or {}),
        }
        if decision.decision.value == "invalid_no_preference_gap":
            payload["invalid_reason"] = "no_preference_gap"
        return payload

    @staticmethod
    def _raw_generation(generation: Any) -> dict[str, Any]:
        """提取 generation 的原始 tensor 数据。"""
        return {
            "sequences": generation.sequences,
            "attention_mask": generation.attention_mask,
            "action_mask": generation.action_mask,
            "prompt_len": generation.prompt_len,
        }

    # ── 样本最终化 ────────────────────────────────────────────────────────────

    def _finalize_sample(self, state: _QueuedSample, *, epoch: int) -> bool:
        """对已完成 rollout 的样本进行最终处理：决策、追加 replay buffer。"""
        if not state.attempts:
            raise RuntimeError("Cannot finalize queued sample without rollout attempts")
        record = state.attempts[-1]
        generation = record.generation
        rewards = record.rewards
        decision = record.decision
        readable = record.readable
        timing = record.timing
        if not decision.should_train:
            if decision.decision.value == "perfect_skip":
                self.stats.perfect_skipped += 1
            elif decision.decision.value == "invalid_no_preference_gap":
                self.stats.invalid_no_preference_gap += 1
            else:
                self.stats.invalid += 1
                self._write_error_log(readable, decision.decision.value)
            if self._is_primary():
                self.logger.write_raw({**readable, "raw": self._raw_generation(generation)})
            self._commit_sample_attempts(state, epoch=epoch)
            return self._finish_sample_and_maybe_optimize(epoch=epoch)
        if not has_reward_variance(rewards):
            self.stats.invalid += 1
            self._write_error_log(readable, "no_reward_variance")
            if self._is_primary():
                self.logger.write_raw({**readable, "raw": self._raw_generation(generation)})
            timing["decision"] = "invalid"
            self._commit_sample_attempts(state, epoch=epoch)
            return self._finish_sample_and_maybe_optimize(epoch=epoch)

        old_logprob_started_at = time.monotonic()
        if isinstance(generation.metadata, dict) and "_multimodal_rows" in generation.metadata:
            old_log_probs = self.runtime.sequence_log_probs(
                generation.sequences,
                generation.attention_mask,
                metadata=generation.metadata,
            )
        else:
            old_log_probs = self.runtime.sequence_log_probs(
                generation.sequences,
                generation.attention_mask,
            )
        timing["old_logprob_sec"] = time.monotonic() - old_logprob_started_at
        replay_started_at = time.monotonic()
        advantages = expand_advantages_like(rewards, old_log_probs)
        self._append_experiences(generation, rewards, old_log_probs, advantages)
        timing["replay_append_sec"] = time.monotonic() - replay_started_at
        timing["attempt_total_sec"] = (
            float(timing.get("rollout_sec") or 0.0)
            + float(timing.get("reward_cpu_sec") or 0.0)
            + float(timing.get("decision_sec") or 0.0)
            + float(timing.get("old_logprob_sec") or 0.0)
            + float(timing.get("replay_append_sec") or 0.0)
        )
        self.stats.trainable += 1
        if decision.decision.value == "trainable_max_correct":
            self.stats.trainable_max_correct += 1
        else:
            self.stats.trainable_not_correct += 1
        if self._is_primary():
            self.logger.write_raw(
                {
                    **readable,
                    "raw": {
                        **self._raw_generation(generation),
                        "old_log_probs": old_log_probs,
                        "advantages": advantages,
                    },
                }
            )
        self._commit_sample_attempts(state, epoch=epoch)
        return self._finish_sample_and_maybe_optimize(epoch=epoch)

    def _append_experiences(
        self,
        generation: Any,
        rewards: list[float],
        old_log_probs: Any,
        advantages: Any,
    ) -> None:
        """将单条 generation 的 experiences 追加到 replay buffer。"""
        import torch

        reward_tensor = torch.tensor(
            rewards,
            dtype=old_log_probs.dtype,
            device=old_log_probs.device,
        )
        items: list[Experience] = []
        for idx in range(len(rewards)):
            items.append(
                Experience(
                    sequences=generation.sequences[idx].detach().cpu(),
                    old_log_probs=old_log_probs[idx].detach().cpu(),
                    advantages=advantages[idx].detach().cpu(),
                    attention_mask=generation.attention_mask[idx].detach().cpu(),
                    action_mask=generation.action_mask[idx].detach().cpu(),
                    rewards=reward_tensor[idx].detach().cpu(),
                    metadata=experience_metadata_for_row(generation.metadata, idx),
                )
            )
        self.replay_buffer.append_many(items)

    def _commit_sample_attempts(self, state: _QueuedSample, *, epoch: int) -> None:
        """将样本的所有 attempts 记录到监控和 batch 缓存中。"""
        for record in state.attempts:
            self.recent_groups.append(monitor_group(record.readable))
            self.pending_batch_attempts.append(record.readable)
            self._record_epoch_attempt(record.readable)
            self._record_attempt_timing(
                epoch=epoch, retry_count=record.retry_count, timing=record.timing
            )

    def _finish_sample_and_maybe_optimize(self, *, epoch: int) -> bool:
        """样本处理完成后的收尾：递增计数器，检查是否需要优化。"""
        self.current_epoch_stats.samples_seen += 1
        self.sample_index += 1
        return (
            self._maybe_optimize(epoch=epoch)
            and 0 < self.config.training.max_steps <= self.global_step
        )

    def _record_epoch_attempt(self, payload: dict[str, Any]) -> None:
        """将单次 attempt 的指标累加到当前 epoch 统计中。"""
        rewards = [float(value) for value in payload.get("rewards", [])]
        content_scores = [float(value) for value in payload.get("content_scores", [])]
        decision = str(payload.get("decision"))
        self.current_epoch_stats.attempt_groups += 1
        self.current_epoch_stats.completion_count += len(rewards)
        if rewards:
            self.current_epoch_stats.reward_mean_sum += sum(rewards) / len(rewards)
            self.current_epoch_stats.best_reward = max(
                self.current_epoch_stats.best_reward, max(rewards)
            )
        if content_scores:
            self.current_epoch_stats.content_mean_sum += sum(content_scores) / len(content_scores)
        if decision == "retry":
            self.current_epoch_stats.retries += 1
        elif decision == "perfect_skip":
            self.current_epoch_stats.perfect_skipped += 1
        elif decision == "invalid":
            self.current_epoch_stats.invalid += 1
        elif decision == "invalid_no_preference_gap":
            self.current_epoch_stats.invalid_no_preference_gap += 1
        elif decision == "trainable_max_correct":
            self.current_epoch_stats.trainable += 1
            self.current_epoch_stats.trainable_max_correct += 1
        elif decision == "trainable_not_correct":
            self.current_epoch_stats.trainable += 1
            self.current_epoch_stats.trainable_not_correct += 1

    def _record_attempt_timing(
        self, *, epoch: int, retry_count: int, timing: dict[str, Any]
    ) -> None:
        """记录单次 attempt 的 timing 数据。"""
        self.pending_batch_timings.append(timing)
        if self._is_primary():
            self.logger.write_timing_event(
                self._timing_event(
                    phase="rollout_attempt",
                    duration_sec=float(timing.get("attempt_total_sec") or 0.0),
                    epoch=epoch,
                    sample_index=self.sample_index,
                    attempt_index=retry_count + 1,
                    retry_count=retry_count,
                    details=timing,
                )
            )

    # ── 错误日志汇聚 ────────────────────────────────────────────────────────────

    def _write_error_log(self, readable: dict[str, Any], reason: str) -> None:
        """Write an ERROR-level event to the common error log.

        Called when a group is classified as invalid or has no reward variance,
        so errors are aggregated in ``logs/error.log`` for post-run inspection.
        """
        self.logger.write_error(
            {
                "event": "group_decision",
                "decision": "invalid",
                "reason": reason,
                "sample_index": self.sample_index,
                "global_step": self.global_step,
                "messages": readable.get("messages"),
                "prompt_preview": readable.get("prompt_preview"),
                "group_stats": readable.get("group_stats"),
                "invalid_reason": readable.get("invalid_reason"),
            }
        )
        logging.getLogger("graspo.trainer").error(
            "Invalid group: sample_index=%s step=%s reason=%s",
            self.sample_index,
            self.global_step,
            reason,
        )
