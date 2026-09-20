"""GraspoAlgorithmCore — 后端无关的算法核心。

组合 ripple 层已有模块：奖励评分、逐字符标注、组分类决策、
token 级 advantage 计算、PPO-clip loss、经验回放缓冲。
纯计算，零 I/O，零分布式依赖，可在 CPU 上独立测试。
不依赖 graspo.flow 的任何模块。
"""

from __future__ import annotations

from typing import Any

import torch

from graspo.core.schema import RewardConfig
from graspo.ripple.annotation.advantages import (
    compute_group_advantages,
    compute_ripple_advantages,
)
from graspo.ripple.annotation.labeler import AnnotationInput, AnnotationResult, annotate
from graspo.ripple.buffer import Experience, ReplayBuffer
from graspo.ripple.group_decision import (
    GroupSampleDecision,
    classify_group,
    replay_buffer_optimize_threshold,
    replay_ready,
)
from graspo.ripple.loss import GRASPORippleLoss
from graspo.ripple.reward.reward import GraspoReward, RewardResult, create_reward


class GraspoAlgorithmCore:
    """组合 ripple 层模块的算法核心：纯计算，零设施依赖。

    advantage/classify/replay 方法为静态方法；reward/loss/buffer 通过实例委托。
    """

    def __init__(
        self,
        reward_config: RewardConfig,
        *,
        policy_ratio_clip_eps: float = 0.2,
        replay_buffer_limit: int = 0,
    ) -> None:
        self.reward: GraspoReward = create_reward(reward_config)
        self.loss_fn: GRASPORippleLoss = GRASPORippleLoss(
            policy_ratio_clip_eps=policy_ratio_clip_eps
        )
        self.buffer: ReplayBuffer = ReplayBuffer(limit=replay_buffer_limit)

    # ── 奖励评分 ──────────────────────────────────────────────────────

    def score_completion(self, completion: str, targets: Any) -> RewardResult:
        """对一条 completion 评分（JSON 格式）。"""
        return self.reward.score(completion, targets)

    def score_parsed(
        self, parsed: Any, targets: Any, *, is_tool_call: bool = False
    ) -> RewardResult:
        """对一条已解析 completion 评分（支持 tool_call 格式）。"""
        return self.reward.score_parsed(parsed, targets, is_tool_call=is_tool_call)

    # ── 标注 ──────────────────────────────────────────────────────────

    @staticmethod
    def annotate_completion(
        completion_text: str,
        targets: list[dict[str, Any]],
        tokenizer: Any,
        format_type: str,
        *,
        check_json_markdown: bool = True,
        check_think: bool = False,
    ) -> AnnotationResult:
        """对一条 completion 做逐字符结构标注。"""
        return annotate(
            AnnotationInput(
                completion_text=completion_text,
                targets=targets,
                tokenizer=tokenizer,
                format_type=format_type,
                check_json_markdown=check_json_markdown,
                check_think=check_think,
            )
        )

    # ── 组分类 ────────────────────────────────────────────────────────

    @staticmethod
    def classify_group(
        rewards: list[float],
        content_scores: list[float],
        *,
        retry_count: int,
        rollout_max_retries: int,
        perfect_skip_reward_threshold: float = 1.0,
        best_completion_has_parse_error: bool = False,
        reject_unparseable_groups: bool = True,
    ) -> GroupSampleDecision:
        """将一组 rollout 分类为训练决策（PERFECT_SKIP / RETRY / TRAINABLE / INVALID）。"""
        return classify_group(
            rewards=rewards,
            content_scores=content_scores,
            retry_count=retry_count,
            rollout_max_retries=rollout_max_retries,
            perfect_skip_reward_threshold=perfect_skip_reward_threshold,
            best_completion_has_parse_error=best_completion_has_parse_error,
            reject_unparseable_groups=reject_unparseable_groups,
        )

    # ── Advantage 计算 ────────────────────────────────────────────────

    @staticmethod
    def compute_group_advantages(
        *,
        completions: list[str],
        annotations: list[AnnotationResult],
        targets: list[dict[str, Any]],
        tokenizer: Any,
        format_type: str,
        numeric_tolerance: float = 0.2,
        truncated_by_max: list[bool] | None = None,
    ) -> list[list[float]]:
        """为一个 rollout group 计算 per-token advantage（ragged）。

        Returns:
            ragged per-token advantage，与每条 completion 的生成 token 数对齐：
            ``S→+1.0 / V→raw−μ_f / T·W·D→0 / E→−1.0``。
        """
        return compute_group_advantages(
            completions=completions,
            annotations=annotations,
            targets=targets,
            tokenizer=tokenizer,
            format_type=format_type,
            numeric_tolerance=numeric_tolerance,
            truncated_by_max=truncated_by_max,
        )

    @staticmethod
    def compute_ripple_advantages(
        ragged_advantages: list[list[float]],
        old_log_probs: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """将 ragged per-token advantage 对齐到 ``(B, seq_len-1)`` tensor。"""
        return compute_ripple_advantages(ragged_advantages, old_log_probs, prompt_len)

    # ── Loss 计算 ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 PPO-clip loss（标量，用于反向传播）。"""
        return self.loss_fn(log_probs, old_log_probs, advantages, action_mask)

    # ── 经验回放缓冲 ──────────────────────────────────────────────────

    def add_to_buffer(self, items: list[Experience]) -> None:
        """将经验追加到回放缓冲。"""
        self.buffer.append_many(items)

    def take_from_buffer(self, count: int) -> list[Experience]:
        """从回放缓冲取前 ``count`` 条经验。"""
        return self.buffer.take(count)

    def clear_buffer(self) -> None:
        """清空回放缓冲。"""
        self.buffer.clear()

    @property
    def buffer_size(self) -> int:
        """当前缓冲中的经验数量。"""
        return len(self.buffer)

    def get_buffer_metrics(self) -> dict[str, int]:
        """回放缓冲指标摘要。"""
        return {"size": len(self.buffer), "limit": self.buffer.limit}

    # ── Replay 触发判断 ───────────────────────────────────────────────

    @staticmethod
    def replay_buffer_optimize_threshold(
        rollout_queue_batch_size: int,
        rollout_group_size: int,
    ) -> int:
        """计算触发 optimize 的缓冲阈值。"""
        return replay_buffer_optimize_threshold(rollout_queue_batch_size, rollout_group_size)

    @staticmethod
    def replay_ready(
        replay_size: int,
        rollout_queue_batch_size: int,
        rollout_group_size: int,
    ) -> bool:
        """判断缓冲是否已达到触发 optimize 的阈值。"""
        return replay_ready(replay_size, rollout_queue_batch_size, rollout_group_size)
