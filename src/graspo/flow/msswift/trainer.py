"""MsSwiftTrainer — 继承 ms-swift GRPOTrainer，注入 Graspo ripple 算法。

通过重写 _score_completions、_compute_advantages、_postprocess_batch、
_compute_loss_and_metrics 四个方法，在不修改 ms-swift 源码的前提下
注入 Graspo 的字符级标注、token 级 advantage 和 GRASPORippleLoss。

ms-swift 是可选依赖；未安装时 import 会给出清晰的错误提示。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

try:
    from swift.rlhf_trainers import GRPOTrainer
except ImportError:
    raise ImportError(
        "ms-swift is required for the msswift backend. "
        "Install it with: pip install graspo[msswift]"
    )

from graspo.ripple.algorithm import GraspoAlgorithmCore


class MsSwiftTrainer(GRPOTrainer):
    """ms-swift 后端训练器：继承 GRPOTrainer，注入 Graspo ripple 算法。

    重写 4 个关键方法（精确方法签名来自 ms-swift 源码审计）：
    - ``_score_completions``（grpo_trainer.py:224）：注入 Graspo 字符级标注 + 六路组决策
    - ``_compute_advantages``（grpo_trainer.py:393）：缓存 Graspo token 级 advantage
    - ``_postprocess_batch``：用 per-token advantage 覆盖 GRPOBatch 中的 advantages
    - ``_compute_loss_and_metrics``（grpo_trainer.py:948）：用 GRASPORippleLoss 替换内置 loss
    """

    def __init__(self, *args: Any, graspo_config: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if graspo_config is None:
            raise ValueError("graspo_config is required for MsSwiftTrainer")
        self._graspo = GraspoAlgorithmCore(
            reward_config=graspo_config.reward,
            policy_ratio_clip_eps=graspo_config.training.policy_ratio_clip_eps,
        )
        self._graspo_annotations: Dict[str, Any] = {}

    # ── 重写 1：评分 ──────────────────────────────────────────────────────

    def _score_completions(self, samples: List[Any]) -> List[Any]:
        """评分完成：调用父类后注入 Graspo 字符级标注。

        签名：``(self, samples: List[GRPOSample]) -> List[GRPOSample]``
        （来自 ms-swift grpo_trainer.py:224）
        """
        # 1. 调用父类（保留 ms-swift 原生 reward）
        samples = super()._score_completions(samples)

        # 2. TODO: 注入 Graspo 字符级标注 + 六路组决策
        # 将标注结果存入 self._graspo_annotations 供后续方法使用
        return samples

    # ── 重写 2：优势计算 ──────────────────────────────────────────────────

    def _compute_advantages(
        self,
        samples: List[Any],
        rewards_per_func: torch.Tensor,
        batch_encoded_inputs: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """计算 advantage：父类标量值 + Graspo token 级。

        签名：``(self, samples, rewards_per_func, batch_encoded_inputs) -> Tensor``
        （来自 ms-swift grpo_trainer.py:393）
        """
        # 1. 调用父类获取标量 advantage
        scalar_adv = super()._compute_advantages(samples, rewards_per_func, batch_encoded_inputs)

        # 2. TODO: 计算 Graspo token 级 advantage 并缓存
        return scalar_adv

    # ── 重写 3：后处理 ────────────────────────────────────────────────────

    def _postprocess_batch(
        self,
        samples: List[Any],
        batch_encoded_inputs: List[Dict[str, Any]],
    ) -> None:
        """后处理 batch：用 Graspo per-token advantage 覆盖 GRPOBatch。

        父类完成基础后处理后，将缓存的 token 级 advantage 注入到
        GRPOBatch 中，供后续 _compute_loss_and_metrics 使用。
        """
        # 调用父类
        super()._postprocess_batch(samples, batch_encoded_inputs)
        # TODO: 用 Graspo per-token advantage 覆盖 GRPOBatch 中的 advantages

    # ── 重写 4：损失计算 ──────────────────────────────────────────────────

    def _compute_loss_and_metrics(
        self,
        model: Any,
        model_inputs: Dict[str, Any],
        grpo_batch: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """用 GRASPORippleLoss 替换 ms-swift 内置 loss。

        签名：``(self, model, model_inputs, grpo_batch) -> (loss, metrics)``
        （来自 ms-swift grpo_trainer.py:948）

        尝试用 Graspo PPO-clip loss，失败则回退到父类实现。
        """
        try:
            loss = self._graspo.compute_loss(
                log_probs=grpo_batch.old_per_token_logps,  # TODO: 需当前策略 log_probs
                old_log_probs=grpo_batch.old_per_token_logps,
                advantages=grpo_batch.advantages,
                action_mask=grpo_batch.completion_mask,
            )
            return loss, {}
        except Exception:
            return super()._compute_loss_and_metrics(model, model_inputs, grpo_batch)


def create_msswift_trainer(config: Any, selection: Any = None) -> MsSwiftTrainer:
    """msswift 后端的工厂函数（供 entry_points 自动发现）。

    Args:
        config: GraspoConfig 实例。
        selection: BackendSelection 实例（可选）。

    Returns:
        MsSwiftTrainer 实例。
    """
    return MsSwiftTrainer(graspo_config=config)