"""MsSwift RL(GRPO) 后端：在 ms-swift ``GRPOTrainer`` 上注入 graspo ``ripple`` 算法核。

**职责边界（宪法 §1.1 一事一责）**

- **本模块**：msswift 后端的 **RL(GRPO) 训练器**——把 graspo 的算法核（``ripple``）
  接进 ms-swift 的 GRPO 训练循环，并提供 ``graspo.backends`` 注册表需要的训练器门面。
- **不负责**：参数映射（``_config_mapping.py``）、数据集转换（``dataset.py``）、
  奖励适配（``reward.py``）、SFT（``sft_trainer.py``）。

**接入路线（决策 D6：Python API 作库，非 CLI 透传）**

``import swift`` 作库，通过 ms-swift 的公开扩展点 ``TrainerFactory.TRAINER_MAPPING``
把 ``graspo`` 的 GRPO 训练器挂到 ``rlhf_type='grpo'`` 上，再用
``swift.pipelines.rlhf_main`` 的等价入口 ``SwiftRLHF`` 跑训练。整条链路在本进程内完成：
**没有子进程、没有 shell、不把 graspo YAML 喂给 ms-swift CLI**（ms-swift 不认识它）。

**实测 API 事实（T1 复验，ms-swift 4.5.3，2026-09-16）**

- ``swift.llm`` 子模块**不存在**（4.3.2 与 4.5.3 均实测缺失）→ 决策 D6 写作时的
  ``from swift.llm import ...`` 已失效。实际入口是 ``swift.pipelines``（``sft_main`` /
  ``rlhf_main``）、``swift.rlhf_trainers.GRPOTrainer``、``swift.trainers.TrainerFactory``。
- ``RLHFArguments.rlhf_type`` 是 ``Literal[...]``，**不接受** ``graspo_grpo`` 这类自定义名
  → 因此 Graspo 训练器挂在 ``'grpo'`` 键上，且只在本次运行的上下文内生效、退出即还原
  （见 :func:`registered_trainer_class`）。

**四个重写方法（D6 / M1）**

| 方法 | 注入内容 |
|---|---|
| ``_score_completions`` | 父类 ms-swift 奖励打分之后，追加 graspo
**字符级结构标注**（S/V/T/W/E/D） |
| ``_compute_advantages`` | 父类标量 advantage 之后，按 rollout group 计算 graspo
**token 级 advantage** |
| ``_postprocess_batch`` | 用 graspo per-token advantage **覆盖** ``grpo_batch.advantages`` |
| ``_compute_loss_and_metrics`` | 用 ``GraspoAlgorithmCore.compute_loss`` 替换内置 loss，
**修复 ratio 恒 1** |

**ratio 恒 1 是什么 bug（必须说清楚，否则"修好了"无从验证）**

修复前（E1 遗留代码 ``trainer.py:114``）传给 loss 的两个张量是**同一个对象**
（``log_probs=grpo_batch.old_per_token_logps`` 且 ``old_log_probs=同一个``），
于是 ``ratio = exp(logp − logp) ≡ 1.0``，PPO-clip 的两个分支完全相同、
裁剪永不触发、策略梯度退化为"advantage 加权和"。

修复后：``log_probs`` 取**当前策略前向**（``_get_per_token_logps_and_entropies(model, …)``），
``old_log_probs`` 取 ms-swift 在 rollout/前一批算出的基线。两者不再恒等，
ratio 随之 ≠ 1，并以 ``graspo/ratio_*`` 指标落进 ms-swift 的日志。
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from graspo.flow.msswift._config_mapping import graspo_to_ms_swift_argv, launcher_env
from graspo.flow.msswift._logps_shape_guard import check_logps_shapes
from graspo.flow.msswift.dataset import prepare_ms_swift_dataset
from graspo.flow.msswift.reward import (
    GRASPO_REWARD_NAME,
    GRASPO_TARGETS_COLUMN,
    register_graspo_reward,
)
from graspo.ripple.annotation.char_tag import CharTag

logger = logging.getLogger(__name__)

#: ms-swift 缺失时的统一提示（单一真相源，不在别处再写一份）。
MS_SWIFT_RL_PREREQUISITE = (
    "backend='msswift' RL(GRPO) requires the ms-swift package "
    "(pip install graspo[msswift] / pip install ms-swift). "
    "Native RL is fully available: set backend: native in the YAML."
)

try:  # pragma: no cover - 分支取决于运行环境是否装了 ms-swift
    from swift.rlhf_trainers import GRPOTrainer as _MsSwiftGRPOTrainerBase
except ImportError:  # ms-swift 未安装：模块仍可导入，实例化时才给出精确错误
    _MsSwiftGRPOTrainerBase = object  # type: ignore[assignment,misc]


def _require_ms_swift() -> None:
    """ms-swift 未安装时抛出精确可操作的 ``RuntimeError``（不静默降级）。"""
    if _MsSwiftGRPOTrainerBase is object:
        raise RuntimeError(MS_SWIFT_RL_PREREQUISITE)


class GraspoMsSwiftGRPOTrainer(_MsSwiftGRPOTrainerBase):  # type: ignore[misc,valid-type]
    """ms-swift ``GRPOTrainer`` 子类：保留 ms-swift 的训练循环与基础设施，只替换算法。

    Args:
        graspo_config: ``GraspoConfig`` 实例，由 :class:`GraspoMsSwiftRlhfPipeline`
            通过 ``_get_trainer_kwargs`` 显式注入（不用全局变量、不用环境变量——
            宪法 §2.2 显式即防呆）。
        *args / **kwargs: 原样交给 ``GRPOTrainer``（model / ref_model / args / template /
            train_dataset / reward_funcs / …）。
    """

    def __init__(self, *args: Any, graspo_config: Any = None, **kwargs: Any) -> None:
        """按注入的 graspo 配置装配 ``GRPOTrainer`` 包装（§2.2 显式即防呆）。

        Args:
            *args: 原样交给 ``GRPOTrainer``。
            graspo_config: ``GraspoConfig`` 实例（必填，由 pipeline 注入）。
            **kwargs: 原样交给 ``GRPOTrainer``（model / ref_model / args / template /
                train_dataset / reward_funcs / …）。
        """
        _require_ms_swift()
        if graspo_config is None:
            raise ValueError(
                "graspo_config is required for GraspoMsSwiftGRPOTrainer — construct it via "
                "GraspoMsSwiftRlhfPipeline (which injects the graspo config), not directly."
            )
        self._graspo_config = graspo_config
        # 算法核与 native 同源（ripple.algorithm_core），此处只做装配，不重复实现（§1.1/§1.3）。
        from graspo.ripple.algorithm_core import GraspoAlgorithmCore

        self._graspo = GraspoAlgorithmCore(
            reward_config=graspo_config.reward,
            policy_ratio_clip_eps=graspo_config.training.policy_ratio_clip_eps,
        )
        # per-sample 派生状态：键用 request_id（随样本跨 SP gather 一起走，见 _sample_key）。
        self._graspo_annotations: dict[str, Any] = {}
        # completion 文本必须在 `_score_completions` 阶段就抓下来：
        # `_prepare_batch_inputs` 会把 assistant 消息的 content **原地替换成 token ids**
        # （`rlhf_trainers/utils.py:encode_sample` → `replace_assistant_response_with_ids`），
        # 而 `_compute_advantages` / `_postprocess_batch` 都在那之后运行——晚一步就只能
        # 拿到 token id 列表（实测踩过：拿 `str([1,2,3])` 去做字符级标注 → 全 0 优势）。
        self._graspo_completion_text: dict[str, str] = {}
        self._graspo_token_advantages: dict[str, list[float]] = {}
        self._graspo_injection_failures: int = 0
        # 组决策台账（G3 有效性缺口）：本 rollout batch 内每个 group 的 classify_group 结果。
        # 复位点在 `_compute_graspo_token_advantages`（每个 batch 一次），
        # 消费点是 `_overwrite_advantages`（跳过训练）与 `_record_ratio_metrics`（指标）。
        self._graspo_group_decisions: list[Any] = []
        #: 本 step 是否真的把 graspo token 级 advantage 注进了 loss。false 时
        #: ``graspo/advantage_abs_mean`` 读的是 **ms-swift 的标量 advantage**（回退路径），
        #: 不是 graspo 的值——这个区分必须显式（E3 实测：两者会被混读）。
        self._graspo_advantages_injected: bool = False
        super().__init__(*args, **kwargs)

    # ── 重写 1：评分 ──────────────────────────────────────────────────────

    def _score_completions(self, samples: list[Any]) -> list[Any]:
        """父类奖励打分 + graspo 字符级结构标注（S/V/T/W/E/D）。"""
        samples = super()._score_completions(samples)
        self._annotate_with_graspo(samples)
        return samples

    # ── 重写 2：优势计算 ──────────────────────────────────────────────────

    def _compute_advantages(
        self,
        samples: list[Any],
        rewards_per_func: Any,
        batch_encoded_inputs: list[dict[str, Any]],
    ) -> Any:
        """父类标量 advantage + graspo 按 rollout group 的 token 级 advantage。

        顺序有意为之：``super()`` 先跑（它可以改写 ``samples``——DAPO 动态采样会
        重采样），graspo 再基于**它返回的**样本列表算 token 级 advantage，
        与 ``_postprocess_batch`` 拿到的是同一份列表。
        """
        scalar_advantages = super()._compute_advantages(
            samples, rewards_per_func, batch_encoded_inputs
        )
        self._compute_graspo_token_advantages(samples, rewards_per_func)
        return scalar_advantages

    # ── 重写 3：后处理 ────────────────────────────────────────────────────

    def _postprocess_batch(
        self,
        samples: list[Any],
        batch_encoded_inputs: list[dict[str, Any]],
    ) -> None:
        """父类后处理之后，用 graspo per-token advantage 覆盖 ``grpo_batch.advantages``。"""
        super()._postprocess_batch(samples, batch_encoded_inputs)
        self._overwrite_advantages(samples, batch_encoded_inputs)

    # ── 重写 4：损失与指标 ────────────────────────────────────────────────

    def _compute_loss_and_metrics(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        grpo_batch: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """用 ``GraspoAlgorithmCore.compute_loss`` 替换 ms-swift 内置 GRPO loss。

        **与父类的唯一区别**：``log_probs`` 取**当前策略前向**（而非旧策略的副本），
        因此 ``ratio = exp(logp_cur − logp_old)`` 不再恒等于 1。
        """
        per_token_logps, _entropies = self._get_per_token_logps_and_entropies(
            model, model_inputs, grpo_batch, compute_entropy=False
        )
        completion_mask = grpo_batch.completion_mask
        old_per_token_logps = grpo_batch.old_per_token_logps
        if old_per_token_logps is None:
            # 透明退路（§3.2）：没有旧策略基线时 ratio ≡ 1 是数学必然，不是被掩盖的 bug。
            # 必须让用户知道 PPO 裁剪此刻未生效。
            logger.warning(
                "graspo: grpo_batch.old_per_token_logps is None — falling back to the current "
                "forward as the PPO baseline, so the importance ratio is identically 1.0 for this "
                "step and clipping cannot trigger. Enable a rollout logprob source (e.g. vLLM "
                "rollout) to get a real ratio."
            )
            old_per_token_logps = per_token_logps.detach()

        advantages = grpo_batch.advantages
        if advantages is None:
            raise RuntimeError(
                "grpo_batch.advantages is None — graspo loss needs per-token advantages; "
                "this indicates _postprocess_batch did not run (ms-swift API drift?)."
            )

        loss = self._graspo.compute_loss(
            per_token_logps,
            old_per_token_logps,
            advantages,
            completion_mask,
        )
        self._record_ratio_metrics(
            per_token_logps, old_per_token_logps, completion_mask, advantages, loss
        )

        mode = "train" if self.model.training else "eval"
        token_count = completion_mask.sum().clamp(min=1.0)
        return loss, {
            "mode": mode,
            "entropy": {},
            "completion_mask": completion_mask,
            "completion_token_count": token_count,
        }

    # ── 守卫：per-token logps 的形状自洽（fail-closed，只报错不改数值）──────

    def _get_logps_via_local_forward(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        logits_to_keep: int,
        input_ids: Any,
        compute_entropy: bool = False,
    ) -> Any:
        """父类实现 + 进入两处切片前做形状守卫（**不改数值**）。

        为什么在 graspo 侧做（宪法 §1.2：不改上游，只在扩展点适配）：

        ms-swift 4.5.3 ``swift/rlhf_trainers/grpo_trainer.py:1570/1573`` 的两处切片，在
        "本次前向序列长度 ``S ≤ logits_to_keep``" 时行数/列数**恰好差 1**，随后在
        ``trl.trainer.utils.selective_log_softmax`` 的 gather 处抛一个不含张量语义的
        ``RuntimeError: Size does not match at dimension 0 ...``（T037 实测签名：
        ``expected index [2382, 1] to be no larger than self [2381, 248320]``）。
        守卫把同一个失败提前成带六个量与 file:line 的 :class:`LogpsShapeGuardError`。

        判据的推导见 :mod:`graspo.flow.msswift._logps_shape_guard`。
        """
        check_logps_shapes(
            input_ids=input_ids,
            logits_to_keep=int(logits_to_keep),
            padding_free=bool(getattr(self.template, "padding_free", False)),
            is_multimodal=bool(getattr(self, "is_multimodal", False)),
            dynamic_num_samples=bool(getattr(self, "dynamic_num_samples", False)),
            model=model,
        )
        return super()._get_logps_via_local_forward(
            model, model_inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy
        )

    # ── 内部：标注 / 优势 / 覆盖 / 指标 ───────────────────────────────────

    def _annotate_with_graspo(self, samples: list[Any]) -> None:
        """对每条 completion 做字符级结构标注（缺 targets 时透明降级并记警告）。"""
        tokenizer = self._resolve_tokenizer()
        if tokenizer is None:
            self._warn_injection("no tokenizer is available on the trainer")
            return
        for sample in samples:
            targets = _sample_targets(sample)
            if targets is None:
                self._warn_injection(
                    f"sample {_sample_key(sample)!r} has no {GRASPO_TARGETS_COLUMN!r} column"
                )
                continue
            key = _sample_key(sample)
            completion_text = self._completion_text(sample)
            self._graspo_completion_text[key] = completion_text
            try:
                self._graspo_annotations[key] = self._graspo.annotate_completion(
                    completion_text,
                    targets,
                    tokenizer,
                    _format_type(sample),
                    check_json_markdown=self._graspo_config.reward.check_json_markdown,
                    check_think=self._graspo_config.reward.check_think,
                )
            except Exception as exc:  # noqa: BLE001 - 降级必须带可读原因（§13.1）
                self._warn_injection(f"annotation failed for {key!r}: {exc!r}")

    def _compute_graspo_token_advantages(
        self, samples: list[Any], rewards_per_func: Any = None
    ) -> None:
        """按 rollout group 计算 token 级 advantage，并注入 **native 同源**的组决策。

        **为什么必须注入 ``classify_group``**（G3 有效性缺口，2026-09-16）

        native 侧在 rollout 之后先用 ``classify_group`` 把每个 group 判为
        ``PERFECT_SKIP`` / ``RETRY`` / ``TRAINABLE`` / ``INVALID``，只有
        ``should_train`` 的组才进 replay buffer（``flow/trainer/rollout.py:262``）。
        msswift 路径原先只做"标注 → token 级 advantage"，**没有这层决策**：于是
        整组不可解析（8B 模型输出散文、无 JSON 围栏 → 标注全是 TEXT → 每个 token 的
        advantage 都是 0）时，链路会**静默产出全 0 advantage**——loss 0、梯度 0，
        而指标看起来一切正常。E2a 如实记录了这一点（``graspo/advantage_abs_mean: 0``）。

        修复后：不可训练的组被判定出来并**整组清零 + 计数**，可训练的组产出非零
        token 级 advantage。判据与 native 逐一对应：

        ==========================================  ==========================================
        native（``flow/trainer/rollout.py``）        msswift（本方法）
        ==========================================  ==========================================
        ``rewards``（打分函数的 group 内得分）        同一张量表按 ``reward_weights`` 加权求和
        ``content_scores``（各条 completion 的内容分） ``0.0`` if 有结构标注 else ``1.0``
        ``best_completion_has_parse_error``          最高分那条的标注里**没有** S/V/E 标签
        ``rollout_max_retries``（配置）              ``training.rollout_max_retries``
        ``perfect_skip_reward_threshold``（配置）    ``training.perfect_skip_reward_threshold``
        ``reject_unparseable_groups``（配置）        ``training.reject_unparseable_groups``
        ``should_train`` → 进 buffer                ``should_train`` → 注入 advantage
        ==========================================  ==========================================

        Args:
            samples: ms-swift 的 ``GRPOSample`` 列表（本进程可见的切片）。
            rewards_per_func: ``[N_global, num_reward_funcs]`` 的全局奖励表；``None`` 时
                各条奖励按 0.0 参与分类（跳过决策仍生效，但 INVALID/RETRY 的判据会退化）。
        """
        tokenizer = self._resolve_tokenizer()
        self._graspo_token_advantages = {}
        self._graspo_group_decisions = []
        if tokenizer is None or not self._graspo_annotations:
            return
        rewards = _group_rewards(samples, rewards_per_func, getattr(self, "reward_weights", None))
        num_generations = int(getattr(self, "num_generations", 1) or 1)
        groups = _rollout_groups(samples, num_generations)
        # 逐批摘要走 DEBUG（不是 WARNING）：它是排查用的上下文，不是用户需要被告知的事件。
        # 用户需要知道的只有"这个组没产出训练信号"（下面的 WARNING）。
        logger.debug(
            "graspo: advantage pass — num_generations=%d, local samples=%d, groups=%s, "
            "prompt_ids=%s, rewards_resolved=%d",
            num_generations,
            len(samples),
            [len(group) for group in groups],
            [getattr(sample, "prompt_id", None) for sample in samples],
            len(rewards),
        )
        for group in groups:
            keys = [_sample_key(sample) for sample in group]
            if not all(key in self._graspo_annotations for key in keys):
                self._warn_injection(f"incomplete graspo annotations for group {keys}")
                continue
            targets = [_sample_targets(sample) for sample in group]
            if any(target is None for target in targets):
                self._warn_injection(f"missing {GRASPO_TARGETS_COLUMN!r} in group {keys}")
                continue
            # ``compute_group_advantages`` 收的是**一个样本的** targets 列表（native 同口径：
            # `flow/trainer/rollout.py` 传 `state.sample.targets`），不是"每组一份 targets 列表"。
            # 传入 list-of-list 会让 `gt_value_for_field` 对 list 调 `.get` → AttributeError。
            group_targets = list(targets[0])
            format_types = {_format_type(sample) for sample in group}
            if len(format_types) > 1:
                self._warn_injection(
                    f"mixed format_type within group {keys}: {sorted(format_types)}"
                )
                continue
            annotations = [self._graspo_annotations[key] for key in keys]
            try:
                ragged = self._graspo.compute_group_advantages(
                    completions=[self._completion_text(sample) for sample in group],
                    annotations=annotations,
                    targets=group_targets,
                    tokenizer=tokenizer,
                    format_type=format_types.pop(),
                    numeric_tolerance=self._graspo_config.reward.numeric_tolerance,
                    truncated_by_max=[bool(sample.is_truncated) for sample in group],
                )
            except Exception as exc:  # noqa: BLE001 - 降级必须带可读原因（§13.1）
                self._warn_injection(
                    f"token advantage computation failed for group {keys}: {exc!r}"
                )
                continue

            group_rewards = [rewards.get(key, 0.0) for key in keys]
            content_scores = [
                1.0 if _is_structurally_annotated(ann) else 0.0 for ann in annotations
            ]
            best = max(range(len(keys)), key=lambda index: group_rewards[index])
            decision = self._graspo.classify_group(
                group_rewards,
                content_scores,
                retry_count=0,
                rollout_max_retries=self._graspo_config.training.rollout_max_retries,
                perfect_skip_reward_threshold=(
                    self._graspo_config.training.perfect_skip_reward_threshold
                ),
                best_completion_has_parse_error=content_scores[best] == 0.0,
                reject_unparseable_groups=self._graspo_config.training.reject_unparseable_groups,
            )
            logger.debug(
                "graspo: classify_group — rewards=%s content_scores=%s best=%d "
                "best_has_parse_error=%s -> decision=%s should_train=%s",
                [round(float(value), 6) for value in group_rewards],
                content_scores,
                best,
                content_scores[best] == 0.0,
                decision.decision.value,
                decision.should_train,
            )
            self._graspo_group_decisions.append((keys, group_rewards, content_scores, decision))
            if not decision.should_train:
                # 组决策生效：整组不进 advantage（native 侧同样不进 replay buffer）。
                # 不删 ``self._graspo_token_advantages`` 里的键（本来就还没写），
                # 由 ``_overwrite_advantages`` 的"全有或全无"继续兜住缺失样本。
                logger.warning(
                    "graspo: rollout group skipped by classify_group (decision=%s, "
                    "rewards=%s, content_scores=%s, keys=%s) — this group produces NO training "
                    "signal; the native backend would not have enqueued it either.",
                    decision.decision.value,
                    [round(float(value), 4) for value in group_rewards],
                    content_scores,
                    keys,
                )
                continue
            for key, per_token in zip(keys, ragged):
                self._graspo_token_advantages[key] = list(per_token)

    def _overwrite_advantages(
        self,
        samples: list[Any],
        batch_encoded_inputs: list[dict[str, Any]],
    ) -> None:
        """把 graspo per-token advantage 写进每个 ``grpo_batch``（全有或全无）。

        全有或全无（§2.3 边界校验）：只要本 micro-batch 里有任何一条样本缺 graspo
        advantage，就整批保留父类的 advantage 并记警告——绝不把"部分覆盖、部分为零"
        的半成品送进 loss（那会让某些样本被静默地按 0 优势训练）。
        """
        import torch

        if not self._graspo_token_advantages:
            # 没有任何可训练组：保留父类的标量 advantage（透明退路，§3.2），
            # 但把"这一批的 advantage 不是 graspo 的"这件事记进指标。
            self._graspo_advantages_injected = False
            return
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batch_encoded_inputs):
            raise RuntimeError(
                f"graspo advantage injection misaligned: {len(chunks)} sample chunks vs "
                f"{len(batch_encoded_inputs)} encoded batches"
            )
        for chunk, encoded in zip(chunks, batch_encoded_inputs):
            keys = [_sample_key(sample) for sample in chunk]
            missing = [key for key in keys if key not in self._graspo_token_advantages]
            if missing:
                self._warn_injection(f"chunk is missing graspo advantages for {missing}")
                continue
            grpo_batch = encoded["grpo_batch"]
            mask = grpo_batch.completion_mask
            dtype = (
                grpo_batch.advantages.dtype if grpo_batch.advantages is not None else torch.float32
            )
            advantages = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
            for row, key in enumerate(keys):
                positions = mask[row].nonzero(as_tuple=True)[0]
                per_token = self._graspo_token_advantages[key]
                count = min(len(per_token), int(positions.numel()))
                if count:
                    advantages[row, positions[:count]] = torch.tensor(
                        per_token[:count], dtype=dtype, device=mask.device
                    )
            grpo_batch.advantages = advantages * mask.to(dtype)
        self._graspo_advantages_injected = True

    def _record_ratio_metrics(
        self,
        per_token_logps: Any,
        old_per_token_logps: Any,
        completion_mask: Any,
        advantages: Any,
        loss: Any,
    ) -> None:
        """记录 PPO ratio 的真实取值（ratio ≠ 1 即 G3 的产物证据）。"""
        import torch

        mode = "train" if self.model.training else "eval"
        with torch.no_grad():
            mask = completion_mask.float()
            ratio = (per_token_logps.detach().float() - old_per_token_logps.float()).exp()
            denom = mask.sum().clamp(min=1.0)
            deviation = (ratio - 1.0).abs() * mask
            ratio_mean = float((ratio * mask).sum() / denom)
            dev_mean = float(deviation.sum() / denom)
            dev_max = float(deviation.max())
            is_one = float(dev_max == 0.0)
            advantage_abs_mean = float((advantages.detach().float().abs() * mask).sum() / denom)
        metrics = self._metrics[mode]
        metrics["graspo/ratio_mean"].append(ratio_mean)
        metrics["graspo/ratio_abs_dev_mean"].append(dev_mean)
        metrics["graspo/ratio_abs_dev_max"].append(dev_max)
        metrics["graspo/ratio_is_exactly_one"].append(is_one)
        metrics["graspo/loss"].append(float(loss.detach()))
        metrics["graspo/advantage_abs_mean"].append(advantage_abs_mean)
        metrics["graspo/injection_failures"].append(float(self._graspo_injection_failures))
        # 组决策台账（G3 缺口修复的可观测面）：跳过/训练/不可解析各多少组，
        # 以及被跳过的原因（classify_group 的 decision 取值）。这样"advantage 全 0"
        # 一定能被区分成两种情形：**真的没有可训练组**（skipped>0 且有原因）
        # 还是**链路坏了**（groups_seen>0 但 trainable+skipped 对不上）。
        decisions = [
            entry[3].decision.value for entry in getattr(self, "_graspo_group_decisions", [])
        ]
        trainable_groups = sum(
            1 for entry in getattr(self, "_graspo_group_decisions", []) if entry[3].should_train
        )
        metrics["graspo/groups_seen"].append(float(len(decisions)))
        metrics["graspo/groups_trainable"].append(float(trainable_groups))
        metrics["graspo/skipped_groups"].append(float(len(decisions) - trainable_groups))
        # 不可解析（没有 S/V/E 标注）的组数：这是"真实 ARD 数据上几乎没有学习信号"
        # 的直接读数，与"链路坏了"区分开。
        metrics["graspo/unparseable_groups"].append(
            float(
                sum(
                    1
                    for entry in getattr(self, "_graspo_group_decisions", [])
                    if all(score == 0.0 for score in entry[2])
                )
            )
        )
        metrics["graspo/token_advantage_abs_mean"].append(
            _ragged_abs_mean(self._graspo_token_advantages)
        )
        # `graspo/advantages_injected` = 1 表示上面那个 advantage_abs_mean 读的是
        # **graspo** 的 token 级 advantage；= 0 表示它是 ms-swift 的标量回退值。
        # 没有这一位，读指标的人会把两条完全不同的链路当成一回事（E3 实测教训）。
        metrics["graspo/advantages_injected"].append(
            1.0 if getattr(self, "_graspo_advantages_injected", False) else 0.0
        )
        logger.info(
            "graspo ppo ratio: mean=%.10f mean|Δ|=%.6e max|Δ|=%.6e ratio_is_exactly_one=%s "
            "loss=%.6e adv|mean|=%.6e groups=%s",
            ratio_mean,
            dev_mean,
            dev_max,
            is_one == 1.0,
            float(loss.detach()),
            advantage_abs_mean,
            {
                "seen": len(decisions),
                "trainable": trainable_groups,
                "decisions": decisions,
                "injected": bool(getattr(self, "_graspo_advantages_injected", False)),
            },
        )

    def _completion_text(self, sample: Any) -> str:
        """取 rollout 生成的 completion 文本。

        优先用 ``_score_completions`` 阶段抓下的原文（那时 content 还是 str），
        回退到按 ms-swift 口径解码 token ids。**绝不**把 token id 列表 ``str()`` 掉
        当文本用——那会让字符级标注与 token 对齐彻底错位（实测踩过）。
        """
        cached = self._graspo_completion_text.get(_sample_key(sample))
        if cached is not None:
            return cached
        return _decode_completion(sample, self._resolve_tokenizer())

    def _resolve_tokenizer(self) -> Any:
        """取 ms-swift 侧的分词器。

        不同 ms-swift 版本把分词器放在不同属性上（``processing_class`` / ``tokenizer``），
        这里逐项显式探测——属于"跨版本兼容性处理"，不是探测本项目的业务接口（§2.2）。
        ProcessorMixin 形态（多模态）再向下取一层 ``.tokenizer``。
        """
        for attribute in ("processing_class", "tokenizer"):
            candidate = getattr(self, attribute, None)
            if candidate is None:
                continue
            inner = getattr(candidate, "tokenizer", None)
            resolved = inner if inner is not None else candidate
            if callable(resolved):
                return resolved
        return None

    def _warn_injection(self, reason: str) -> None:
        """透明退路：记录一次可读警告并计数（不吞异常、不静默）。"""
        self._graspo_injection_failures += 1
        logger.warning(
            "graspo: token-level advantage injection skipped (%s); falling back to ms-swift "
            "scalar advantages for this batch. Count=%d",
            reason,
            self._graspo_injection_failures,
        )


# ── ms-swift 扩展点：TrainerFactory 注册（作用域内生效，退出即还原）────────────


@contextlib.contextmanager
def registered_trainer_class(rlhf_type: str, trainer_cls: type) -> Iterator[None]:
    """在 ms-swift 的 ``TrainerFactory`` 上临时把 ``rlhf_type`` 指向 graspo 训练器。

    这是 ms-swift 的**公开扩展点**（它自己的 ``--external_plugins`` 也走这里），
    因此不需要改 ms-swift 源码（宪法 §1.2）。用上下文管理器而不是永久改写：
    退出即还原，避免同一进程内的其它 ms-swift 用法被静默改变（§2.2 显式即防呆）。
    """
    from swift.trainers import TrainerFactory  # 延迟导入：本模块在无 ms-swift 时可导入

    mapping = TrainerFactory.TRAINER_MAPPING
    # ms-swift 的 TRAINER_MAPPING 值是 **点分** 路径
    # （`get_cls` 用 `rsplit('.', 1)` 切模块与类名，见 trainer_factory.py:53-55），
    # 例如 'swift.trainers.Seq2SeqTrainer'。写成 `module:Class` 会被切成
    # module='pkg.mod'、class='last:Class' → AttributeError（实测踩过）。
    dotted = f"{trainer_cls.__module__}.{trainer_cls.__qualname__}"
    previous = mapping.get(rlhf_type)
    mapping[rlhf_type] = dotted
    logger.info("graspo: registered ms-swift trainer for rlhf_type=%r -> %s", rlhf_type, dotted)
    try:
        yield
    finally:
        if previous is None:
            mapping.pop(rlhf_type, None)
        else:
            mapping[rlhf_type] = previous


_PIPELINE_CLASS_CACHE: dict[str, type] = {}


def _graspo_rlhf_pipeline_class() -> type:
    """惰性构造 ``SwiftRLHF`` 子类（ms-swift 未安装时本模块仍可导入）。

    子类做两件事，各有唯一职责：
    1. ``_get_trainer_kwargs``：把 ``graspo_config`` **显式**注入训练器（唯一注入点）；
    2. ``run``：在作用域内把 ``TrainerFactory`` 的 ``rlhf_type`` 换成 graspo 训练器。
    """
    if "cls" not in _PIPELINE_CLASS_CACHE:
        try:
            from swift.pipelines.train.rlhf import SwiftRLHF
        except ImportError as exc:
            raise RuntimeError(MS_SWIFT_RL_PREREQUISITE) from exc

        class GraspoMsSwiftRlhfPipeline(SwiftRLHF):  # type: ignore[misc,valid-type]
            """ms-swift RLHF 流水线 + graspo 算法核（等价于 ``rlhf_main``，只是换了训练器）。"""

            def __init__(self, args: Any = None, *, graspo_config: Any) -> None:
                self._graspo_config = graspo_config
                super().__init__(args)

            def _get_trainer_kwargs(self) -> dict[str, Any]:
                kwargs = super()._get_trainer_kwargs()
                kwargs["graspo_config"] = self._graspo_config
                return kwargs

            def run(self) -> Any:
                rlhf_type = getattr(self.args, "rlhf_type", "grpo")
                with registered_trainer_class(rlhf_type, GraspoMsSwiftGRPOTrainer):
                    return super().run()

        GraspoMsSwiftRlhfPipeline.__qualname__ = "GraspoMsSwiftRlhfPipeline"
        _PIPELINE_CLASS_CACHE["cls"] = GraspoMsSwiftRlhfPipeline
    return _PIPELINE_CLASS_CACHE["cls"]


def run_graspo_rlhf(argv: list[str], *, graspo_config: Any) -> Any:
    """跑一次 graspo-msswift 的 RLHF 训练（Python API，进程内）。

    **必须传 argv 列表而不是参数对象**：ms-swift 的参数类有自己的 ``parse_args``
    （别名、默认值、嵌套字段、类别字面量校验都在里面），直接
    ``RLHFArguments(argv)`` 会走上另一条构造路径并**静默丢掉大部分参数**
    （实测：``model`` 变成 None → ``ValueError: Please set --model``）。
    ``SwiftPipeline._parse_args`` 才是正确入口，传列表给它即可。

    ``graspo_config`` 同时决定 **RoPE 键名适配**是否启用（``rope_scaling`` 非 None 时
    才在模型加载边界装补丁，见 ``_rope_compat.rope_parameters_compatible``）
    以及 **on-policy rollout 播种**是否启用（见下）。

    **rollout 播种（可复现性措施，不改变算法语义）**

    ``train_method=graspo`` 在 ``use_vllm=false`` 下现场采样走 ms-swift 的
    ``TransformersEngine``，它从**全局 torch RNG** 取随机数、不接
    ``RequestConfig.seed``；而 ms-swift 只在 trainer ``__init__`` 播一次种
    （``rlhf_trainers/grpo_trainer.py:143`` 的 ``set_seed(..., device_specific=True)``
    ⇒ 有效种子 = ``seed + rank``），rollout 前不重播 ⇒ 多卡时同 config 同 seed 的
    两跑生成内容可以不同（T033 4 卡真机定案：``mean_length`` 37.5 vs 39.0）。

    这里用 ``_rollout_seed.rollout_seeding`` —— 与 OPD 通道**同一个**入口（§1.4）
    —— 把整次训练包起来，使每次 rollout 生成之前全局 RNG 都从一个**由
    (``training.seed``, 进程 rank, 该进程内第几次推理调用) 唯一确定**的值起步：

    - **确定性** ⇒ 同 config/seed/卡位双跑逐位相同；
    - **rank 区分**（种子里带 ``RANK_STRIDE * rank``）⇒ 复现上游
      ``device_specific=True`` 的意图，跨 rank 不再生成重复补全，**GRPO 组内方差
      不被压掉**（上一版播同一个字面值，把 4 卡的补全压成同一长度、advantage 全零）；
    - **调用序号区分** ⇒ 相邻两次 rollout 不复用同一条随机流。

    只钉随机性起点，不动温度/top_p/top_k/采样分布，也不强制 greedy。
    """
    from graspo.flow.msswift._rollout_seed import rollout_seeding
    from graspo.flow.msswift._rope_compat import rope_parameters_compatible

    pipeline_cls = _graspo_rlhf_pipeline_class()
    with (
        rope_parameters_compatible(graspo_config.msswift.rope_scaling),
        rollout_seeding(graspo_config),
    ):
        return pipeline_cls(argv, graspo_config=graspo_config).main()


# ── 训练器门面（``graspo.backends`` 注册表契约：``factory(config, selection)``）────


class MsSwiftRlTrainer:
    """RL(GRPO) 的 msswift 训练器门面（延迟解析：构造不导入 ms-swift / torch）。

    接口契约与 native 侧一致（``cli/train_worker.py`` 依赖）：``.train(smoke: bool)``。
    """

    def __init__(self, config: Any, selection: Any = None) -> None:
        self.config = config
        self.selection = selection

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<MsSwiftRlTrainer train_method={getattr(self.config, 'train_method', '?')!r}>"

    def train(self, *, smoke: bool = False) -> None:
        """执行 ms-swift 后端的 RL(GRPO) 训练。

        Args:
            smoke: 冒烟边界——只跑 1 个 optimizer step（基础设施参数，不改训练语义）。

        Raises:
            RuntimeError: ms-swift 未安装 / 训练器类解析失败（附可操作指引）。
        """
        _require_ms_swift()

        from graspo.flow.msswift._config_mapping import (
            native_only_field_notes,
            native_only_fields,
            validate_combinations,
        )

        # 前置校验（E2b 缺陷 3/4 的防御性报错改进，**非能力修复**）。
        validate_combinations(self.config)
        register_graspo_reward()
        notes = native_only_field_notes()
        for field in native_only_fields(self.config):
            logger.warning(
                "graspo: %s is not mapped by the msswift backend — %s", field, notes[field]
            )

        output_dir = Path(str(self.config.training.output_dir))
        work_dir = output_dir / "msswift"
        dataset_path = prepare_ms_swift_dataset(self.config, stage="rlhf", work_dir=work_dir)

        # 冒烟边界 = **一个 rollout batch 的完整优化轮次**（num_iterations 步）：
        # num_iterations=1 时就是"跑 1 步"；>1 时跑满一轮，这样"策略相对基线发生
        # 移动"才能被观察到（PPO ratio 修复的验证场景）。
        num_iterations = int(self.config.msswift.num_iterations or 1)
        extra_argv = ["--max_steps", str(max(1, num_iterations))] if smoke else []
        if smoke:
            # 冒烟要拿到 checkpoint 落盘证据，因此显式规定"每 1 步保存一次"。
            extra_argv += [
                "--save_strategy",
                "steps",
                "--save_steps",
                "1",
                "--save_total_limit",
                "1",
                "--logging_steps",
                "1",
                "--log_completions",
                "true",
            ]
        # 每 rank 首步探针（只读旁路；默认关 ⇒ 不注册回调、不追加任何参数）。
        # 与 SFT 通道**同一个接线点**（`first_step_probe.probe_active_extra_argv`，§1.4）：
        # 探针实现与字段在 ms-swift 的 `--callbacks` + `callbacks_map` 扩展点上复用，
        # 本通道不另造一份（§1.2 对扩展开放、§14.4 不 fork 第三方）。
        # ★ 为什么 GRPO 需要它：本档的首步 loss 是**跨 rank 均值**
        #   （`swift/trainers/mixin.py` 的 `nested_gather(tr_loss).mean()`），
        #   只有 per-rank 的 `input_ids_sha256` + `local_loss` 旁路才能把
        #   "输入侧不同" 与 "同输入不同结果（数值侧）" 分开（判读规则见探针模块 docstring）。
        from graspo.flow.msswift.first_step_probe import probe_active_extra_argv

        extra_argv += probe_active_extra_argv()
        argv = graspo_to_ms_swift_argv(
            self.config,
            stage="rlhf",
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            extra_argv=extra_argv,
        )
        argv += ["--reward_funcs", GRASPO_REWARD_NAME]

        # S1 数据并行：ms-swift 侧由 launcher 环境变量承载（T1 复验结论）。
        os.environ.update(launcher_env(self.config))

        # argv 是运行产物的完整描述（§10.1）；打日志便于事后复现与排查。
        logger.warning("graspo msswift RL argv: %s", " ".join(argv))
        run_graspo_rlhf(argv, graspo_config=self.config)


def create_msswift_trainer(config: Any, selection: Any = None) -> MsSwiftRlTrainer:
    """``graspo.backends`` 注册表中 ``msswift`` 项的工厂（RL 路径）。

    Returns:
        ``MsSwiftRlTrainer``；构造不触发 ms-swift / torch 导入。
    """
    return MsSwiftRlTrainer(config, selection)


# ── 样本辅助（纯函数，不依赖 self 状态）────────────────────────────────────


def _sample_key(sample: Any) -> str:
    """样本在派生状态字典里的键。

    优先 ``request_id``——它随样本一起被 ``all_gather_object`` 跨 SP 进程组传递，
    因此在序列并行下仍然唯一且稳定；缺失时回退到 ``prompt_id``。
    """
    request_id = getattr(sample, "request_id", "") or ""
    if request_id:
        return str(request_id)
    prompt_id = getattr(sample, "prompt_id", "") or ""
    return f"prompt:{prompt_id}" if prompt_id else f"id:{id(sample)}"


def _is_structurally_annotated(annotation: Any) -> bool:
    """这条 completion 的标注里是否**真的**出现了结构角色（S/V/E）。

    对应 native 的 ``content_scores[i] > 0``：native 的 ``content_score`` 只有当
    JSON 围栏 / tool_call 标签被识别、且至少有一个字段被比对时才非零。纯散文
    （8B 模型在真实 ARD 样例上的典型输出）标注结果全是 ``TEXT`` → 这里返回 ``False``
    → ``classify_group`` 的 ``best_completion_has_parse_error=True`` → 组被
    RETRY/INVALID 拦下，而不再静默产出全 0 advantage。

    判据只取"可训练角色"（``CharTag.trainable``：S/V/E）——``T``(think) / ``W``(waste) /
    ``D``(dropped) 的 advantage 恒为 0，出现它们不代表这条 completion 有可用结构。
    """
    return any(
        tag in (CharTag.STRUCTURE, CharTag.VALUE, CharTag.ERROR)
        for tag in getattr(annotation, "tags", []) or []
    )


def _group_rewards(
    samples: list[Any], rewards_per_func: Any, reward_weights: Any
) -> dict[str, float]:
    """把 ms-swift 的 ``rewards_per_func`` 表还原成"每条样本一个加权标量"。

    ms-swift 的奖励表是 ``[N_global, num_reward_funcs]``（``_compute_rewards_per_func``
    里 gather 过），而 ``_compute_advantages`` 拿到的是**本进程的样本切片**。对应关系
    按 ``request_id`` 而不是行号——行号在序列并行/多进程下会被 gather 重排，而
    ``request_id`` 是随样本一起传递的稳定标识（与 :func:`_sample_key` 同一真相源）。

    Returns:
        ``{sample_key: reward}``；拿不到表或表行数不匹配时返回 ``{}``（调用方按 0.0 处理，
        并因 ``classify_group`` 的阈值语义退化为"全部需要重采样"——**不会**伪装成可训练）。
    """
    if rewards_per_func is None or not samples:
        return {}
    try:
        import torch

        weights = reward_weights
        if weights is None:
            weights = torch.ones(rewards_per_func.shape[1], dtype=rewards_per_func.dtype)
        else:
            weights = torch.as_tensor(weights, dtype=rewards_per_func.dtype)
        weighted = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1).tolist()
    except Exception:  # noqa: BLE001 - 奖励表形态异常时不让分类污染训练（§13.1 有痕降级）
        logger.warning(
            "graspo: could not reduce rewards_per_func to per-sample rewards", exc_info=True
        )
        return {}
    ordered = _global_sample_order(samples)
    if len(ordered) != len(weighted):
        logger.warning(
            "graspo: rewards_per_func has %d rows but %d samples were ordered by request_id — "
            "falling back to positional pairing for this batch.",
            len(weighted),
            len(ordered),
        )
    return {
        _sample_key(sample): float(weighted[index])
        for index, sample in enumerate(ordered)
        if index < len(weighted)
    }


def _ragged_abs_mean(advantages: dict[str, list[float]]) -> float:
    """所有可训练组的 per-token advantage 的 |mean|（G3 的"非零信号"读数）。

    ``graspo/advantage_abs_mean`` 读的是**已注入 loss 的** advantage 张量（权威判据）；
    本函数读的是**注入前**的 ragged 值，用来区分"算出来是 0"与"注进去变 0"。
    """
    values = [abs(float(value)) for per_token in advantages.values() for value in per_token]
    return sum(values) / len(values) if values else 0.0


def _global_sample_order(samples: list[Any]) -> list[Any]:
    """按 ``request_id`` 排序的样本列表（gather 的顺序即该顺序；缺失时保持原序）。"""
    if not all(getattr(sample, "request_id", "") for sample in samples):
        return list(samples)
    return sorted(samples, key=lambda sample: str(sample.request_id))


def _sample_targets(sample: Any) -> list[dict[str, Any]] | None:
    """从数据集透传列里取 graspo ``targets``（JSON 字符串 → list）。"""
    import json

    extra = getattr(sample, "extra", None) or {}
    raw = extra.get(GRASPO_TARGETS_COLUMN)
    if raw is None:
        return None
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _decode_completion(sample: Any, tokenizer: Any) -> str:
    """把已 token 化的 completion 解回文本（镜像 ms-swift ``_log_rollout`` 的做法）。

    ``_prepare_batch_inputs`` 之后，assistant 消息的 content 可能是：
    ``str``（未替换）、``list[int]``（token ids）、或 ``{"token_ids": [...]}``。
    这里按 ms-swift 自己的口径逐种显式处理，不猜。
    """
    for message in reversed(getattr(sample, "messages", []) or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            content = content.get("token_ids")
        if isinstance(content, list) and tokenizer is not None:
            return tokenizer.decode(content)
        return ""
    return ""


def _format_type(sample: Any) -> str:
    """graspo 标注的格式族：tool_call（有 tool_calls 目标）或 json。

    与 native 侧同判据（``Sample.expects_tool_calls``）：目标里出现 ``tool_calls``
    即为工具调用格式，否则按 fenced JSON 处理。
    """
    targets = _sample_targets(sample) or []
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if isinstance(output, dict) and output.get("tool_calls") is not None:
            return "tool_call"
    return "json"


def _rollout_groups(samples: list[Any], num_generations: int) -> list[list[Any]]:
    """把样本切成 rollout group。

    判据优先用 ``prompt_id``（同一 prompt 的 ``num_generations`` 条 completion 同组）；
    ``prompt_id`` 缺失时按 ``num_generations`` 顺序切块——ms-swift 的 ``RepeatSampler``
    就是以 mini_repeat_count=num_generations 复制的，两种判据在这里等价。
    """
    if num_generations <= 1 or len(samples) <= num_generations:
        return [list(samples)]
    if all(getattr(sample, "prompt_id", "") for sample in samples):
        grouped: OrderedDict[str, list[Any]] = OrderedDict()
        for sample in samples:
            grouped.setdefault(str(sample.prompt_id), []).append(sample)
        return list(grouped.values())
    return [
        list(samples[start : start + num_generations])
        for start in range(0, len(samples), num_generations)
    ]
