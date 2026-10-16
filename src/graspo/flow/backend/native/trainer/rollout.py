"""GraspoFlowTrainer rollout 生成与评分的 mixin。"""

import logging
import time
from typing import Any

import torch

from graspo.flow.backend.native.trainer.helpers import (
    experience_metadata_for_row,
    generated_token_counts,
    public_generation_metadata,
    raw_generation_payload,
    safe_sample_metadata,
)
from graspo.flow.backend.native.trainer.rollout_shard import (
    gather_field_stats,
    gather_rollout_values,
    union_field_names,
)
from graspo.ripple.annotation.advantages import (
    compute_group_advantages,
    compute_ripple_advantages,
    field_means,
    group_field_stats,
)
from graspo.ripple.annotation.labeler import AnnotationInput, annotate
from graspo.ripple.buffer import Experience
from graspo.ripple.group_decision import classify_group, group_stats, has_reward_variance
from graspo.ripple.monitoring.stats import AttemptRecord, QueuedSample
from graspo.ripple.monitoring.summary import (
    monitor_group,
    scalar_generation_timing,
)
from graspo.ripple.multimodal.contract import assert_rl_training_has_multimodal
from graspo.ripple.reward.reward import reward_detail


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

    def _rollouts_per_rank(self, prompt_count: int) -> int:
        """本 step 每卡对**每条 prompt**生成/训练的 rollout 数（★ 切片量子 ``G//m``）。

        ⛔ **不复用** ``config.training.rollout_group_size`` 作为生成入参（W-1「一名两义」）：
        `rollout_group_size` 是**算法组大小 G**（决策 / ``μ_f`` 的定义域），切片后
        **生成入参**变成 ``G//m``。两者由本方法显式分开。

        ★ 取值优先用 ``train()`` 的 epoch 循环落下的 ``_ucd_rollouts_per_rank``
        （与**步边界量子 / 读数除数 / 配置阈值**同源）；未进 ``train()`` 的调用点
        （单测 / 诊断）⇒ 现算一次。``dp_size <= 1`` ⇒ 返回 ``G``（**恒等**）
        ⇒ dp=1 路径逐位不变（A-1 回归锚）。
        """
        del prompt_count  # 签名保留：量子只由 (G, m) 决定，与批大小无关
        preset = getattr(self, "_ucd_rollouts_per_rank", None)
        if preset is not None:
            return int(preset)
        from graspo.flow.backend.native.assignment import rollouts_per_rank

        return int(
            rollouts_per_rank(
                int(self.config.training.rollout_group_size), self._dp_size_for_slicing()
            )
        )

    def _dp_size_for_slicing(self) -> int:
        """切片口径下的 ``m``（= 实际 ``dp_size``）；取不到或 ≤1 时返回 1（**切片关闭**）。"""
        degree = getattr(self, "parallel_degree", None)
        dp_size = int(getattr(degree, "dp_size", 1) or 1)
        return max(1, dp_size)

    def _generate_groups(self, samples: list[Any], *, rollouts_per_rank: int | None = None) -> list[Any]:
        """为 samples 生成 rollout groups。

        :param rollouts_per_rank: ★ rollout 分卡下每卡对**每条** prompt 的 rollout 数
            （``G//m``）。``None`` ⇒ 用 ``training.rollout_group_size``（= 未切片口径）。
            ⚠ **这是行为变更**：生成微批形状 ``min(micro_batch_size, 该值)`` 与
            ``torch.multinomial`` 的 philox 消费序列随之变 ⇒ dp≥2 的产出是**新的
            realization**（统计等价、逐位不等价，见 `report.md` §7.7 / X-10）。
        """
        message_batches = [sample.messages for sample in samples]
        tool_batches = [sample.tools for sample in samples]
        group_size = (
            int(self.config.training.rollout_group_size)
            if rollouts_per_rank is None
            else int(rollouts_per_rank)
        )
        generations = self.runtime.generate_groups(
            message_batches=message_batches,
            tool_batches=tool_batches,
            rollout_group_size=group_size,
            max_new_tokens=self.config.training.max_new_tokens,
            max_prompt_length=self.config.data.max_prompt_length,
            temperature=self.config.training.temperature,
            top_p=self.config.training.top_p,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        if len(generations) != len(message_batches):
            raise RuntimeError(
                f"native generate_groups returned {len(generations)} "
                f"groups for {len(message_batches)} prompts"
            )
        return generations

    def _generate_sample_groups(
        self, samples: list[Any], *, rollouts_per_rank: int | None = None
    ) -> list[Any]:
        """为 samples 生成 rollout groups（支持多模态）。

        :param rollouts_per_rank: 同 :meth:`_generate_groups`（★ 多模态路径**也必须**切片，
            否则该路径下分卡不生效、逐卡条数仍会不等 ⇒ 死锁面回归）。
        """
        group_size = (
            int(self.config.training.rollout_group_size)
            if rollouts_per_rank is None
            else int(rollouts_per_rank)
        )
        if not any(sample.media for sample in samples):
            return self._generate_groups(samples, rollouts_per_rank=group_size)
        generations = self.runtime.generate_sample_groups(
            samples=samples,
            rollout_group_size=group_size,
            max_new_tokens=self.config.training.max_new_tokens,
            max_prompt_length=self.config.data.max_prompt_length,
            temperature=self.config.training.temperature,
            top_p=self.config.training.top_p,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        if len(generations) != len(samples):
            raise RuntimeError(
                f"native generate_sample_groups returned {len(generations)} "
                f"groups for {len(samples)} samples"
            )
        return generations

    # ── 解析 completion ───────────────────────────────────────────────────────

    def _parse_completion(self, completion: str, sample: Any) -> Any:
        """解析单条 completion 为 ParsedCompletion。"""
        return self.runtime.parse_completion(completion, sample)

    def _completion_parser_name(self) -> str:
        """获取 completion 解析器名称（ABC 声明的类属性，无需探测）。"""
        return str(self.runtime._require_adapter().completion_parser_name)

    # ── rollout 批次 ──────────────────────────────────────────────────────────

    def _rollout_queue_attempt(
        self, active: list[QueuedSample], *, epoch: int
    ) -> list[AttemptRecord]:
        """对一组 active samples 执行一次 rollout 尝试。

        ★ **rollout 分卡（U-C）**：本方法在 ``classify_group`` **之前**做一次
        ``dp_group`` 上的 ``all_gather`` 把**整组 G 条**的决策输入凑齐
        （``rewards`` / ``content_scores`` / parse-error 标志；**载荷无文本、无标注**），
        再把整组喂给 ``classify_group`` ⇒ 组决策 / retry / perfect-skip 在 m 张卡上
        **逐字段同判**（`report.md` §4-②）。``dp_size <= 1`` ⇒ **0 次新增集合**（A-1）。
        """
        rollout_started_at = time.monotonic()
        # ★ 会合长度：本卡活跃条目数**可能少于全局**（某 rank 的条目已在本轮终结）
        #   ⇒ 必须取**全局 MAX 计数**作为逐 rank 一致的会合长度。
        #   ★ 这里**当场再归约一次**（`_collective_any_pending(n_pending=n_real)` 会把
        #     全局 MAX 写进 `_ucd_max_pending` 并**返回计数本身**）：
        #     ⛔ 不复用"上一轮结束时"的 `_ucd_max_pending` —— 那是**上一轮**的待重试数，
        #     与本轮 `active` 数**语义不同**，一旦不等就会发出不同形状的 `all_gather`
        #     （集合失配 ⇒ 死锁，而不是一条清晰的报错）。多一个标量集合换来形状确定，
        #     按宪法 §2.3「边界校验即防呆」是划算的。
        #   ★ dp≤1 ⇒ `_collective_any_pending` 直接返回 `n_real > 0`、
        #     `_ucd_max_pending = n_real` ⇒ `cutoff == n_real` ⇒ **零新增集合**（A-1）。
        n_real = len(active)
        self._collective_any_pending(n_real)
        cutoff = max(int(getattr(self, "_ucd_max_pending", 0) or 0), n_real)
        work = list(active) + [active[index % n_real] for index in range(cutoff - n_real)]
        generations = self._generate_sample_groups(
            [state.sample for state in work],
            rollouts_per_rank=self._rollouts_per_rank(cutoff),
        )
        active = work
        rollout_sec = time.monotonic() - rollout_started_at
        rollout_sec_per_prompt = rollout_sec / max(len(generations), 1)
        group_size = int(self.config.training.rollout_group_size)
        rpr = self._rollouts_per_rank(cutoff)
        sharded = rpr != group_size

        # ── 第 1 遍：本地逐条算标量（reward / content / 标注）──
        local: list[dict[str, Any]] = []
        for state, generation in zip(active, generations, strict=True):
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
            local.append(
                {
                    "state": state,
                    "generation": generation,
                    "parsed_completions": parsed_completions,
                    "annotations": self._annotate_completions(state, generation),
                    "rewards": [float(result.reward) for result in results],
                    "content_scores": [float(result.content_score) for result in results],
                    "base_content_scores": [float(r.base_content_score) for r in results],
                    "all_right": [bool(r.all_right) for r in results],
                    "reward_details": [reward_detail(r) for r in results],
                    "reward_cpu_sec": reward_cpu_sec,
                }
            )

        # ── 第 2 遍：★ 决策前把整组凑齐（dp≤1 时全部返回本地值 = 零新增集合）──
        local_rewards = [torch.tensor(entry["rewards"], dtype=torch.float64) for entry in local]
        local_content = [
            torch.tensor(entry["content_scores"], dtype=torch.float64) for entry in local
        ]
        # parse-error 标志逐 **slot** 广播（补齐的 slot 记 0）⇒ 全局 `best_idx` 可直接查表。
        local_flag = [self._parse_error_flags(entry, rpr) for entry in local]
        whole_rewards = gather_rollout_values(
            self.runtime,
            local_rewards,
            cutoff=cutoff,
            rollouts_per_rank=rpr,
            what="rewards（决策输入）",
        )
        whole_content = gather_rollout_values(
            self.runtime,
            local_content,
            cutoff=cutoff,
            rollouts_per_rank=rpr,
            what="content_scores（决策输入）",
        )
        whole_flags = gather_rollout_values(
            self.runtime,
            local_flag,
            cutoff=cutoff,
            rollouts_per_rank=rpr,
            what="parse_error 标志（决策输入）",
        )
        field_stats: list[dict[str, tuple[float, int]]] | None = None
        field_names: list[str] = []
        if sharded:
            # ``μ_f`` 是组内**唯一**的跨条耦合量（`report.md` §4-③）⇒ 决策前把各卡切片的
            # `(Σraw, n)` 合并成整组统计，`_finalize_sample` 再把整组 μ 注入
            # `compute_group_advantages`。⛔ 载荷只有标量与其计数（**无文本、无标注**）；
            # ⛔ 不在本层重写 μ 的算术（走 `ripple` 的 `group_field_stats`，§1.4）。
            # ⛔ 非可训路径：`μ_f` 只在可训时用 ⇒ 下面的可训性由**整组** decision 判定，
            #    而 decision 逐卡同判 ⇒ m 张卡在本分支上同进同出（不会集合失配）。
            per_prompt = [
                self._prompt_field_stats(entry["state"], entry["generation"], entry["annotations"])
                for entry in local
            ]
            field_names = union_field_names(per_prompt)
            field_stats = gather_field_stats(
                self.runtime,
                per_prompt,
                cutoff=cutoff,
                field_names=field_names,
                what="μ_f 的 (Σraw, n)",
            )

        records: list[AttemptRecord] = []
        for index, entry in enumerate(local):
            state = entry["state"]
            generation = entry["generation"]
            parsed_completions = entry["parsed_completions"]
            attempt_started_at = time.monotonic()
            # ★ 决策输入 = **整组**（会合后）；`decision` 用整组 rewards/content/标志 ⇒
            #   逐卡同判，retry / perfect-skip 天然同步（`report.md` §4-②）。
            decision_rewards = [float(v) for v in whole_rewards[index].tolist()]
            decision_content = [float(v) for v in whole_content[index].tolist()]
            decision_flags = [int(v) for v in whole_flags[index].tolist()]
            best_idx = max(range(len(decision_rewards)), key=lambda i: decision_rewards[i])
            best_has_parse_error = bool(decision_flags[best_idx])
            decision_started_at = time.monotonic()
            decision = classify_group(
                decision_rewards,
                decision_content,
                retry_count=state.retry_count,
                rollout_max_retries=self.config.training.rollout_max_retries,
                perfect_skip_reward_threshold=self.config.training.perfect_skip_reward_threshold,
                best_completion_has_parse_error=best_has_parse_error,
                reject_unparseable_groups=self.config.training.reject_unparseable_groups,
            )
            decision_sec = time.monotonic() - decision_started_at
            # ⚠ 统计口径：只统计**真实**条目（补齐行不计入 ⇒ 与单卡的组数/retry 数一致）。
            if index < n_real:
                self.stats.total_groups += 1
                if decision.should_retry:
                    self.stats.retries += 1
            rewards = entry["rewards"]
            content_scores = entry["content_scores"]
            base_content_scores = entry["base_content_scores"]
            all_right = entry["all_right"]
            reward_details = entry["reward_details"]
            reward_cpu_sec = entry["reward_cpu_sec"]

            readable = self._group_payload(
                sample=state.sample,
                epoch=epoch,
                generation=generation,
                rewards=rewards,
                content_scores=content_scores,
                base_content_scores=base_content_scores,
                all_right=all_right,
                reward_details=reward_details,
                parsed_completions=parsed_completions,
                decision=decision,
                retry_count=state.retry_count,
            )
            if self._is_primary():
                self.logger.write_readable(readable)
            timing = {
                # 单位显式化：rollout_sec_per_prompt 是 per-prompt 分摊秒数
                # （同一队列共享 rollout_queue_sec 总耗时），字段名自证单位
                "rollout_sec_per_prompt": rollout_sec_per_prompt,
                "rollout_queue_sec": rollout_sec,
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
                AttemptRecord(
                    sample=state.sample,
                    generation=generation,
                    parsed_completions=parsed_completions,
                    rewards=rewards,
                    content_scores=content_scores,
                    base_content_scores=base_content_scores,
                    all_right=all_right,
                    reward_details=reward_details,
                    decision=decision,
                    retry_count=state.retry_count,
                    readable=readable,
                    timing=timing,
                )
            )
            if index < n_real:
                # ★ 分卡相关的会合结果存进 **flow 自己的**旁挂表（⛔ 不给算法层的
                #   `AttemptRecord` 加字段：那是 `ripple/` 的数据契约）。
                self._ucd_payload_store()[id(state)] = {
                    "whole_group_rewards": decision_rewards,
                    "whole_group_content_scores": decision_content,
                    "field_stats": field_stats[index] if field_stats is not None else None,
                    "field_names": list(field_names),
                    "annotations": entry["annotations"],
                    "is_padding": False,
                }
        return records

    # ── ★ rollout 分卡（U-C）的辅助：切片相关的标量与统计 ─────────────────────

    def _ucd_payload_store(self) -> dict[int, dict[str, Any]]:
        """分卡会合结果的**旁挂表**（按 ``id(QueuedSample)`` 索引；懒初始化）。

        为什么不塞进 ``AttemptRecord``：那是 ``ripple/``（算法层）的数据契约，
        分卡是**设施层**机制（宪法 §1.1 分层）⇒ ⛔ 不得把设施字段塞进算法层容器。

        为什么**懒**初始化：U-A 的判别测试用 ``__new__`` 绕过 ``__init__``
        直接搭 trainer（`tests/flow/backend/native/trainer/test_world_gate_rhythm.py`）
        ⇒ 不能假设 ``__init__`` 一定跑过。

        ⚠ 有界性：``_finalize_sample`` 每次 ``pop`` 掉自己那条 ⇒ 表长 = 未终结样本数。
        """
        store = getattr(self, "_ucd_shard_payload", None)
        if store is None:
            store = {}
            self._ucd_shard_payload = store
        return store

    def _annotate_completions(self, state: QueuedSample, generation: Any) -> list[Any]:
        """对本卡切片的 completions 做字符级标注（生成时按 ``G//m`` 条 ⇒ 只标自己的）。

        ★ 与 ``_finalize_sample`` 的标注**同一口径**（同一 tokenizer / format_type /
        两个 check 开关）；本方法只是把它提前到 attempt 阶段，供 ``μ_f`` 统计复用。
        """
        tokenizer = self.runtime._require_adapter().tokenizer
        if tokenizer is None:
            raise RuntimeError(
                "tokenizer is None — GRASPO requires a tokenizer for token-level"
                " advantage computation.  Ensure the model adapter sets"
                " self.tokenizer during initialization."
            )
        format_type = "tool_call" if state.sample.expects_tool_calls else "json"
        return [
            annotate(
                AnnotationInput(
                    completion_text=text,
                    targets=state.sample.targets,
                    tokenizer=tokenizer,
                    format_type=format_type,
                    check_json_markdown=bool(self.reward.config.check_json_markdown),
                    check_think=bool(self.reward.config.check_think),
                )
            )
            for text in generation.completions
        ]

    def _prompt_field_stats(
        self, state: QueuedSample, generation: Any, annotations: list[Any]
    ) -> dict[str, tuple[float, int]]:
        """单条 prompt **本卡切片**的 per-field ``(Σraw, n)``（走 ``ripple`` 的单一真相源）。"""
        tokenizer = self.runtime._require_adapter().tokenizer
        format_type = "tool_call" if state.sample.expects_tool_calls else "json"
        return group_field_stats(
            completions=list(generation.completions),
            annotations=list(annotations),
            targets=state.sample.targets,
            tokenizer=tokenizer,
            format_type=format_type,
            numeric_tolerance=float(self.reward.config.numeric_tolerance),
        )

    @staticmethod
    def _parse_error_flags(entry: dict[str, Any], rollouts_per_rank: int) -> Any:
        """本卡切片的 parse-error 标志张量 ``(G/m,)``（int8）。

        口径与改动前逐字一致：``best_completion_has_parse_error`` 的本地值 =
        「本条 completion 有解析错误 **或** ``content_score == 0``」。
        切片下每条 completion 只能看到**自己卡上那一段**，故先算**逐 slot** 标志
        （长度 = 本卡切片长度），会合后按**全局** ``best_idx`` 查表 ⇒ 与单卡整组同值。
        """
        flags = [
            1
            if (bool(parsed.parse_errors) or float(score) == 0.0)
            else 0
            for parsed, score in zip(
                entry["parsed_completions"], entry["content_scores"], strict=True
            )
        ]
        if len(flags) != rollouts_per_rank:
            raise RuntimeError(
                f"[rollout 分卡] 本卡切片条数 {len(flags)} ≠ rollouts_per_rank "
                f"{rollouts_per_rank} —— 生成入参未按 G//m 切片，会合后拼不齐整组。"
            )
        return torch.tensor(flags, dtype=torch.int8)

    # ── 群体 payload 构建 ─────────────────────────────────────────────────────

    def _group_payload(
        self,
        *,
        sample: Any,
        epoch: int,
        generation: Any,
        rewards: list[float],
        content_scores: list[float],
        base_content_scores: list[float],
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
            "base_content_scores": base_content_scores,
            "all_right": all_right,
            "reward_details": reward_details,
            "generated_tokens": generated_token_counts(generation),
            "decision": decision.decision.value,
            # 1 基序号（= retry_count+1），与 0 基的 retry_count 语义自显不混淆
            "attempt_number": retry_count + 1,
            "max_attempts": self.config.training.rollout_max_retries + 1,
            "retry_count": retry_count,
            "group_stats": group_stats(rewards),
            "reward_max_median_gap": decision.reward_max_median_gap,
            "generation_metadata": public_generation_metadata(generation.metadata or {}),
        }
        if decision.decision.value == "invalid_no_preference_gap":
            payload["invalid_reason"] = "no_preference_gap"
        return payload

    # ── 样本最终化 ────────────────────────────────────────────────────────────

    def _finalize_sample(self, state: QueuedSample, *, epoch: int) -> bool:
        """对已完成 rollout 的样本进行最终处理：决策、追加 replay buffer。

        ★ **rollout 分卡（U-C）**：第二道门（``has_reward_variance``）必须用**会合后的整组**
        ``rewards``（W2）—— 否则逐卡可训条数仍会不等（`【更正 X-01】`）。
        ``dp_size <= 1`` 时旁挂表里放的就是本地整组 ⇒ 逐位不变。
        """
        if not state.attempts:
            raise RuntimeError("Cannot finalize queued sample without rollout attempts")
        record = state.attempts[-1]
        generation = record.generation
        rewards = record.rewards
        shard = self._ucd_payload_store().pop(id(state), None)
        whole_group_rewards = (
            list(shard["whole_group_rewards"]) if shard is not None else list(rewards)
        )
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
                self.logger.write_raw({**readable, "raw": raw_generation_payload(generation)})
            self._commit_sample_attempts(state, epoch=epoch)
            return self._finish_sample_and_maybe_optimize(epoch=epoch)
        if not has_reward_variance(whole_group_rewards):
            self.stats.invalid += 1
            self._write_error_log(readable, "no_reward_variance")
            if self._is_primary():
                self.logger.write_raw({**readable, "raw": raw_generation_payload(generation)})
            timing["decision"] = "invalid"
            self._commit_sample_attempts(state, epoch=epoch)
            return self._finish_sample_and_maybe_optimize(epoch=epoch)

        old_logprob_started_at = time.monotonic()
        has_multimodal_rows = isinstance(generation.metadata, dict) and (
            "_multimodal_rows" in generation.metadata
        )
        # 防呆：多模态模型生成序列含图像 token 但 metadata 无 rows →
        # 硬失败。断链修复后此检查防止回归（视觉 LoRA 再次静默失训）。
        assert_rl_training_has_multimodal(
            generation.metadata,
            generation.sequences,
            image_token_id=getattr(
                getattr(self.runtime._require_adapter().model, "config", None),
                "image_token_id",
                None,
            ),
            expected_rows=int(generation.sequences.shape[0]),
            context="rollout old_logprob",
        )
        if has_multimodal_rows:
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
        # GRASPO-Ripple: token-level rewards and advantages
        tokenizer = self.runtime._require_adapter().tokenizer
        if tokenizer is None:
            raise RuntimeError(
                "tokenizer is None — GRASPO requires a tokenizer for token-level"
                " advantage computation.  Ensure the model adapter sets"
                " self.tokenizer during initialization."
            )
        if not record.parsed_completions:
            raise RuntimeError(
                "parsed_completions is empty — GRASPO requires parsed completions"
                " for token-level advantage computation.  This indicates a bug"
                " in the parsing pipeline or rollout logic."
            )
        # 标注驱动链路(v0.20.0):字符级标注 → 末尾 EOS 惩罚 → token 级 advantage。
        # 替换旧的 token_rewards/is_format_masks 链路(token_reward.py 已删除)。
        format_type = "tool_call" if state.sample.expects_tool_calls else "json"
        annotations = [
            annotate(
                AnnotationInput(
                    completion_text=text,
                    targets=state.sample.targets,
                    tokenizer=tokenizer,
                    format_type=format_type,
                    check_json_markdown=bool(self.reward.config.check_json_markdown),
                    check_think=bool(self.reward.config.check_think),
                )
            )
            for text in generation.completions
        ]
        truncated_by_max = [
            len(tokenizer(text, add_special_tokens=False).input_ids)
            >= self.config.training.max_new_tokens
            for text in generation.completions
        ]
        # ★ rollout 分卡：`μ_f` 必须是**整组**的（跨卡合并后的 `(Σraw, n)`）。
        #   `field_mean_override=None` ⇒ 算法层走原式（**dp=1 路径逐位不变**）。
        field_mean_override = None
        if shard is not None and shard.get("field_stats") is not None:
            field_mean_override = field_means(shard["field_stats"])
        ragged_advantages = compute_group_advantages(
            completions=generation.completions,
            annotations=annotations,
            targets=state.sample.targets,
            tokenizer=tokenizer,
            format_type=format_type,
            numeric_tolerance=float(self.reward.config.numeric_tolerance),
            truncated_by_max=truncated_by_max,
            field_mean_override=field_mean_override,
        )
        advantages = compute_ripple_advantages(
            ragged_advantages,
            old_log_probs,
            generation.prompt_len,
        )
        self._append_experiences(
            generation,
            rewards,
            old_log_probs,
            advantages,
            entry_index=self.sample_index,
            slot_offset=self._dp_rank_for_metadata() * max(1, len(generation.completions)),
        )
        timing["replay_append_sec"] = time.monotonic() - replay_started_at
        # attempt_total_sec 保持实测墙钟值（初始化时已记录），不再用组件求和覆写
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
                        **raw_generation_payload(generation),
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
        *,
        entry_index: int,
        slot_offset: int = 0,
    ) -> None:
        """将单条 generation 的 experiences 追加到 replay buffer。

        ★ **rollout 分卡（U-C）**：本卡切片里第 `idx` 条 completion 的**全局槽位** =
        ``slot_offset + idx``；两者写进 ``Experience.metadata``
        （``metadata`` 已是 ``dict`` ⇒ **零结构改动**，`buffer.py` 的 ``extra="forbid"``
        对**顶层**字段才生效）。A-6 的"切片互补（不丢不重）"机核就靠这两个键：
        跨 rank 的并集必须是 ``{(entry, 0..G−1)}`` **恰一次**。

        ⛔ 只写元数据、不碰张量 ⇒ 与"文本/标注不出卡"无关（这是**本地**落盘）。
        """
        reward_tensor = torch.tensor(
            rewards,
            dtype=old_log_probs.dtype,
            device=old_log_probs.device,
        )
        items: list[Experience] = []
        for idx in range(len(rewards)):
            metadata = experience_metadata_for_row(generation.metadata, idx)
            if metadata is None:
                metadata = {}
            metadata = {
                **metadata,
                "entry_key": int(entry_index),
                "rollout_index": int(slot_offset + idx),
            }
            items.append(
                Experience(
                    sequences=generation.sequences[idx].detach().cpu(),
                    old_log_probs=old_log_probs[idx].detach().cpu(),
                    advantages=advantages[idx].detach().cpu(),
                    attention_mask=generation.attention_mask[idx].detach().cpu(),
                    action_mask=generation.action_mask[idx].detach().cpu(),
                    rewards=reward_tensor[idx].detach().cpu(),
                    metadata=metadata,
                )
            )
        self.replay_buffer.append_many(items)

    def _dp_rank_for_metadata(self) -> int:
        """本卡在 ``dp_group`` 内的序号（= 切片槽位区间的下标）；取不到 ⇒ 0。"""
        degree = getattr(self, "parallel_degree", None)
        rank = getattr(degree, "dp_rank", None)
        if rank is None:
            adapter = self.runtime._require_adapter()
            tp_state = getattr(adapter, "tp_state", None)
            rank = getattr(tp_state, "dp_rank", 0)
        return int(rank or 0)

    def _commit_sample_attempts(self, state: QueuedSample, *, epoch: int) -> None:
        """将样本的所有 attempts 记录到监控和 batch 缓存中。"""
        for record in state.attempts:
            self.recent_groups.append(monitor_group(record.readable))
            self.pending_batch_attempts.append(record.readable)
            self._record_epoch_attempt(record.readable)
            self._record_attempt_timing(
                epoch=epoch, retry_count=record.retry_count, timing=record.timing
            )

    def _finish_sample_and_maybe_optimize(self, *, epoch: int) -> bool:
        """样本处理完成后的收尾：递增计数器，检查是否需要优化。

        训练长度只由 training.max_epochs 控制（v0.23.0 起 max_steps 已移除）；
        提前终止仅剩 smoke 边界（--smoke 运行到首轮 optimize 后停止）。
        """
        self.current_epoch_stats.samples_seen += 1
        self.sample_index += 1
        boundary_reached = self._smoke_boundary and self.global_step >= 1
        return self._maybe_optimize(epoch=epoch) and boundary_reached

    def _record_epoch_attempt(self, payload: dict[str, Any]) -> None:
        """将单次 attempt 的指标累加到当前 epoch 统计中。

        口径：attempt_groups/completion_count 与质量均值只统计**终结 attempt**
        （decision != retry）——与 mc/nc 的 sample 终结口径对齐（v20 教训）；
        retry 中间态只进 retries 计数。
        """
        rewards = [float(value) for value in payload.get("rewards", [])]
        content_scores = [float(value) for value in payload.get("content_scores", [])]
        base_content_scores = [float(value) for value in payload.get("base_content_scores", [])]
        decision = str(payload.get("decision"))
        if decision != "retry":
            self.current_epoch_stats.attempt_groups += 1
            self.current_epoch_stats.completion_count += len(rewards)
            if rewards:
                self.current_epoch_stats.reward_mean_sum += sum(rewards) / len(rewards)
                self.current_epoch_stats.best_reward = max(
                    self.current_epoch_stats.best_reward, max(rewards)
                )
            if content_scores:
                self.current_epoch_stats.content_mean_sum += sum(content_scores) / len(
                    content_scores
                )
            if base_content_scores:
                self.current_epoch_stats.base_content_mean_sum += sum(base_content_scores) / len(
                    base_content_scores
                )
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
                    attempt_number=retry_count + 1,
                    retry_count=retry_count,
                    details=timing,
                )
            )

    # ── 错误日志汇聚 ────────────────────────────────────────────────────────────

    def _write_error_log(self, readable: dict[str, Any], reason: str) -> None:
        """写 invalid group 错误。

        JSON 事件进 ``logs/events.jsonl``（结构化错误事件流），
        error.log 只保留人类可读文本（§13.3 易读/原始分离）。
        """
        self.logger.write_event(
            {
                "event": "group_decision",
                "level": "error",
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


# ── GRASPO-Ripple 辅助函数 ──────────────────────────────────────────────────
# compute_token_rewards_for_group 已迁入 ripple/reward/token_reward.py
# （算法层，供 flow 与 ripple 共用，受 mypy 检查）。
