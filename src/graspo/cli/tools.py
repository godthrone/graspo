"""CLI 工具命令实现：validate-reward / evaluate-checkpoint / analyze-profile。

工具命令不参与训练产物（配置驱动命令的"三选一"约束）：
- validate-reward：只 print 不落盘（输入定位 --data/--limit/--completions）
- evaluate-checkpoint：接受 --config，评测输出写入 config 决定的
  ``<output_dir>/evaluate/``（禁止输出定位参数）
- analyze-profile：只 print 不落盘（输入定位 run_dirs）
"""

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from graspo.core.data import load_jsonl
from graspo.core.schema import GraspoConfig, Sample
from graspo.flow.runtime import GraspoFlowRuntime
from graspo.ripple.parsing.completion import ParsedCompletion, raw_parsed_completion
from graspo.ripple.reward.reward import GraspoReward, RewardConfig

# ── validate-reward ──────────────────────────────────────────────────────────


def validate_reward_scores(
    samples: list[Sample],
    completions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """对每个样本评分并返回结果摘要（不落盘，由调用方决定如何输出）。

    - 提供显式 completion 时按该 completion 评分；
    - 否则用 ground truth 构造理想 completion（工具调用取 tool_calls，纯文本取 content）。
    """
    reward = GraspoReward(RewardConfig())
    completions = completions or []
    scores: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        has_explicit_completion = idx < len(completions)
        if has_explicit_completion:
            completion = completions[idx]
        else:
            output = sample.targets[0]["output"]
            if sample.expects_tool_calls:
                calls = output.get("tool_calls") or []
                completion = json.dumps(calls, ensure_ascii=False)
            else:
                content = output.get("content") or {}
                # 默认配置 check_json_markdown=True，理想 completion 需带 ```json 围栏
                completion = f"```json\n{json.dumps(content, ensure_ascii=False)}\n```"
        if sample.expects_tool_calls:
            parsed = (
                raw_parsed_completion(completion)
                if has_explicit_completion
                else ParsedCompletion(
                    raw_text=completion,
                    tool_calls=list(sample.targets[0]["output"].get("tool_calls") or []),
                    parser_name="validate_reward_canonical",
                )
            )
            result = reward.score_parsed(parsed, sample.targets, is_tool_call=True)
        else:
            result = reward.score(completion, sample.targets)
        scores.append(
            {
                "idx": idx,
                "reward": round(result.reward, 6),
                "content_score": round(result.content_score, 6),
                "all_right": result.all_right,
                "useless_len": len(result.useless_text),
            }
        )
    return scores


# ── evaluate-checkpoint ──────────────────────────────────────────────────────


def evaluate_samples(
    runtime: GraspoFlowRuntime,
    config: GraspoConfig,
    samples: list[Sample],
    output_dir: Path,
    *,
    checkpoint: str | None,
) -> dict[str, Any]:
    """用 runtime 对样本生成 rollout groups 并评分，落盘 completions 与汇总。

    output_dir 由调用方（config 决定的评测目录）传入；本函数只负责写入。
    """
    reward = GraspoReward(config.reward)
    output_dir.mkdir(parents=True, exist_ok=True)
    completions_path = output_dir / "completions.jsonl"
    rewards: list[float] = []
    reward_max_values: list[float] = []
    reward_range_values: list[float] = []
    content_values: list[float] = []
    all_right_count = 0
    group_count = 0

    primary = runtime.is_primary()
    handle = completions_path.open("w", encoding="utf-8") if primary else None
    try:
        for index, sample in enumerate(samples):
            if sample.media:
                generation = runtime.generate_sample_groups(
                    samples=[sample],
                    rollout_group_size=config.training.rollout_group_size,
                    max_new_tokens=config.training.max_new_tokens,
                    max_prompt_length=config.data.max_prompt_length,
                    temperature=config.training.temperature,
                    top_p=config.training.top_p,
                    chat_template_kwargs=config.model.chat_template_kwargs,
                )[0]
            else:
                generation = runtime.generate_groups(
                    message_batches=[sample.messages],
                    rollout_group_size=config.training.rollout_group_size,
                    max_new_tokens=config.training.max_new_tokens,
                    max_prompt_length=config.data.max_prompt_length,
                    temperature=config.training.temperature,
                    top_p=config.training.top_p,
                    chat_template_kwargs=config.model.chat_template_kwargs,
                )[0]
            parsed_completions = [
                _parse_completion(runtime, completion, sample)
                for completion in generation.completions
            ]
            results = [
                reward.score_parsed(
                    parsed,
                    sample.targets,
                    is_tool_call=sample.expects_tool_calls,
                )
                for parsed in parsed_completions
            ]
            group_rewards = [result.reward for result in results]
            if primary and handle is not None:
                for completion_idx, (completion, result, parsed) in enumerate(
                    zip(generation.completions, results, parsed_completions, strict=True)
                ):
                    handle.write(
                        json.dumps(
                            {
                                "sample_index": index,
                                "completion_index": completion_idx,
                                "reward": result.reward,
                                "content_score": result.content_score,
                                "all_right": result.all_right,
                                "parsed_tool_calls": parsed.tool_calls,
                                "parser_name": parsed.parser_name,
                                "parser_errors": parsed.parse_errors,
                                "matched_target_index": result.matched_target_index,
                                "matched_target_id": result.matched_target_id,
                                "target_scores": result.target_scores,
                                "completion": completion,
                                "targets": sample.targets,
                                "metadata": _safe_metadata(sample),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if not group_rewards:
                continue
            group_count += 1
            rewards.append(sum(group_rewards) / len(group_rewards))
            reward_max_values.append(max(group_rewards))
            reward_range_values.append(max(group_rewards) - min(group_rewards))
            content_values.append(sum(result.content_score for result in results) / len(results))
            all_right_count += sum(1 for result in results if result.all_right)
    finally:
        if handle is not None:
            handle.close()

    completion_count = group_count * int(config.training.rollout_group_size)
    return {
        "count": group_count,
        "completion_count": completion_count,
        "reward_mean": _mean(rewards),
        "reward_max_mean": _mean(reward_max_values),
        "reward_range_mean": _mean(reward_range_values),
        "content_mean": _mean(content_values),
        "all_right_rate": all_right_count / completion_count if completion_count else 0.0,
        "max_new_tokens": config.training.max_new_tokens,
        "rollout_group_size": config.training.rollout_group_size,
        "checkpoint": checkpoint,
        "data": str(config.data.train_path),
    }


def run_evaluate(
    config_path: str,
    data_path: str,
    *,
    checkpoint: str | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    """evaluate-checkpoint 命令核心：加载 config、运行评测、输出到 config 决定的目录。"""
    config = GraspoConfig.from_yaml(config_path)
    samples = load_jsonl(data_path)
    if limit and limit > 0:
        samples = samples[:limit]

    # 评测输出位置由 config 决定（禁止输出定位参数）
    output_dir = Path(config.training.output_dir) / "evaluate"
    runtime = GraspoFlowRuntime.from_config(config)
    started_at = time.monotonic()
    try:
        runtime.setup()
        if checkpoint:
            runtime.load_checkpoint(checkpoint)
        summary = evaluate_samples(
            runtime, config, samples, output_dir, checkpoint=checkpoint
        )
    finally:
        runtime.close()

    if runtime.is_primary():
        summary["elapsed_sec"] = time.monotonic() - started_at
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_metadata(sample: Sample) -> dict[str, Any]:
    metadata = dict(sample.metadata)
    media = []
    for item in sample.media:
        media.append({key: value for key, value in item.items() if key != "path"})
    if media:
        metadata["media"] = media
    return metadata


def _parse_completion(runtime: GraspoFlowRuntime, completion: str, sample: Sample):
    parse_completion = getattr(runtime, "parse_completion", None)
    if callable(parse_completion):
        return parse_completion(completion, sample)
    return raw_parsed_completion(completion)


# ── analyze-profile ──────────────────────────────────────────────────────────


TIMING_KEYS = (
    "total_observed_sec",
    "rollout_sec",
    "prefill_sec",
    "decode_sec",
    "sampling_sec",
    "stop_check_sec",
    "old_logprob_sec",
    "optimize_sec",
    "train_batch_total_sec",
    "micro_batch_forward_sec",
    "backward_sec",
    "optimizer_step_sec",
    "checkpoint_sec",
    "decode_tokens",
    "rollout_generation_split_count",
)


def summarize_run(run_dir: Path, *, skip_warmup_steps: int = 1) -> dict[str, Any]:
    """汇总一次运行目录的 train_step 事件、GPU 与 rank 指标（只读不写）。"""
    train_steps = _read_train_steps(run_dir)
    measured = (
        train_steps[skip_warmup_steps:] if len(train_steps) > skip_warmup_steps else train_steps
    )
    latest = train_steps[-1] if train_steps else {}
    timing_rows = [dict(step.get("timing") or {}) for step in measured]
    gpu_summary = _read_gpu_summary(run_dir)
    rank_summary = _read_rank_summary(run_dir)
    decisions = latest.get("batch", {}).get("decisions", {}) if latest else {}
    latest_timing = latest.get("timing", {}) if latest else {}
    latest_reward = latest.get("batch", {}).get("reward_mean")
    if latest_reward is None:
        latest_reward = latest.get("epoch", {}).get("reward_mean")
    latest_content = latest.get("batch", {}).get("content_mean")
    if latest_content is None:
        latest_content = latest.get("epoch", {}).get("content_mean")
    total_sec = _mean_key(timing_rows, "total_observed_sec")
    decode_tokens = _sum_key(timing_rows, "decode_tokens")
    rollout_sec = _sum_key(timing_rows, "rollout_sec")
    trainable_groups = _sum_latest_or_batch(
        train_steps, measured, ("batch", "decisions", "trainable", "total")
    )
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "step_count": len(train_steps),
        "measured_step_count": len(measured),
        "latest_step": latest.get("run", {}).get("step")
        or latest.get("run", {}).get("optimized_steps"),
        "latest_epoch": latest.get("epoch", {}).get("index")
        or latest.get("epoch", {}).get("epoch"),
        "latest_reward_mean": latest_reward,
        "latest_content_mean": latest_content,
        "latest_trainable_groups": decisions.get("trainable", {}).get("total"),
        "latest_invalid": decisions.get("terminal", {}).get("invalid"),
        "latest_invalid_no_preference_gap": decisions.get("terminal", {}).get(
            "invalid_no_preference_gap"
        ),
        "timing_mean": {key: _mean_key(timing_rows, key) for key in TIMING_KEYS},
        "latest_timing": {
            key: latest_timing.get(key) for key in TIMING_KEYS if key in latest_timing
        },
        "decode_tokens_per_sec": decode_tokens / rollout_sec if rollout_sec > 0 else None,
        "trainable_groups_per_hour": trainable_groups * 3600.0 / (total_sec * len(measured))
        if total_sec and measured
        else None,
        "gpu": gpu_summary,
        "rank": rank_summary,
    }


def run_analyze(run_dirs: list[str], *, skip_warmup_steps: int = 1, as_json: bool = False) -> None:
    """analyze-profile 命令核心：汇总多个运行目录并打印（不落盘）。"""
    summaries = [
        summarize_run(Path(path), skip_warmup_steps=skip_warmup_steps) for path in run_dirs
    ]
    if as_json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        print_table(summaries)


def print_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "run",
        "steps",
        "reward",
        "total_s",
        "rollout_s",
        "opt_s",
        "decode_tok_s",
        "groups_h",
        "gpu_util",
        "gpu_peak_gib",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(item) for item in headers))
    for summary in summaries:
        timing = summary.get("timing_mean") or {}
        gpu_util, gpu_peak_gib = _gpu_rollup(summary.get("gpu") or {})
        values = [
            summary.get("name"),
            f"{summary.get('measured_step_count')}/{summary.get('step_count')}",
            _fmt(summary.get("latest_reward_mean")),
            _fmt(timing.get("total_observed_sec")),
            _fmt(timing.get("rollout_sec")),
            _fmt(timing.get("optimize_sec")),
            _fmt(summary.get("decode_tokens_per_sec")),
            _fmt(summary.get("trainable_groups_per_hour")),
            _fmt(gpu_util),
            _fmt(gpu_peak_gib),
        ]
        print(" | ".join(str(item) for item in values))


def _read_train_steps(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in (run_dir / "nohup.out", run_dir / "logs" / "train.log", run_dir / "train.log"):
        if not path.exists():
            continue
        for payload in _iter_json_lines(path):
            if payload.get("event") == "train_step":
                events.append(payload)
    by_step: dict[Any, dict[str, Any]] = {}
    for event in events:
        step = event.get("run", {}).get("step")
        by_step[step if step is not None else len(by_step)] = event
    return list(by_step.values())


def _read_gpu_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "gpu_memory" / "gpu_memory_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return _compact_gpu_summary(summary)
    memory_path = run_dir / "gpu_memory" / "gpu_memory.jsonl"
    if not memory_path.exists():
        return {}
    rows = list(_iter_json_lines(memory_path))
    by_gpu: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gpu.setdefault(str(row.get("gpu_index")), []).append(row)
    per_gpu = {}
    for gpu, gpu_rows in by_gpu.items():
        memory = [float(row.get("memory_used_mib") or 0.0) for row in gpu_rows]
        util = [float(row.get("utilization_gpu_pct") or 0.0) for row in gpu_rows]
        per_gpu[gpu] = {
            "samples": len(gpu_rows),
            "memory_used_mib_peak": max(memory) if memory else None,
            "memory_used_mib_mean": mean(memory) if memory else None,
            "utilization_gpu_pct_mean": mean(util) if util else None,
        }
    return {"per_gpu": per_gpu}


def _compact_gpu_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"sample_count": summary.get("sample_count"), "per_gpu": {}}
    for gpu, values in (summary.get("per_gpu") or {}).items():
        compact["per_gpu"][str(gpu)] = {
            "samples": values.get("samples"),
            "memory_used_mib_peak": values.get("memory_used_mib_peak"),
            "memory_used_mib_p95": values.get("memory_used_mib_p95"),
            "memory_used_mib_mean": values.get("memory_used_mib_mean"),
            "utilization_gpu_pct_mean": values.get("utilization_gpu_pct_mean"),
        }
    return compact


def _read_rank_summary(run_dir: Path) -> dict[str, Any]:
    latest_by_rank: dict[int, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("rank_metrics.rank_*.jsonl")):
        for payload in _iter_json_lines(path):
            if payload.get("phase") != "pipeline_train_batch_after":
                continue
            metrics = payload.get("metrics") or {}
            for item in metrics.get("rank_metrics") or [metrics]:
                if "rank" in item:
                    latest_by_rank[int(item["rank"])] = item
    per_rank = {}
    for rank, metrics in sorted(latest_by_rank.items()):
        stage_timing = metrics.get("pipeline_stage_timing") or {}
        per_rank[str(rank)] = {
            "placement_strategy": metrics.get("placement_strategy"),
            "pipeline_train_schedule": metrics.get("pipeline_train_schedule"),
            "pipeline_stage_rank": metrics.get("pipeline_stage_rank"),
            "pipeline_stage_compute_sec": stage_timing.get("pipeline_stage_compute_sec"),
            "pipeline_backward_autograd_sec": stage_timing.get("pipeline_backward_autograd_sec"),
            "pipeline_send_sec": stage_timing.get("pipeline_send_sec"),
            "pipeline_recv_sec": stage_timing.get("pipeline_recv_sec"),
            "pipeline_grad_send_sec": stage_timing.get("pipeline_grad_send_sec"),
            "pipeline_grad_recv_sec": stage_timing.get("pipeline_grad_recv_sec"),
            "pipeline_norm_sec": stage_timing.get("pipeline_norm_sec"),
            "pipeline_lm_head_sec": stage_timing.get("pipeline_lm_head_sec"),
            "pipeline_loss_sec": stage_timing.get("pipeline_loss_sec"),
        }
    return {"per_rank": per_rank}


def _iter_json_lines(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return payloads


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _sum_key(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows if row.get(key) is not None)


def _sum_latest_or_batch(
    train_steps: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    path: tuple[str, ...],
) -> float:
    latest = train_steps[-1] if train_steps else {}
    value: Any = latest
    for key in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    total = 0.0
    for step in measured:
        item: Any = dict(step) if isinstance(step, dict) else {}
        for key in path:
            if not isinstance(item, dict):
                item = None
                break
            item = item.get(key)
        if isinstance(item, (int, float)):
            total += float(item)
    return total


def _gpu_rollup(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    util = summary.get("utilization_percent")
    peak = summary.get("max_mib")
    if util is None or peak is None:
        return None, None
    return float(util), float(peak)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
