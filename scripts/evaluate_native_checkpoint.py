#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from graspo.backends.native_tp.runtime import NativeTPRuntime
from graspo.core.data import load_jsonl
from graspo.core.reward import GraspoReward
from graspo.core.schema import GraspoConfig, Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a native-tp checkpoint by generating rollout groups and scoring rewards."
    )
    parser.add_argument("--config", required=True, help="Training YAML config used to build the native runtime.")
    parser.add_argument("--data", required=True, help="Evaluation JSONL path.")
    parser.add_argument("--checkpoint", help="Recoverable native checkpoint directory to load.")
    parser.add_argument("--output-dir", required=True, help="Directory for completions and summary JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of samples to evaluate; 0 means all.")
    parser.add_argument("--rollout-group-size", type=int, help="Override training.rollout_group_size.")
    parser.add_argument("--temperature", type=float, help="Override training.temperature.")
    parser.add_argument("--top-p", type=float, help="Override training.top_p.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GraspoConfig.from_yaml(args.config)
    config.data.train_path = args.data
    config.training.output_dir = args.output_dir
    if args.rollout_group_size is not None:
        config.training.rollout_group_size = args.rollout_group_size
    if args.temperature is not None:
        config.training.temperature = args.temperature
    if args.top_p is not None:
        config.training.top_p = args.top_p

    samples = load_jsonl(
        args.data,
        prompt_field=config.data.prompt_field,
        ground_truth_field=config.data.ground_truth_field,
        messages_field=config.data.messages_field,
    )
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    output_dir = Path(args.output_dir)
    runtime = NativeTPRuntime.from_config(config)
    started_at = time.monotonic()
    try:
        runtime.setup()
        if args.checkpoint:
            runtime.load_checkpoint(args.checkpoint)
        summary = evaluate_samples(runtime, config, samples, output_dir, checkpoint=args.checkpoint)
    finally:
        runtime.close()

    if runtime.is_primary():
        summary["elapsed_sec"] = time.monotonic() - started_at
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
    return 0


def evaluate_samples(
    runtime: NativeTPRuntime,
    config: GraspoConfig,
    samples: list[Sample],
    output_dir: Path,
    *,
    checkpoint: str | None,
) -> dict[str, Any]:
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
            generation = runtime.generate_sample_groups(
                samples=[sample],
                rollout_group_size=config.training.rollout_group_size,
                max_new_tokens=config.training.max_new_tokens,
                max_prompt_length=config.data.max_prompt_length,
                temperature=config.training.temperature,
                top_p=config.training.top_p,
                chat_template_kwargs=config.model.chat_template_kwargs,
            )[0]
            results = [reward.score(completion, sample.ground_truth) for completion in generation.completions]
            group_rewards = [result.reward for result in results]
            if primary and handle is not None:
                for completion_idx, (completion, result) in enumerate(zip(generation.completions, results, strict=True)):
                    handle.write(
                        json.dumps(
                            {
                                "sample_index": index,
                                "completion_index": completion_idx,
                                "reward": result.reward,
                                "content_score": result.content_score,
                                "all_right": result.all_right,
                                "completion": completion,
                                "ground_truth": sample.ground_truth,
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


if __name__ == "__main__":
    raise SystemExit(main())
