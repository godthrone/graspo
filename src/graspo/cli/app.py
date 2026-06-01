from __future__ import annotations

import argparse
import json
from pathlib import Path

from graspo.core.data import convert_excel_to_samples, load_json, load_jsonl, write_jsonl
from graspo.core.reward import GraspoReward, RewardConfig
from graspo.core.schema import GraspoConfig


def cmd_prepare_data(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        samples = convert_excel_to_samples(input_path)
    elif suffix == ".jsonl":
        samples = load_jsonl(input_path)
    elif suffix == ".json":
        samples = load_json(input_path)
    else:
        raise SystemExit("Only .json, .jsonl, .xlsx, and .xls are supported.")
    write_jsonl(samples, output_path)
    print(f"Wrote {len(samples)} samples to {output_path}")
    return 0


def cmd_validate_reward(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.data)[: args.limit]
    completions: list[str] = []
    if args.completions:
        with Path(args.completions).open("r", encoding="utf-8") as handle:
            for line in handle:
                completions.append(json.loads(line)["completion"])

    reward = GraspoReward(
        RewardConfig(check_think=args.check_think, check_json_markdown=args.check_json_markdown)
    )
    scores: list[float] = []
    for idx, sample in enumerate(samples):
        if idx < len(completions):
            completion = completions[idx]
        else:
            gt = sample.ground_truth if isinstance(sample.ground_truth, str) else json.dumps(sample.ground_truth)
            completion = f"```json\n{gt}\n```" if args.check_json_markdown else gt
        result = reward.score(completion, sample.ground_truth)
        scores.append(result.reward)
        print(
            json.dumps(
                {
                    "idx": idx,
                    "reward": round(result.reward, 6),
                    "content_score": round(result.content_score, 6),
                    "all_right": result.all_right,
                    "useless_len": len(result.useless_text),
                },
                ensure_ascii=False,
            )
        )
    if scores:
        print(
            json.dumps(
                {
                    "count": len(scores),
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                },
                ensure_ascii=False,
            )
        )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    groups: list[list[float]] = []
    with Path(args.rewards).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rewards = json.loads(line).get("rewards")
            if isinstance(rewards, list) and rewards:
                groups.append([float(value) for value in rewards])

    invalid = sum(1 for group in groups if max(group) == min(group))
    perfect = sum(1 for group in groups if max(group) >= 1.0)
    effective = len(groups) - invalid
    print(
        json.dumps(
            {
                "groups": len(groups),
                "invalid_rate": invalid / len(groups) if groups else 0,
                "effective_groups": effective,
                "effective_rate": effective / len(groups) if groups else 0,
                "perfect_rate": perfect / len(groups) if groups else 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = GraspoConfig.from_yaml(args.config)
    if args.backend:
        config.backend = args.backend
    from graspo.backends import create_trainer, select_backend

    selection = select_backend(config)
    if args.print_backend:
        print(selection.to_json())
        return 0
    create_trainer(config, selection).train()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graspo", description="GRASPO training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Convert JSONL/XLSX to standard JSONL.")
    prepare.add_argument("--input", "-i", required=True)
    prepare.add_argument("--output", "-o", required=True)
    prepare.set_defaults(func=cmd_prepare_data)

    validate = subparsers.add_parser("validate-reward", help="Validate reward on local data.")
    validate.add_argument("--data", "-d", required=True)
    validate.add_argument("--completions", "-c")
    validate.add_argument("--check-think", action="store_true")
    validate.add_argument("--check-json-markdown", action=argparse.BooleanOptionalAction, default=True)
    validate.add_argument("--limit", type=int, default=20)
    validate.set_defaults(func=cmd_validate_reward)

    analyze = subparsers.add_parser("analyze", help="Analyze rollout reward groups.")
    analyze.add_argument("--rewards", "-r", required=True)
    analyze.set_defaults(func=cmd_analyze)

    train = subparsers.add_parser("train", help="Start training.")
    train.add_argument("--config", "-c", required=True)
    train.add_argument("--backend", choices=["auto", "megatron-native", "hf-reference"])
    train.add_argument(
        "--print-backend",
        action="store_true",
        help="Resolve backend selection and exit without starting training.",
    )
    train.set_defaults(func=cmd_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
