from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graspo.core.data import convert_excel_to_samples, load_json, load_jsonl, write_jsonl
from graspo.core.reward import GraspoReward, RewardConfig
from graspo.core.schema import GraspoConfig


def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    from graspo.trainer import FSDPGraspoTrainer

    FSDPGraspoTrainer(config).train()
    return 0


def cmd_anchor_generate(args: argparse.Namespace) -> int:
    from graspo.anchor import AnchorGenerationConfig, generate_anchor_prompts, load_ontology
    from graspo.anchor.bank import write_jsonl as write_anchor_jsonl

    config_data = _read_yaml(args.config)
    ontology_cfg = config_data.get("ontology", {})
    generation_cfg = config_data.get("generation", {})

    knowledge_path = args.knowledge_ontology or ontology_cfg.get("knowledge_path")
    language_path = args.language_ontology or ontology_cfg.get("language_path")
    output_path = args.output or generation_cfg.get("output_path")
    if not knowledge_path or not language_path or not output_path:
        raise SystemExit("--knowledge-ontology, --language-ontology, and --output are required")

    languages = _csv(args.languages) or generation_cfg.get("languages")
    task_types = _csv(args.task_types) or generation_cfg.get("task_types")
    prompt_config = AnchorGenerationConfig(
        count=args.count if args.count is not None else int(generation_cfg.get("count", 100)),
        seed=args.seed if args.seed is not None else int(generation_cfg.get("seed", 42)),
        languages=list(languages or ["English", "简体中文"]),
        task_types=list(task_types or []),
        language_features_per_prompt=(
            args.language_features_per_prompt
            if args.language_features_per_prompt is not None
            else int(generation_cfg.get("language_features_per_prompt", 2))
        ),
    )
    if not prompt_config.task_types:
        from graspo.anchor.sampler import DEFAULT_TASK_TYPES

        prompt_config.task_types = list(DEFAULT_TASK_TYPES)

    knowledge = load_ontology(knowledge_path, root_key=args.knowledge_root or ontology_cfg.get("knowledge_root"))
    language = load_ontology(language_path, root_key=args.language_root or ontology_cfg.get("language_root"))
    prompts = generate_anchor_prompts(knowledge, language, prompt_config)
    write_anchor_jsonl(prompts, output_path)
    print(f"Wrote {len(prompts)} anchor prompts to {output_path}")
    return 0


def cmd_anchor_answer(args: argparse.Namespace) -> int:
    from graspo.anchor.teacher import answer_anchor_prompts

    answered = answer_anchor_prompts(
        model_path=args.model_path,
        input_path=args.input,
        output_path=args.output,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
    )
    print(f"Wrote {len(answered)} answered anchors to {args.output}")
    return 0


def cmd_anchor_filter(args: argparse.Namespace) -> int:
    from graspo.anchor.bank import filter_answered_anchors, read_answered_anchors
    from graspo.anchor.bank import write_jsonl as write_anchor_jsonl
    from graspo.anchor.manifest import build_anchor_manifest, write_manifest

    anchors = read_answered_anchors(args.input)
    kept, stats = filter_answered_anchors(
        anchors,
        min_answer_chars=args.min_answer_chars,
        max_answer_chars=args.max_answer_chars,
    )
    write_anchor_jsonl(kept, args.output)
    stats_payload = stats.to_dict()
    print(json.dumps(stats_payload, ensure_ascii=False))
    if args.stats_output:
        _write_json(args.stats_output, stats_payload)
    if args.manifest_output:
        manifest = build_anchor_manifest(
            teacher_model=args.teacher_model or (kept[0].teacher_model if kept else ""),
            generation_config={},
            anchors=kept,
            filter_stats=stats,
            seed=args.seed,
        )
        write_manifest(manifest, args.manifest_output)
    return 0


def cmd_anchor_split(args: argparse.Namespace) -> int:
    from graspo.anchor.bank import read_answered_anchors, split_answered_anchors
    from graspo.anchor.bank import write_jsonl as write_anchor_jsonl

    anchors = read_answered_anchors(args.input)
    train, eval_items = split_answered_anchors(anchors, eval_ratio=args.eval_ratio, seed=args.seed)
    write_anchor_jsonl(train, args.train_output)
    write_anchor_jsonl(eval_items, args.eval_output)
    print(
        json.dumps(
            {"input": len(anchors), "train": len(train), "eval": len(eval_items)},
            ensure_ascii=False,
        )
    )
    return 0


def cmd_train_sft_ard(args: argparse.Namespace) -> int:
    from graspo.sft import ARDSFTTrainer
    from graspo.sft.schema import ARDSFTConfig

    config = ARDSFTConfig.from_yaml(args.config)
    ARDSFTTrainer(config).train()
    return 0


def cmd_eval_forgetting(args: argparse.Namespace) -> int:
    anchors = _read_jsonl_records(args.anchor_eval)
    completions = {str(item.get("id", idx)): item for idx, item in enumerate(_read_jsonl_records(args.completions))}
    exact = 0
    contains = 0
    missing = 0
    scored = 0
    length_ratios: list[float] = []

    for idx, anchor in enumerate(anchors):
        key = str(anchor.get("id", idx))
        completion_record = completions.get(key)
        if completion_record is None:
            missing += 1
            continue
        teacher = str(anchor.get("teacher_answer", "")).strip()
        completion = str(
            completion_record.get(
                "completion",
                completion_record.get("response", completion_record.get("output", "")),
            )
        ).strip()
        if not teacher or not completion:
            continue
        scored += 1
        norm_teacher = " ".join(teacher.lower().split())
        norm_completion = " ".join(completion.lower().split())
        exact += int(norm_teacher == norm_completion)
        contains += int(norm_teacher in norm_completion or norm_completion in norm_teacher)
        length_ratios.append(len(completion) / max(len(teacher), 1))

    payload = {
        "anchors": len(anchors),
        "scored": scored,
        "missing": missing,
        "exact_match": exact / scored if scored else 0,
        "contains_match": contains / scored if scored else 0,
        "mean_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else 0,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if args.output:
        _write_json(args.output, payload)
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
    train.set_defaults(func=cmd_train)

    anchor_generate = subparsers.add_parser("anchor-generate", help="Generate offline anchor prompts.")
    anchor_generate.add_argument("--config")
    anchor_generate.add_argument("--knowledge-ontology")
    anchor_generate.add_argument("--language-ontology")
    anchor_generate.add_argument("--knowledge-root")
    anchor_generate.add_argument("--language-root")
    anchor_generate.add_argument("--output", "-o")
    anchor_generate.add_argument("--count", type=int)
    anchor_generate.add_argument("--seed", type=int)
    anchor_generate.add_argument("--languages")
    anchor_generate.add_argument("--task-types")
    anchor_generate.add_argument("--language-features-per-prompt", type=int)
    anchor_generate.set_defaults(func=cmd_anchor_generate)

    anchor_answer = subparsers.add_parser("anchor-answer", help="Answer anchor prompts with a local base model.")
    anchor_answer.add_argument("--model-path", required=True)
    anchor_answer.add_argument("--input", "-i", required=True)
    anchor_answer.add_argument("--output", "-o", required=True)
    anchor_answer.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    anchor_answer.add_argument("--torch-dtype", default="bfloat16")
    anchor_answer.add_argument("--max-new-tokens", type=int, default=512)
    anchor_answer.add_argument("--max-prompt-length", type=int, default=2048)
    anchor_answer.add_argument("--temperature", type=float, default=0.7)
    anchor_answer.add_argument("--top-p", type=float, default=0.95)
    anchor_answer.add_argument("--limit", type=int)
    anchor_answer.set_defaults(func=cmd_anchor_answer)

    anchor_filter = subparsers.add_parser("anchor-filter", help="Filter answered anchors.")
    anchor_filter.add_argument("--input", "-i", required=True)
    anchor_filter.add_argument("--output", "-o", required=True)
    anchor_filter.add_argument("--min-answer-chars", type=int, default=8)
    anchor_filter.add_argument("--max-answer-chars", type=int, default=4096)
    anchor_filter.add_argument("--stats-output")
    anchor_filter.add_argument("--manifest-output")
    anchor_filter.add_argument("--teacher-model")
    anchor_filter.add_argument("--seed", type=int)
    anchor_filter.set_defaults(func=cmd_anchor_filter)

    anchor_split = subparsers.add_parser("anchor-split", help="Split filtered anchors into train/eval JSONL.")
    anchor_split.add_argument("--input", "-i", required=True)
    anchor_split.add_argument("--train-output", required=True)
    anchor_split.add_argument("--eval-output", required=True)
    anchor_split.add_argument("--eval-ratio", type=float, default=0.1)
    anchor_split.add_argument("--seed", type=int, default=42)
    anchor_split.set_defaults(func=cmd_anchor_split)

    train_sft = subparsers.add_parser("train-sft-ard", help="Run hard-sample SFT with anchor replay distillation.")
    train_sft.add_argument("--config", "-c", required=True)
    train_sft.set_defaults(func=cmd_train_sft_ard)

    eval_forgetting = subparsers.add_parser("eval-forgetting", help="Compare anchor eval completions to teacher answers.")
    eval_forgetting.add_argument("--anchor-eval", required=True)
    eval_forgetting.add_argument("--completions", required=True)
    eval_forgetting.add_argument("--output")
    eval_forgetting.set_defaults(func=cmd_eval_forgetting)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
