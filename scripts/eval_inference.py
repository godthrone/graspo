#!/usr/bin/env python3
"""Load merged GRASPO model, run inference on training data, score with reward."""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Add GRASPO source to path (mounted in Docker at /workspace/graspo/src)
sys.path.insert(0, "/workspace/graspo/src")
from graspo.core.reward import GraspoReward, RewardConfig
from graspo.core.completion import raw_parsed_completion


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/data/outputs/merged_model")
    data_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/data/data/train_docker.jsonl")
    num_completions = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_new_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512

    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Completions per sample: {num_completions}")
    print(f"Max new tokens: {max_new_tokens}")
    print()

    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    print(f"Model loaded on {device}")

    # Reward scorer
    reward_cfg = RewardConfig(
        check_think=False,
        check_json_markdown=False,
        check_tool_call=True,
        check_list_order=False,
        marker_reward_weight=10,
        content_reward_weight=100,
    )
    scorer = GraspoReward(reward_cfg)

    # Load data
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples\n")

    all_rewards = []
    all_contents = []
    all_right_count = 0
    total_comps = 0

    for si, sample in enumerate(samples):
        messages = sample["messages"]
        tools = sample.get("tools")
        targets = sample["targets"]

        # Build prompt via processor chat template
        apply_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        if tools:
            apply_kwargs["tools"] = tools

        inputs = processor.apply_chat_template(
            messages,
            **apply_kwargs,
        )
        # Handle different return types: dict or single tensor
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
            inputs = {"input_ids": inputs}

        prompt_len = inputs["input_ids"].shape[1]

        # Generate multiple completions
        sample_rewards = []
        sample_contents = []
        for _ in range(num_completions):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id,
                )
            # Extract only the generated part
            generated_ids = outputs[0][prompt_len:]
            completion = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Parse and score
            parsed = raw_parsed_completion(completion)
            result = scorer.score_parsed(parsed, targets, is_tool_call=True)

            sample_rewards.append(result.reward)
            sample_contents.append(result.content_score)
            if result.all_right:
                all_right_count += 1

        total_comps += num_completions
        all_rewards.extend(sample_rewards)
        all_contents.extend(sample_contents)

        r_mean = sum(sample_rewards) / len(sample_rewards)
        r_max = max(sample_rewards)
        sample_all_right = sum(1 for r in sample_rewards if r >= 1.0)

        print(f"sample {si:2d}  reward_mean={r_mean:.4f}  max={r_max:.4f}  "
              f"content_mean={sum(sample_contents)/len(sample_contents):.4f}  "
              f"all_right={sample_all_right}/{num_completions}",
              end="")
        if r_max >= 1.0:
            print(" ★", end="")
        print()

    # Summary
    print()
    print("=" * 60)
    print(f"Total: {total_comps} completions across {len(samples)} samples")
    print(f"reward mean:  {sum(all_rewards)/len(all_rewards):.4f}")
    print(f"reward range: {min(all_rewards):.4f} - {max(all_rewards):.4f}")
    print(f"content mean: {sum(all_contents)/len(all_contents):.4f}")
    print(f"all_right count: {all_right_count}/{total_comps}")
    print(f"perfect samples (reward>=1): {sum(1 for r in all_rewards if r >= 1.0)}")


if __name__ == "__main__":
    main()
