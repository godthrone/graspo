#!/usr/bin/env python3
"""Send 30 training samples to vLLM, score completions with GRASPO reward."""

import base64
import json
import sys
import time
from pathlib import Path
from urllib import request

sys.path.insert(0, "/workspace/graspo/src")
from graspo.backends.graspoflow.tool_parser import parse_qwen_tool_completion
from graspo.core.reward import GraspoReward, RewardConfig

VLLM_URL = "http://localhost:18000/v1/chat/completions"
DATA = "data/train.jsonl"
NUM_COMPLETIONS = 8


def image_to_data_url(path: str) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def convert_message(msg: dict) -> dict:
    """Convert GRASPO-format message to OpenAI-vision format."""
    content = msg["content"]
    if isinstance(content, str):
        return {"role": msg["role"], "content": content}
    parts = []
    for item in content:
        if item["type"] == "image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(item["image"])},
                }
            )
        elif item["type"] == "text":
            parts.append({"type": "text", "text": item["text"]})
    return {"role": msg["role"], "content": parts}


def vllm_generate(messages: list, tools: list | None, n: int, max_tokens: int) -> list[str]:
    body = {
        "model": "elam-graspo-v1",
        "messages": [convert_message(m) for m in messages],
        "n": n,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools

    req = request.Request(
        VLLM_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = request.urlopen(req, timeout=300)
    data = json.loads(resp.read())

    completions = []
    for choice in data.get("choices", []):
        msg = choice["message"]
        content = msg.get("content") or ""
        tc_list = msg.get("tool_calls") or []
        if tc_list:
            # Convert vLLM tool_calls to Qwen XML format that GRASPO parser expects
            lines = []
            for tc in tc_list:
                func = tc["function"]
                args_str = func["arguments"]
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                lines.append("<tool_call>")
                lines.append(f"<function={func['name']}>")
                for k, v in args.items():
                    lines.append(f"<parameter={k}>{v}</parameter>")
                lines.append("</function>")
                lines.append("</tool_call>")
            completions.append("\n".join(lines))
        else:
            completions.append(content)
    return completions


def main():
    # Load data
    samples = []
    with open(DATA) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples")

    scorer = GraspoReward(
        RewardConfig(
            check_think=False,
            check_json_markdown=False,
            check_tool_call=True,
            check_list_order=False,
            marker_reward_weight=10,
            content_reward_weight=100,
        )
    )

    all_rewards = []
    all_right_count = 0
    perf_samples = 0

    for si, sample in enumerate(samples):
        messages = sample["messages"]
        tools = sample.get("tools")
        targets = sample["targets"]

        t0 = time.time()
        completions = vllm_generate(messages, tools, NUM_COMPLETIONS, 512)
        elapsed = time.time() - t0

        sample_rewards = []
        for comp in completions:
            parsed = parse_qwen_tool_completion(comp, tools=tools)
            result = scorer.score_parsed(parsed, targets, is_tool_call=True)
            sample_rewards.append(result.reward)
            if result.all_right:
                all_right_count += 1

        all_rewards.extend(sample_rewards)
        r_mean = sum(sample_rewards) / len(sample_rewards)
        r_max = max(sample_rewards)

        has_perfect = r_max >= 1.0
        if has_perfect:
            perf_samples += 1

        marker = " ★" if has_perfect else ""
        print(
            f"sample {si:2d}  reward={r_mean:.4f}  max={r_max:.4f}  "
            f"time={elapsed:.1f}s  n={len(completions)}{marker}"
        )

    print()
    print("=" * 60)
    print(f"Total: {len(all_rewards)} completions, {len(samples)} samples")
    print(f"reward mean:  {sum(all_rewards) / len(all_rewards):.4f}")
    print(f"reward min:   {min(all_rewards):.4f}")
    print(f"reward max:   {max(all_rewards):.4f}")
    print(f"all_right:    {all_right_count}/{len(all_rewards)}")
    print(f"perfect samples (max>=1.0): {perf_samples}/{len(samples)}")


if __name__ == "__main__":
    main()
