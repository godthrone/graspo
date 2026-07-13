#!/usr/bin/env python3
"""Convert multi-target training data to AND-mode single-target format.

Input: targets with multiple entries, each containing 1 tool_call
Output: single target containing all tool_calls (AND mode)

Also updates the system prompt from "每轮只输出一个" to "每轮输出所有需要的调整动作".
"""

import json
import sys
from collections import Counter
from pathlib import Path

NEW_SYSTEM_TEXT = (
    "你是一个人形机器人，左右眼看到当前桌面，右手是 SO100 机械臂。\n"
    "图像顺序：第一张是 left_eye，第二张是 right_eye。\n"
    "你的右臂从底座伸出，可以旋转、伸缩、升降。\n"
    "俯视方向：逆时针旋转 = 夹爪向左转, 顺时针旋转 = 夹爪向右转。\n"
    "每轮输出所有需要的调整动作（可能 1-3 个），每个动作一个 robot_atomic_control 工具调用。"
)


def convert_sample(sample: dict) -> dict | None:
    """Convert a single sample from OR to AND mode."""
    targets = sample.get("targets", [])
    if not targets:
        return None

    # Merge all tool_calls into one target
    merged_tool_calls = []
    for t in targets:
        output = t.get("output", {})
        calls = output.get("tool_calls", [])
        merged_tool_calls.extend(calls)

    if not merged_tool_calls:
        return None

    new_targets = [
        {
            "id": "primary",
            "output": {"tool_calls": merged_tool_calls},
        }
    ]

    # Update system prompt
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", [])
            for item in content:
                if item.get("type") == "text":
                    item["text"] = NEW_SYSTEM_TEXT

    sample["targets"] = new_targets
    return sample


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.jsonl> [output.jsonl]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".and.jsonl")

    stats = Counter()
    converted = 0
    skipped = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            result = convert_sample(sample)
            if result is None:
                skipped += 1
                continue
            n_tc = len(result["targets"][0]["output"]["tool_calls"])
            stats[n_tc] += 1
            converted += 1
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Converted: {converted}, Skipped: {skipped}")
    print()
    print("Tool calls per target:")
    for n in sorted(stats):
        print(f"  {n} tc: {stats[n]} ({stats[n]/converted*100:.1f}%)")


if __name__ == "__main__":
    main()