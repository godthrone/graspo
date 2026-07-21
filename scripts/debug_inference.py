#!/usr/bin/env python3
"""统一 debug 推理脚本 —— 加载 HF 模型，对 JSONL 数据推理，打印原始输出和期望输出。

用途：
- 排查 SFT 训练后模型输出格式是否与训练数据对齐
- 对比 base model vs. fine-tuned model 的输出差异
- 快速验证模型推理质量

用法:
    # 对合并后的 SFT 模型推理 3 条数据
    python scripts/debug_inference.py \\
        --model /data/outputs/sft_elam_v3_merged \\
        --data /data/.../train.jsonl \\
        --limit 3

    # 同时对比 base model
    python scripts/debug_inference.py \\
        --model /data/outputs/sft_elam_v3_merged \\
        --baseline /data/zhangzy/models/Qwen3.5-9B \\
        --data /data/.../train.jsonl \\
        --limit 5

    # 只打印原始输出，不评分
    python scripts/debug_inference.py \\
        --model /data/outputs/sft_elam_v3_merged \\
        --data /data/.../train.jsonl \\
        --raw-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一 debug 推理脚本")
    p.add_argument("--model", required=True, help="HF 模型路径（合并后的 SFT 模型或 base model）")
    p.add_argument(
        "--baseline", default=None, help="可选的 baseline 模型路径，同时推理并对比输出"
    )
    p.add_argument("--data", required=True, help="JSONL 数据文件路径")
    p.add_argument("--limit", type=int, default=3, help="推理样本数（默认 3）")
    p.add_argument("--start", type=int, default=0, help="从第几条数据开始（默认 0）")
    p.add_argument("--max-new-tokens", type=int, default=256, help="最大生成 token 数")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--raw-only", action="store_true", help="只打印原始输出，不评分")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _load_model(model_path: str, device: str) -> tuple[Any, Any]:
    """加载 HF model + processor。"""
    print(f"Loading model: {model_path} ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    print(f"  Loaded on {device}", flush=True)
    return model, processor


def _resolve_image_paths(messages: list[dict], data_dir: str) -> list[dict]:
    """将 messages 中的相对图像路径解析为绝对路径。"""
    base = Path(data_dir)
    resolved = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    path = item.get("image", "")
                    if path and not path.startswith(("http://", "https://", "/", "data:")):
                        item = dict(item)
                        item["image"] = str((base / path).resolve())
                new_content.append(item)
            m["content"] = new_content
        resolved.append(m)
    return resolved


def _build_prompt(
    messages: list[dict],
    tools: list[dict] | None,
    processor: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], int, str]:
    """构建 prompt，返回 (inputs_dict, prompt_len, decoded_prompt_text)。"""
    chat_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    if tools:
        chat_kwargs["tools"] = tools

    inputs = processor.apply_chat_template(messages, **chat_kwargs)
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {"input_ids": inputs.to(device)}

    prompt_len = inputs["input_ids"].shape[1]
    prompt_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return inputs, prompt_len, prompt_text


def _generate(
    model: Any,
    inputs: dict[str, torch.Tensor],
    prompt_len: int,
    processor: Any,
    args: argparse.Namespace,
) -> tuple[str, str]:
    """生成一条推理结果，返回 (completion_text, full_text)。"""
    gen_kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else 1.0,
        "top_p": args.top_p,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        gen_kwargs["attention_mask"] = inputs["attention_mask"]

    with torch.no_grad():
        gen_output = model.generate(**gen_kwargs)

    full_ids = gen_output[0]
    completion_ids = full_ids[prompt_len:]
    completion_text = processor.tokenizer.decode(completion_ids, skip_special_tokens=False)
    full_text = processor.tokenizer.decode(full_ids, skip_special_tokens=False)
    return completion_text, full_text


def _parse_and_score(
    completion_text: str,
    tools: list[dict] | None,
    targets: list[dict],
) -> dict[str, Any]:
    """解析工具调用并评分。"""
    from graspo.backends.graspoflow.tool_parser import parse_qwen_tool_completion
    from graspo.core.reward import GraspoReward, RewardConfig

    parsed = parse_qwen_tool_completion(completion_text, tools=tools)
    reward_cfg = RewardConfig(
        check_think=False,
        check_json_markdown=False,
        check_list_order=False,
        marker_reward_weight=10,
        content_reward_weight=100,
    )
    scorer = GraspoReward(reward_cfg)
    result = scorer.score_parsed(parsed, targets, is_tool_call=True)
    return {
        "parsed_calls": parsed.tool_calls,
        "parse_errors": parsed.parse_errors,
        "reward": result.reward,
        "content_score": result.content_score,
        "all_right": result.all_right,
    }


def _print_sample_header(idx: int, sample: dict, data_dir: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  Sample {idx}  |  id={sample.get('id', 'N/A')}  |  source={sample.get('source', 'N/A')}")
    print(f"{'=' * 80}")


def _print_target(targets: list[dict]) -> None:
    """打印期望输出。"""
    from graspo.core.data import build_sft_target_text

    print(f"\n  ── 期望输出 (target) ──")
    for ti, target in enumerate(targets):
        target_text = build_sft_target_text(target["output"])
        print(f"  target[{ti}] id={target.get('id', '?')}:")
        for line in target_text.split("\n"):
            print(f"    {line}")
        print()


def _print_completion(
    label: str,
    completion_text: str,
    score: dict[str, Any] | None = None,
) -> None:
    """打印一行推理结果。"""
    print(f"\n  ── {label} 输出 ──")
    has_eos = completion_text.endswith("<|im_end|>") or "<|endoftext|>" in completion_text
    # 截断 <|im_end|> 后面的内容
    for marker in ["<|im_end|>", "<|endoftext|>"]:
        if marker in completion_text:
            completion_text = completion_text[: completion_text.index(marker) + len(marker)]
            break
    print(f"  length={len(completion_text)} chars  has_eos={has_eos}")
    for line in completion_text.split("\n"):
        print(f"    {line}")
    if score:
        print(f"  reward={score['reward']:.4f}  all_right={score['all_right']}  parse_errors={score['parse_errors']}")


def _print_prompt_info(prompt_text: str) -> None:
    """打印 prompt 摘要。"""
    print(f"\n  ── Prompt 信息 ──")
    print(f"  prompt 长度: {len(prompt_text)} chars")
    # 打印最后 300 字符（assistant prefix 部分）
    suffix = prompt_text[-300:]
    print(f"  prompt 末尾 (300 chars):")
    for line in suffix.split("\n"):
        print(f"    {line}")


def main() -> int:
    args = parse_args()
    data_dir = str(Path(args.data).parent)

    # 加载模型
    model, processor = _load_model(args.model, args.device)
    baseline_model, baseline_processor = None, None
    if args.baseline:
        baseline_model, baseline_processor = _load_model(args.baseline, args.device)

    # 加载数据
    with open(args.data) as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(all_samples)} samples from {args.data}", flush=True)

    end_idx = min(args.start + args.limit, len(all_samples))
    samples = all_samples[args.start : end_idx]
    print(f"Processing samples {args.start}-{end_idx - 1} ({len(samples)} total)\n", flush=True)

    # 导入评分工具
    from graspo.core.data import build_sft_target_text

    for i, sample in enumerate(samples):
        sidx = args.start + i
        messages = _resolve_image_paths(sample["messages"], data_dir)
        tools = sample.get("tools")
        targets = sample["targets"]

        _print_sample_header(sidx, sample, data_dir)

        # 构建 prompt（SFT 模型）
        inputs, prompt_len, prompt_text = _build_prompt(messages, tools, processor, args.device)

        if i == 0:
            _print_prompt_info(prompt_text)

        # 期望输出
        _print_target(targets)

        # SFT 模型推理
        completion_text, _ = _generate(model, inputs, prompt_len, processor, args)
        score = None if args.raw_only else _parse_and_score(completion_text, tools, targets)
        _print_completion("SFT Model", completion_text, score)

        # Baseline 模型推理
        if baseline_model is not None and baseline_processor is not None:
            b_inputs, b_prompt_len, _ = _build_prompt(
                messages, tools, baseline_processor, args.device
            )
            b_completion, _ = _generate(baseline_model, b_inputs, b_prompt_len, baseline_processor, args)
            b_score = None if args.raw_only else _parse_and_score(b_completion, tools, targets)
            _print_completion("Baseline (base)", b_completion, b_score)

    print(f"\n{'=' * 80}")
    print("  Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())