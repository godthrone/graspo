#!/usr/bin/env python3
"""HF gold-standard baseline: run all 405 samples through pure HF transformers.

Matches GRASPO training params: temperature=1.0, top_p=1.0, max_new_tokens=512,
enable_thinking=False.

Usage:
    python hf_full_baseline.py \
        --model /home/zhangzy/models/Qwen3.5-9B \
        --data /home/zhangzy/elam_v12_fk/data/elam_graspo_train.jsonl \
        --images /home/zhangzy/elam_v12_fk/images \
        --output /home/zhangzy/hf_baseline_results.json \
        --start 0 --count 405
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="hf_baseline_results.json")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=405)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_messages(sample, images_dir):
    """Convert sample messages, resolving relative image paths."""
    msgs = []
    for m in sample["messages"]:
        content = m.get("content", "")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_name = Path(item["image"]).name
                    new_content.append({
                        "type": "image",
                        "image": f"{images_dir}/{img_name}",
                    })
                else:
                    new_content.append(item)
            msgs.append({"role": m["role"], "content": new_content})
        else:
            msgs.append({"role": m["role"], "content": content})
    return msgs


def main():
    args = parse_args()

    print(f"Loading model from {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True,
    ).to(args.device).eval()
    print("Model loaded.", flush=True)

    with open(args.data) as f:
        all_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(all_samples)} samples from {args.data}", flush=True)

    end_idx = min(args.start + args.count, len(all_samples))
    samples = all_samples[args.start:end_idx]
    print(f"Processing samples {args.start}-{end_idx-1} ({len(samples)} total)", flush=True)

    # Lazy import after torch is loaded (graspo imports torch)
    from graspo.backends.native_tp.tool_parser import parse_qwen_tool_completion

    results = []
    total_parse_err = 0
    total_completions = 0
    total_eos = 0
    total_truncated = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        sidx = args.start + i
        msgs = build_messages(sample, args.images)
        tools = sample.get("tools")

        chat_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        if tools:
            chat_kwargs["tools"] = tools

        try:
            inputs = processor.apply_chat_template(msgs, **chat_kwargs)
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids.to(args.device)
                attn = inputs.attention_mask.to(args.device) if (
                    hasattr(inputs, "attention_mask")
                    and inputs.attention_mask is not None
                ) else None
            else:
                input_ids = inputs.to(args.device)
                attn = None

            prompt_len = input_ids.shape[1]
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature if args.temperature > 0 else 1.0,
                "top_p": args.top_p,
                "use_cache": True,
                "pad_token_id": processor.tokenizer.eos_token_id,
            }
            if attn is not None:
                gen_kwargs["attention_mask"] = attn

            with torch.no_grad():
                gen_output = model.generate(**gen_kwargs)

            full_ids = gen_output[0]
            completion_ids = full_ids[prompt_len:]
            text = processor.tokenizer.decode(completion_ids)

            # Check if EOS was generated
            eos_id = processor.tokenizer.eos_token_id
            hit_eos = int(eos_id) in completion_ids.tolist() if len(completion_ids) > 0 else False
            truncated = len(completion_ids) >= args.max_new_tokens and not hit_eos

            parsed = parse_qwen_tool_completion(text, tools=tools)
            has_err = len(parsed.parse_errors) > 0

            if has_err:
                total_parse_err += 1
            total_completions += 1
            if hit_eos:
                total_eos += 1
            if truncated:
                total_truncated += 1

            result = {
                "idx": sidx,
                "id": sample.get("id", f"sample_{sidx}"),
                "prompt_len": prompt_len,
                "completion_len": len(completion_ids),
                "hit_eos": hit_eos,
                "truncated": truncated,
                "parse_err": has_err,
                "parse_errors": parsed.parse_errors if has_err else [],
                "text_preview": text[:200].replace("\n", "\\n"),
                "text_full": text,
            }
            results.append(result)

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            flag = "ERR" if has_err else "OK"
            eos_flag = "EOS" if hit_eos else ("TRUNC" if truncated else "NOEOS")
            print(
                f"  [{i+1:3d}/{len(samples)}] idx={sidx:3d} [{flag}] [{eos_flag}] "
                f"len={len(completion_ids):3d} | {result['text_preview'][:80]}",
                flush=True,
            )

        except Exception as e:
            print(f"  [{i+1:3d}/{len(samples)}] idx={sidx:3d} EXCEPTION: {e}", flush=True)
            results.append({
                "idx": sidx,
                "id": sample.get("id", f"sample_{sidx}"),
                "parse_err": True,
                "parse_errors": [f"EXCEPTION: {e}"],
                "text_preview": "",
                "text_full": "",
            })
            total_parse_err += 1
            total_completions += 1

    elapsed = time.time() - start_time
    summary = {
        "model": args.model,
        "data": args.data,
        "start": args.start,
        "count": args.count,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "total_samples": len(samples),
        "total_parse_err": total_parse_err,
        "parse_err_rate": total_parse_err / max(total_completions, 1),
        "total_eos": total_eos,
        "eos_rate": total_eos / max(total_completions, 1),
        "total_truncated": total_truncated,
        "elapsed_sec": elapsed,
        "samples_per_sec": len(samples) / elapsed if elapsed > 0 else 0,
    }

    output = {"summary": summary, "results": results}
    with open(args.output, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Summary: {total_parse_err}/{total_completions} parse errors "
          f"({100*total_parse_err/max(total_completions,1):.1f}%)", flush=True)
    print(f"EOS: {total_eos}/{total_completions} ({100*total_eos/max(total_completions,1):.1f}%)", flush=True)
    print(f"Truncated: {total_truncated}/{total_completions}", flush=True)
    print(f"Time: {elapsed:.0f}s ({len(samples)/elapsed:.2f} samples/s)", flush=True)
    print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
