#!/usr/bin/env python3
"""Phase 1: Batched TP=1 GRASPO vs HF comparison.

Batches N samples together (matching training) — tests if multi-sample
interaction triggers divergence.

Usage:
    python debug_tp1_batch.py \
        --model /workspace/models/Qwen3.5-9B \
        --data /workspace/data/data/elam_graspo_train.jsonl \
        --images /workspace/images \
        --start 0 --count 16 --batch-size 8 \
        --max-new-tokens 64
"""

import argparse, json, sys
from pathlib import Path

import torch

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="/tmp/debug_batch_results.json")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def load_models(model_path, device):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
    from graspo.backends.native_tp.models.qwen.modeling import load_native_qwen_config
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex
    from graspo.backends.native_tp.placement import build_placement_plan

    print("Loading HF model...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    print("Loading GRASPO model...", flush=True)
    native_cfg = load_native_qwen_config(Path(model_path))
    loader = SafetensorIndex(Path(model_path))
    placement = build_placement_plan(
        strategy="qwen3_tp", model_family=native_cfg.family,
        num_hidden_layers=int(native_cfg.num_hidden_layers),
        tp_size=1, pp_size=1, tp_rank=0, pp_rank=0,
        layer_types=list(getattr(native_cfg, "layer_types", []) or []),
    )
    graspo_model = Qwen35HybridTextModel(
        hf_config=native_cfg, loader=loader,
        tp_rank=0, tp_size=1, placement=placement,
        lora_r=0, lora_alpha=1, lora_dropout=0.0,
        lora_targets=set(), gradient_checkpointing=False,
        torch_dtype=torch.bfloat16, device=device,
    ).eval()
    return hf_model, graspo_model, processor


def build_batch_inputs(samples, images_dir, processor, device):
    """Build batched inputs for N samples using processor.apply_chat_template."""
    msgs_batch = []
    for sample in samples:
        msgs = []
        for m in sample["messages"]:
            content = m.get("content", "")
            if isinstance(content, list):
                new_c = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_name = Path(item["image"]).name
                        new_c.append({"type": "image", "image": f"{images_dir}/{img_name}"})
                    else:
                        new_c.append(item)
                msgs.append({"role": m["role"], "content": new_c})
            else:
                msgs.append({"role": m["role"], "content": content})
        msgs_batch.append(msgs)

    tools = samples[0].get("tools")

    # Process all samples together (batch)
    kwargs = {
        "tokenize": True, "add_generation_prompt": True,
        "return_dict": True, "return_tensors": "pt",
        "enable_thinking": False, "padding": True,
    }
    if tools:
        kwargs["tools"] = tools

    inputs = processor.apply_chat_template(msgs_batch, **kwargs)

    input_ids = inputs["input_ids"].to(device)
    attn = inputs.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(input_ids, device=device)
    else:
        attn = attn.to(device)

    # Extract multimodal keys
    mm_inputs = {}
    for key in ("pixel_values", "image_grid_thw", "video_grid_thw",
                "pixel_values_videos", "video_grid_thw_videos", "mm_token_type_ids"):
        val = inputs.get(key)
        if val is not None:
            if isinstance(val, torch.Tensor):
                mm_inputs[key] = val.to(device)
            elif isinstance(val, list) and len(val) > 0:
                mm_inputs[key] = val

    return input_ids, attn, mm_inputs


def _hf_model_kwargs(mm_inputs):
    kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "video_grid_thw",
                "pixel_values_videos", "video_grid_thw_videos", "mm_token_type_ids"):
        val = mm_inputs.get(key)
        if val is not None:
            kwargs[key] = val
    return kwargs


def hf_generate_batch(model, tokenizer, input_ids, attn, mm_inputs, max_new_tokens, device):
    """Run HF generate on a batch."""
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # greedy
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs.update(_hf_model_kwargs(mm_inputs))

    with torch.no_grad():
        gen_output = model.generate(**gen_kwargs)

    # Extract completion tokens per row
    prompt_len = input_ids.shape[1]
    completions = []
    for row in range(gen_output.shape[0]):
        row_ids = gen_output[row, prompt_len:]
        completions.append(row_ids.tolist())
    return completions


def graspo_step_by_step_batch(model, tokenizer, input_ids, attn, mm_inputs, max_new_tokens, device):
    """Run GRASPO batch step by step, greedy decode. Returns list of token lists per row."""
    batch_size = input_ids.shape[0]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Build initial per-row completion lists and finished flags
    completions = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    sequences = input_ids.clone()
    current_attn = attn.clone()
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                logits, past_key_values = model(
                    input_ids=sequences, attention_mask=current_attn,
                    multimodal_inputs=mm_inputs,
                    use_cache=True,
                )
            else:
                logits, past_key_values = model(
                    input_ids=sequences, attention_mask=current_attn,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        next_tokens = logits[:, -1, :].argmax(dim=-1)
        next_tokens[finished] = pad_id

        # Append tokens
        for row in range(batch_size):
            if not finished[row]:
                completions[row].append(int(next_tokens[row]))
                if int(next_tokens[row]) == eos_id:
                    finished[row] = True

        if finished.all():
            break

        if step == 0:
            sequences = next_tokens.unsqueeze(1)
            current_attn = torch.cat(
                [current_attn, (~finished).long().unsqueeze(1)], dim=1
            )
        else:
            sequences = next_tokens.unsqueeze(1)
            current_attn = torch.cat(
                [current_attn, (~finished).long().unsqueeze(1)], dim=1
            )

    return completions


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    hf_model, graspo_model, processor = load_models(args.model, device)

    with open(args.data) as f:
        all_samples = [json.loads(line) for line in f]

    end_idx = min(args.start + args.count, len(all_samples))
    batch_starts = list(range(args.start, end_idx, args.batch_size))

    results = []
    diverge_batches = []

    for batch_idx, batch_start in enumerate(batch_starts):
        batch_end = min(batch_start + args.batch_size, end_idx)
        batch_samples = all_samples[batch_start:batch_end]
        batch_size = len(batch_samples)
        sample_ids = [s.get("id", f"s{batch_start+i}") for i, s in enumerate(batch_samples)]

        print(f"\n{'='*60}", flush=True)
        print(f"Batch {batch_idx}: samples {batch_start}-{batch_end-1} ({batch_size} rows)", flush=True)

        try:
            input_ids, attn, mm_inputs = build_batch_inputs(
                batch_samples, args.images, processor, device,
            )
        except Exception as e:
            print(f"  BUILD_ERR: {e}", flush=True)
            results.append({"batch": batch_idx, "samples": sample_ids, "build_error": str(e)})
            continue

        has_images = bool(mm_inputs.get("pixel_values") is not None)
        print(f"  input_ids.shape={input_ids.shape} images={has_images}", flush=True)

        # Run HF and GRASPO
        try:
            hf_completions = hf_generate_batch(
                hf_model, processor.tokenizer, input_ids, attn, mm_inputs,
                args.max_new_tokens, device,
            )
        except Exception as e:
            print(f"  HF_GEN_ERR: {e}", flush=True)
            hf_completions = None
            import traceback; traceback.print_exc()

        try:
            g_completions = graspo_step_by_step_batch(
                graspo_model, processor.tokenizer, input_ids, attn, mm_inputs,
                args.max_new_tokens, device,
            )
        except Exception as e:
            print(f"  GRASPO_ERR: {e}", flush=True)
            g_completions = None
            import traceback; traceback.print_exc()

        if hf_completions is None or g_completions is None:
            results.append({
                "batch": batch_idx, "samples": sample_ids,
                "hf_error": hf_completions is None,
                "g_error": g_completions is None,
            })
            continue

        # Compare per row
        batch_diverges = 0
        for row in range(batch_size):
            hf_tokens = hf_completions[row]
            g_tokens = g_completions[row]
            diverge_step = None
            min_len = min(len(hf_tokens), len(g_tokens))
            for s in range(min_len):
                if hf_tokens[s] != g_tokens[s]:
                    diverge_step = s
                    break
            if diverge_step is None and len(hf_tokens) != len(g_tokens):
                diverge_step = min_len

            hf_preview = processor.tokenizer.decode(hf_tokens[:10]) if hf_tokens else "<empty>"
            g_preview = processor.tokenizer.decode(g_tokens[:10]) if g_tokens else "<empty>"

            if diverge_step is not None:
                batch_diverges += 1
                hf_div = processor.tokenizer.decode([hf_tokens[diverge_step]]) if diverge_step < len(hf_tokens) else "EOF"
                g_div = processor.tokenizer.decode([g_tokens[diverge_step]]) if diverge_step < len(g_tokens) else "EOF"
                print(f"  Row {row}: *** DIVERGE step={diverge_step} HF={hf_div!r} G={g_div!r} "
                      f"hf_len={len(hf_tokens)} g_len={len(g_tokens)}", flush=True)
            else:
                print(f"  Row {row}: OK len={len(hf_tokens)} "
                      f"hf={hf_preview[:40]!r} g={g_preview[:40]!r}", flush=True)

        if batch_diverges > 0:
            diverge_batches.append({
                "batch": batch_idx, "start": batch_start, "end": batch_end,
                "diverge_count": batch_diverges, "batch_size": batch_size,
            })

        results.append({
            "batch": batch_idx, "start": batch_start, "end": batch_end,
            "samples": sample_ids, "batch_size": batch_size,
            "diverge_count": batch_diverges,
        })

    # Summary
    total_batches = len(batch_starts)
    print(f"\n{'='*60}", flush=True)
    print(f"Summary ({total_batches} batches, batch_size={args.batch_size}):", flush=True)
    print(f"  Diverge batches: {len(diverge_batches)}/{total_batches}", flush=True)
    for db in diverge_batches:
        print(f"    Batch {db['batch']} (samples {db['start']}-{db['end']}): "
              f"{db['diverge_count']}/{db['batch_size']} rows diverge", flush=True)

    with open(args.output, "w") as f:
        json.dump({"results": results, "diverge_batches": diverge_batches}, f, indent=2)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
