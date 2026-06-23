#!/usr/bin/env python3
"""Phase 0: TP=1 GRASPO vs HF step-by-step comparison for multimodal samples.

Greedy decode (temperature=0), single GPU, no distributed.
Identifies which samples diverge and at which decode step.

Usage:
    python debug_tp1_compare.py \
        --model /workspace/models/Qwen3.5-9B \
        --data /workspace/data/data/elam_graspo_train.jsonl \
        --images /workspace/images \
        --start 0 --count 20 \
        --max-new-tokens 64
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Ensure reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to Qwen3.5-9B model")
    p.add_argument("--data", required=True, help="Path to elam_graspo_train.jsonl")
    p.add_argument("--images", required=True, help="Path to images directory")
    p.add_argument("--output", default="/tmp/debug_tp1_results.json")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def load_hf_model(model_path, device):
    """Load HF Qwen3.5 model with processor."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print("Loading HF processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    print("Loading HF model...", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    return model, processor


def load_graspo_model(model_path, device):
    """Load GRASPO Qwen35HybridTextModel with TP=1."""
    from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
    from graspo.backends.native_tp.models.qwen.modeling import load_native_qwen_config
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex
    from graspo.backends.native_tp.placement import build_placement_plan

    print("Loading GRASPO model...", flush=True)
    native_cfg = load_native_qwen_config(Path(model_path))
    loader = SafetensorIndex(Path(model_path))
    placement = build_placement_plan(
        strategy="qwen3_tp",
        model_family=native_cfg.family,
        num_hidden_layers=int(native_cfg.num_hidden_layers),
        tp_size=1, pp_size=1, tp_rank=0, pp_rank=0,
        layer_types=list(getattr(native_cfg, "layer_types", []) or []),
    )
    model = Qwen35HybridTextModel(
        hf_config=native_cfg, loader=loader,
        tp_rank=0, tp_size=1, placement=placement,
        lora_r=0, lora_alpha=1, lora_dropout=0.0,
        lora_targets=set(), gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    return model


def prepare_sample_inputs(sample, images_dir, processor, device):
    """Apply chat template and return tokenized inputs."""
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

    tools = sample.get("tools")
    kwargs = {
        "tokenize": True, "add_generation_prompt": True,
        "return_dict": True, "return_tensors": "pt",
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools

    result = processor.apply_chat_template(msgs, **kwargs)

    # Result is a dict (BatchEncoding) when return_dict=True
    if isinstance(result, dict):
        input_ids = result["input_ids"].to(device)
        attn = result.get("attention_mask")
        if attn is None:
            attn = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
        else:
            attn = attn.to(device)
        # Extract multimodal keys from dict
        mm_inputs = {}
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw",
                    "pixel_values_videos", "video_grid_thw_videos",
                    "mm_token_type_ids"):
            val = result.get(key)
            if val is not None and (not hasattr(val, '__len__') or len(val) > 0):
                mm_inputs[key] = val.to(device) if isinstance(val, torch.Tensor) else val
    elif hasattr(result, "input_ids"):
        input_ids = result.input_ids.to(device)
        attn = result.attention_mask.to(device) if hasattr(result, "attention_mask") and result.attention_mask is not None else torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
        mm_inputs = {}
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw",
                    "pixel_values_videos", "video_grid_thw_videos",
                    "mm_token_type_ids"):
            val = getattr(result, key, None)
            if val is not None and (not hasattr(val, '__len__') or len(val) > 0):
                mm_inputs[key] = val.to(device) if isinstance(val, torch.Tensor) else val
    else:
        input_ids = result.to(device)
        attn = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
        mm_inputs = {}

    return input_ids, attn, mm_inputs


def _hf_model_kwargs(mm_inputs):
    """Build kwargs dict for HF model from multimodal inputs dict."""
    kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "video_grid_thw",
                "pixel_values_videos", "video_grid_thw_videos",
                "mm_token_type_ids"):
        val = mm_inputs.get(key)
        if val is not None:
            kwargs[key] = val
    return kwargs


def hf_generate(model, processor, input_ids, attn, mm_inputs, max_new_tokens, device):
    """Run HF model.generate() with greedy decode. Returns token list (completion only)."""
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # greedy
        "use_cache": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if attn is not None:
        gen_kwargs["attention_mask"] = attn
    # Add multimodal kwargs
    for key, val in mm_inputs.items():
        if val is not None:
            gen_kwargs[key] = val

    with torch.no_grad():
        gen_output = model.generate(**gen_kwargs)
    completion_ids = gen_output[0, input_ids.shape[1]:].tolist()
    return completion_ids


def hf_prefill_only(model, input_ids, attn, mm_inputs, device):
    """Run HF prefill and return logits + past_key_values."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=attn,
            **_hf_model_kwargs(mm_inputs),
            use_cache=True,
        )
    return out.logits, out.past_key_values


def graspo_prefill_only(model, input_ids, attn, mm_inputs, device):
    """Run GRASPO prefill and return logits + past_key_values."""
    with torch.no_grad():
        logits, pkv = model(
            input_ids=input_ids, attention_mask=attn,
            multimodal_inputs=mm_inputs if mm_inputs else None,
            use_cache=True,
        )
    return logits, pkv


def graspo_step_by_step(model, tokenizer, input_ids, attn, mm_inputs, max_new_tokens, device):
    """Run GRASPO model step by step with KV cache, greedy decode. Returns tokens list."""
    past_key_values = None
    tokens = []
    ids = input_ids
    current_attn = attn
    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                logits, past_key_values = model(
                    input_ids=ids, attention_mask=current_attn,
                    multimodal_inputs=mm_inputs if mm_inputs else None,
                    use_cache=True,
                )
            else:
                logits, past_key_values = model(
                    input_ids=ids, attention_mask=current_attn,
                    past_key_values=past_key_values, use_cache=True,
                )
        next_token = int(logits[0, -1, :].argmax().item())
        tokens.append(next_token)
        if next_token == eos_id:
            break

        if step == 0:
            ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_attn = torch.cat(
                [current_attn, torch.ones(1, 1, dtype=current_attn.dtype, device=device)], dim=1
            )
        else:
            ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_attn = torch.cat(
                [current_attn, torch.ones(1, 1, dtype=current_attn.dtype, device=device)], dim=1
            )

    return tokens


def compare_logits(hf_logits, g_logits, attn, label=""):
    """Compare HF vs GRASPO logits at the last valid token position."""
    actual_len = attn[0].sum().item() - 1
    hf_last = hf_logits[0, actual_len]
    g_last = g_logits[0, actual_len]
    hf_top1 = int(hf_last.argmax().item())
    g_top1 = int(g_last.argmax().item())
    match = hf_top1 == g_top1
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_last.float(), g_last.float(), dim=0
    ).item()
    return {
        "label": label,
        "hf_top1": hf_top1,
        "g_top1": g_top1,
        "match": match,
        "cos_sim": cos_sim,
    }


def compare_kv_cache(hf_pkv, g_pkv, num_layers):
    """Compare HF vs GRASPO KV cache shapes and max abs diff.

    Only compares full-attention layers (HF DynamicLayer with .keys/.values).
    Linear attention layers (HF LinearAttentionLayer) are skipped.
    """
    results = []
    for layer_idx in range(num_layers):
        hf_layer = hf_pkv.layers[layer_idx]

        # Only full-attention layers have keys/values
        if not (hasattr(hf_layer, "keys") and hasattr(hf_layer, "values")):
            continue

        hf_k = hf_layer.keys
        hf_v = hf_layer.values
        g_k = g_pkv[layer_idx][0]
        g_v = g_pkv[layer_idx][1]

        shape_match = hf_k.shape == g_k.shape and hf_v.shape == g_v.shape
        max_diff_k = (hf_k.float() - g_k.float()).abs().max().item()
        max_diff_v = (hf_v.float() - g_v.float()).abs().max().item()

        results.append({
            "layer": layer_idx,
            "shape_match": shape_match,
            "max_diff_k": max_diff_k,
            "max_diff_v": max_diff_v,
        })
    return results


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    hf_model, processor = load_hf_model(args.model, device)
    graspo_model = load_graspo_model(args.model, device)

    with open(args.data) as f:
        all_samples = [json.loads(line) for line in f]

    end_idx = min(args.start + args.count, len(all_samples))
    samples = all_samples[args.start:end_idx]
    print(f"Testing {len(samples)} samples ({args.start}-{end_idx-1})...", flush=True)

    results = []
    diverge_samples = []

    for i, sample in enumerate(samples):
        sidx = args.start + i
        sample_id = sample.get("id", f"sample_{sidx}")

        try:
            input_ids, attn, mm_inputs = prepare_sample_inputs(
                sample, args.images, processor, device,
            )
        except Exception as e:
            print(f"  [{i+1:3d}/{len(samples)}] idx={sidx:3d} PREP_ERR: {e}", flush=True)
            results.append({"idx": sidx, "id": sample_id, "error": str(e)})
            continue

        prompt_len = input_ids.shape[1]
        has_images = bool(mm_inputs.get("pixel_values") is not None)
        print(f"  [{i+1:3d}/{len(samples)}] idx={sidx:3d} id={sample_id} "
              f"prompt_len={prompt_len} images={has_images}", flush=True)

        # Phase 1.1: Compare prefill logits
        try:
            hf_logits, hf_pkv = hf_prefill_only(hf_model, input_ids, attn, mm_inputs, device)
            g_logits, g_pkv = graspo_prefill_only(graspo_model, input_ids, attn, mm_inputs, device)
        except Exception as e:
            print(f"    PREFILL_ERR: {e}", flush=True)
            results.append({"idx": sidx, "id": sample_id, "prefill_error": str(e)})
            continue

        prefill_cmp = compare_logits(hf_logits, g_logits, attn, "prefill")
        num_layers = len(hf_pkv.layers)
        kv_cmp = compare_kv_cache(hf_pkv, g_pkv, num_layers)
        kv_max_diff = max(r["max_diff_k"] for r in kv_cmp)

        print(f"    Prefill: match={prefill_cmp['match']} cos_sim={prefill_cmp['cos_sim']:.6f} "
              f"kv_max_diff={kv_max_diff:.6f}", flush=True)

        if not prefill_cmp["match"]:
            print(f"    *** PREFILL DIVERGENCE: HF={prefill_cmp['hf_top1']} "
                  f"G={prefill_cmp['g_top1']}", flush=True)

        # Run HF generate (greedy) and GRASPO step-by-step
        hf_tokens = []
        g_tokens = []
        decode_error = None
        try:
            hf_tokens = hf_generate(
                hf_model, processor, input_ids, attn, mm_inputs,
                args.max_new_tokens, device,
            )
        except Exception as e:
            decode_error = f"HF generate: {e}"
            print(f"    HF_GEN_ERR: {e}", flush=True)

        if decode_error is None:
            try:
                g_tokens = graspo_step_by_step(
                    graspo_model, processor.tokenizer, input_ids, attn, mm_inputs,
                    args.max_new_tokens, device,
                )
            except Exception as e:
                decode_error = f"GRASPO decode: {e}"
                print(f"    GRASPO_DECODE_ERR: {e}", flush=True)

        if decode_error is not None:
            result = {
                "idx": sidx, "id": sample_id,
                "prompt_len": prompt_len, "has_images": has_images,
                "prefill_match": prefill_cmp["match"],
                "prefill_cos_sim": prefill_cmp["cos_sim"],
                "decode_error": decode_error,
                "hf_tokens": hf_tokens, "g_tokens": g_tokens,
            }
            results.append(result)
            continue

        diverge_step = None
        min_len = min(len(hf_tokens), len(g_tokens))
        for step_idx in range(min_len):
            if hf_tokens[step_idx] != g_tokens[step_idx]:
                diverge_step = step_idx
                break
        if diverge_step is None and len(hf_tokens) != len(g_tokens):
            diverge_step = min_len

        hf_text = processor.tokenizer.decode(hf_tokens[:min(20, len(hf_tokens))])
        g_text = processor.tokenizer.decode(g_tokens[:min(20, len(g_tokens))])

        if diverge_step is not None:
            hf_div = processor.tokenizer.decode([hf_tokens[diverge_step]]) if diverge_step < len(hf_tokens) else "EOF"
            g_div = processor.tokenizer.decode([g_tokens[diverge_step]]) if diverge_step < len(g_tokens) else "EOF"
            print(f"    *** DIVERGE at step {diverge_step}: HF={hf_div!r} G={g_div!r} "
                  f"hf_len={len(hf_tokens)} g_len={len(g_tokens)}", flush=True)
            diverge_samples.append({
                "idx": sidx, "id": sample_id, "diverge_step": diverge_step,
                "hf_tokens": hf_tokens, "g_tokens": g_tokens,
            })
        else:
            print(f"    OK: {len(hf_tokens)} tokens match, text={hf_text[:60]!r}", flush=True)

        result = {
            "idx": sidx,
            "id": sample_id,
            "prompt_len": prompt_len,
            "has_images": has_images,
            "prefill_match": prefill_cmp["match"],
            "prefill_cos_sim": prefill_cmp["cos_sim"],
            "prefill_hf_top1": prefill_cmp["hf_top1"],
            "prefill_g_top1": prefill_cmp["g_top1"],
            "kv_max_diff": kv_max_diff,
            "hf_tokens": hf_tokens,
            "g_tokens": g_tokens,
            "diverge_step": diverge_step,
            "hf_text": hf_text,
            "g_text": g_text,
        }
        results.append(result)

    # Summary
    total = len(results)
    prefill_error = sum(1 for r in results if "prefill_error" in r)
    prefill_mismatch = sum(1 for r in results
                          if "prefill_error" not in r and not r.get("prefill_match", True))
    decode_diverge = sum(1 for r in results
                        if "prefill_error" not in r and r.get("diverge_step") is not None)
    prefill_ok_decode_ok = total - prefill_error - prefill_mismatch - decode_diverge

    print(f"\n{'='*60}", flush=True)
    print(f"Summary ({total} samples):", flush=True)
    print(f"  Prefill OK + Decode OK: {prefill_ok_decode_ok}", flush=True)
    print(f"  Prefill error:          {prefill_error}", flush=True)
    print(f"  Prefill mismatch:       {prefill_mismatch}", flush=True)
    print(f"  Decode divergence:      {decode_diverge}", flush=True)
    print(f"  Diverge samples: {[d['idx'] for d in diverge_samples]}", flush=True)

    with open(args.output, "w") as f:
        json.dump({"results": results, "diverge_samples": diverge_samples}, f, indent=2)
    print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
