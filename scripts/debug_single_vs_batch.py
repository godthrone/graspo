#!/usr/bin/env python3
"""Diagnostic: Compare single-sample vs batched GRASPO decode with same sample.

If a sample works in single mode but fails in batched mode, this isolates
the batching interaction as the root cause.
"""

import json, sys
from pathlib import Path

import torch

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model_and_processor(model_path, device):
    from transformers import AutoProcessor
    from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
    from graspo.backends.native_tp.models.qwen.modeling import load_native_qwen_config
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex
    from graspo.backends.native_tp.placement import build_placement_plan

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    native_cfg = load_native_qwen_config(Path(model_path))
    loader = SafetensorIndex(Path(model_path))
    placement = build_placement_plan(
        strategy="qwen3_tp", model_family=native_cfg.family,
        num_hidden_layers=int(native_cfg.num_hidden_layers),
        tp_size=1, pp_size=1, tp_rank=0, pp_rank=0,
        layer_types=list(getattr(native_cfg, "layer_types", []) or []),
    )
    model = Qwen35HybridTextModel(
        hf_config=native_cfg, loader=loader,
        tp_rank=0, tp_size=1, placement=placement,
        lora_r=0, lora_alpha=1, lora_dropout=0.0,
        lora_targets=set(), gradient_checkpointing=False,
        torch_dtype=torch.bfloat16, device=device,
    ).eval()
    return model, processor


def prepare_inputs(samples, images_dir, processor, device):
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

    mm_inputs = {}
    for key in ("pixel_values", "image_grid_thw", "video_grid_thw",
                "pixel_values_videos", "video_grid_thw_videos", "mm_token_type_ids"):
        val = inputs.get(key)
        if val is not None and hasattr(val, '__len__') and len(val) > 0:
            mm_inputs[key] = val.to(device) if isinstance(val, torch.Tensor) else val

    return input_ids, attn, mm_inputs


def decode_greedy(model, tokenizer, input_ids, attn, mm_inputs, max_new_tokens, device):
    """GRASPO greedy decode, returns tokens per row + logits per step."""
    batch_size = input_ids.shape[0]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    completions = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    sequences = input_ids.clone()
    current_attn = attn.clone()
    past_key_values = None
    step_logits = []  # Store top-5 logits per step for first row

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                logits, past_key_values = model(
                    input_ids=sequences, attention_mask=current_attn,
                    multimodal_inputs=mm_inputs, use_cache=True,
                )
            else:
                logits, past_key_values = model(
                    input_ids=sequences, attention_mask=current_attn,
                    past_key_values=past_key_values, use_cache=True,
                )

        # Record top-5 logits for first row
        first_row_logits = logits[0, -1, :]
        top5_vals, top5_idx = first_row_logits.topk(5)
        step_logits.append({
            "step": step,
            "top5_tokens": top5_idx.tolist(),
            "top5_values": top5_vals.float().tolist(),
            "eos_logit": first_row_logits[eos_id].float().item(),
            "rope_deltas_shape": str(getattr(model, 'rope_deltas', None).shape)
                if getattr(model, 'rope_deltas', None) is not None else None,
        })

        next_tokens = logits[:, -1, :].argmax(dim=-1)
        next_tokens[finished] = pad_id

        for row in range(batch_size):
            if not finished[row]:
                tok = int(next_tokens[row])
                completions[row].append(tok)
                if tok == eos_id:
                    finished[row] = True

        if finished.all():
            break

        sequences = next_tokens.unsqueeze(1)
        current_attn = torch.cat(
            [current_attn, (~finished).long().unsqueeze(1)], dim=1
        )

    return completions, step_logits


def main():
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    images_dir = sys.argv[3]
    device = torch.device("cuda:0")

    model, processor = load_model_and_processor(model_path, device)

    with open(data_path) as f:
        all_samples = [json.loads(line) for line in f]

    # Pick one target sample (idx 0) and 7 dummy samples for batching
    target_idx = 0
    target = all_samples[target_idx]

    # Single-sample test
    print("=" * 60, flush=True)
    print("SINGLE-SAMPLE decode:", flush=True)
    ids1, attn1, mm1 = prepare_inputs([target], images_dir, processor, device)
    model.rope_deltas = None  # Reset state
    comps1, logs1 = decode_greedy(model, processor.tokenizer, ids1, attn1, mm1, 64, device)
    t1 = comps1[0]
    print(f"  Tokens: {len(t1)}, text: {processor.tokenizer.decode(t1[:20])!r}...", flush=True)

    # Batched test (target + 7 other samples)
    print("=" * 60, flush=True)
    print("BATCHED decode (target + 7 others):", flush=True)
    batch_samples = [target] + all_samples[1:8]
    ids8, attn8, mm8 = prepare_inputs(batch_samples, images_dir, processor, device)
    model.rope_deltas = None  # Reset state
    comps8, logs8 = decode_greedy(model, processor.tokenizer, ids8, attn8, mm8, 64, device)
    t8 = comps8[0]  # Row 0 is the target sample
    print(f"  Tokens: {len(t8)}, text: {processor.tokenizer.decode(t8[:20])!r}...", flush=True)

    # Compare
    print("=" * 60, flush=True)
    print("COMPARISON (single vs batched, same sample):", flush=True)
    diverge_step = None
    min_len = min(len(t1), len(t8))
    for s in range(min_len):
        if t1[s] != t8[s]:
            diverge_step = s
            break
    if diverge_step is None and len(t1) != len(t8):
        diverge_step = min_len

    if diverge_step is not None:
        print(f"  DIVERGE at step {diverge_step}: "
              f"single={processor.tokenizer.decode([t1[diverge_step]])!r} "
              f"batch={processor.tokenizer.decode([t8[diverge_step]])!r}", flush=True)
        # Compare logits at divergence step
        if diverge_step < len(logs1) and diverge_step < len(logs8):
            ls = logs1[diverge_step]
            lb = logs8[diverge_step]
            print(f"  Single top5: {ls['top5_tokens']} vals={[f'{v:.4f}' for v in ls['top5_values']]}", flush=True)
            print(f"  Batch  top5: {lb['top5_tokens']} vals={[f'{v:.4f}' for v in lb['top5_values']]}", flush=True)
            print(f"  Single EOS logit: {ls['eos_logit']:.4f}", flush=True)
            print(f"  Batch  EOS logit: {lb['eos_logit']:.4f}", flush=True)
            print(f"  rope_deltas: single={ls['rope_deltas_shape']} batch={lb['rope_deltas_shape']}", flush=True)
    else:
        print(f"  OK: {len(t1)} tokens match", flush=True)


if __name__ == "__main__":
    main()
