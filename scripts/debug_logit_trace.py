#!/usr/bin/env python3
"""Phase 1b: Compare GRASPO vs HF logits at each decode step (single sample).

Finds exactly where the probability distributions start to diverge,
even when argmax still matches.
"""

import json, sys
from pathlib import Path

import torch

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_models(model_path, device):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
    from graspo.backends.native_tp.models.qwen.modeling import load_native_qwen_config
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex
    from graspo.backends.native_tp.placement import build_placement_plan

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    print("Loading HF model...", flush=True)
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
    g_model = Qwen35HybridTextModel(
        hf_config=native_cfg, loader=loader,
        tp_rank=0, tp_size=1, placement=placement,
        lora_r=0, lora_alpha=1, lora_dropout=0.0,
        lora_targets=set(), gradient_checkpointing=False,
        torch_dtype=torch.bfloat16, device=device,
    ).eval()
    return hf_model, g_model, processor


def prepare_inputs(sample, images_dir, processor, device):
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
    inputs = processor.apply_chat_template(msgs, **kwargs)
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
        if val is not None and (not hasattr(val, '__len__') or len(val) > 0):
            mm_inputs[key] = val.to(device) if isinstance(val, torch.Tensor) else val
    return input_ids, attn, mm_inputs


def main():
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    images_dir = sys.argv[3]
    device = torch.device("cuda:0")

    hf_model, g_model, processor = load_models(model_path, device)

    with open(data_path) as f:
        sample = json.loads(f.readline())

    input_ids, attn, mm_inputs = prepare_inputs(sample, images_dir, processor, device)
    prompt_len = input_ids.shape[1]
    eos_id = processor.tokenizer.eos_token_id
    print(f"Sample: {sample['id']}, prompt_len={prompt_len}", flush=True)

    # Build HF kwargs
    hf_kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if key in mm_inputs:
            hf_kwargs[key] = mm_inputs[key]

    # Prefill both models
    print("\n--- Prefill ---", flush=True)
    hf_ids = input_ids.clone()
    hf_attn = attn.clone()
    g_ids = input_ids.clone()
    g_attn = attn.clone()

    with torch.no_grad():
        hf_out = hf_model(input_ids=hf_ids, attention_mask=hf_attn, **hf_kwargs, use_cache=True)
        hf_logits = hf_out.logits
        hf_pkv = hf_out.past_key_values

        g_logits, g_pkv = g_model(
            input_ids=g_ids, attention_mask=g_attn,
            multimodal_inputs=mm_inputs, use_cache=True,
        )

    # Compare prefill logits
    hf_last = hf_logits[0, -1, :]
    g_last = g_logits[0, -1, :]
    print(f"Prefill cos_sim: {torch.nn.functional.cosine_similarity(hf_last.float(), g_last.float(), dim=0).item():.6f}")
    print(f"Prefill argmax: HF={hf_last.argmax().item()} G={g_last.argmax().item()}")

    # Decode step by step, comparing logits
    hf_next = hf_last.argmax().item()
    g_next = g_last.argmax().item()
    print(f"Step 0: HF={processor.tokenizer.decode([hf_next])!r} G={processor.tokenizer.decode([g_next])!r}")

    # For each decode step, extract the logit at last position and compare
    for step in range(64):
        hf_ids = torch.tensor([[hf_next]], dtype=torch.long, device=device)
        hf_attn = torch.cat([hf_attn, torch.ones(1, 1, dtype=hf_attn.dtype, device=device)], dim=1)

        g_ids = torch.tensor([[g_next]], dtype=torch.long, device=device)
        g_attn = torch.cat([g_attn, torch.ones(1, 1, dtype=g_attn.dtype, device=device)], dim=1)

        with torch.no_grad():
            hf_out = hf_model(
                input_ids=hf_ids, attention_mask=hf_attn,
                past_key_values=hf_pkv, use_cache=True,
            )
            hf_pkv = hf_out.past_key_values
            hf_logits = hf_out.logits

            g_logits, g_pkv = g_model(
                input_ids=g_ids, attention_mask=g_attn,
                past_key_values=g_pkv, use_cache=True,
            )

        hf_step_logit = hf_logits[0, -1, :]
        g_step_logit = g_logits[0, -1, :]

        cos_sim = torch.nn.functional.cosine_similarity(
            hf_step_logit.float(), g_step_logit.float(), dim=0
        ).item()

        hf_next = int(hf_step_logit.argmax().item())
        g_next = int(g_step_logit.argmax().item())

        hf_top5 = hf_step_logit.topk(5)
        g_top5 = g_step_logit.topk(5)

        match_str = "MATCH" if hf_next == g_next else "DIVERGE"
        print(f"  Step {step+1}: {match_str} cos_sim={cos_sim:.6f} "
              f"HF={processor.tokenizer.decode([hf_next])!r} G={processor.tokenizer.decode([g_next])!r} "
              f"HF_top5={hf_top5.indices.tolist()} G_top5={g_top5.indices.tolist()}",
              flush=True)

        if hf_next != g_next:
            print(f"  *** LOGIT DIFF at argmax: HF={hf_step_logit[hf_next].float().item():.4f} "
                  f"G={g_step_logit[g_next].float().item():.4f}", flush=True)
            # Check EOS logit
            print(f"  EOS logit: HF={hf_step_logit[eos_id].float().item():.4f} "
                  f"G={g_step_logit[eos_id].float().item():.4f}", flush=True)
            break

        if hf_next == eos_id or g_next == eos_id:
            print(f"  EOS generated, stopping")
            break


if __name__ == "__main__":
    main()
