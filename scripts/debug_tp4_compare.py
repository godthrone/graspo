#!/usr/bin/env python3
"""Phase 2: TP=4 GRASPO vs HF comparison — greedy decode, single sample.

Usage:
    torchrun --nproc_per_node=4 debug_tp4_compare.py \
        --model /workspace/models/Qwen3.5-9B \
        --data /workspace/data/data/elam_graspo_train.jsonl \
        --images /workspace/images
"""

import json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--max-new-tokens", type=int, default=50)
    return p.parse_args()


def worker():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    tp_size = world_size  # TP=4

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from graspo.backends.native_tp.tensor_utils import _set_tensor_parallel_group
    ranks = list(range(tp_size))
    tp_group = dist.new_group(ranks)
    _set_tensor_parallel_group(tp_group, tp_size)

    model_path = args.model
    data_path = args.data
    images_dir = args.images

    # --- Load GRASPO TP=4 model ---
    from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
    from graspo.backends.native_tp.models.qwen.modeling import load_native_qwen_config
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex
    from graspo.backends.native_tp.placement import build_placement_plan

    if rank == 0:
        print("Loading GRASPO TP=4 model...", flush=True)
    native_cfg = load_native_qwen_config(Path(model_path))
    loader = SafetensorIndex(Path(model_path))
    layer_types = list(getattr(native_cfg, "layer_types", []) or [])
    placement = build_placement_plan(
        strategy="qwen3_tp", model_family=native_cfg.family,
        num_hidden_layers=int(native_cfg.num_hidden_layers),
        tp_size=tp_size, pp_size=1, tp_rank=rank, pp_rank=0,
        layer_types=layer_types,
    )
    g_model = Qwen35HybridTextModel(
        hf_config=native_cfg, loader=loader,
        tp_rank=rank, tp_size=tp_size, placement=placement,
        lora_r=0, lora_alpha=1, lora_dropout=0.0,
        lora_targets=set(), gradient_checkpointing=False,
        torch_dtype=torch.bfloat16, device=device,
    ).eval()
    if rank == 0:
        print("GRASPO TP=4 model loaded.", flush=True)

    # --- Load HF model on rank 0 only ---
    hf_model = None
    processor = None
    if rank == 0:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        print("Loading HF model on rank 0...", flush=True)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True,
        ).to(device).eval()
        print("HF model loaded.", flush=True)

    # --- Load sample and prepare inputs (all ranks load independently) ---
    from transformers import AutoProcessor
    _proc = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )

    with open(data_path) as f:
        sample = json.loads(f.readline())

    msgs = []
    for m in sample["messages"]:
        c = m.get("content", "")
        if isinstance(c, list):
            nc = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_name = Path(item["image"]).name
                    nc.append({"type": "image", "image": f"{images_dir}/{img_name}"})
                else:
                    nc.append(item)
            msgs.append({"role": m["role"], "content": nc})
        else:
            msgs.append({"role": m["role"], "content": c})

    kwargs = {
        "tokenize": True, "add_generation_prompt": True,
        "return_dict": True, "return_tensors": "pt",
        "enable_thinking": False,
    }
    tools = sample.get("tools")
    if tools:
        kwargs["tools"] = tools
    inputs = _proc.apply_chat_template(msgs, **kwargs)
    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)

    mm_inputs = {}
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        val = inputs.get(key)
        if val is not None and len(val) > 0:
            mm_inputs[key] = val.to(device) if isinstance(val, torch.Tensor) else val

    if rank == 0:
        processor = _proc  # Save for later use
    del _proc

    # Barrier before starting
    dist.barrier()
    if rank == 0:
        print(f"Input shape: {input_ids.shape}, attn sum: {attn.sum().item()}", flush=True)

    # --- Run GRASPO decode (all ranks) ---
    g_tokens = []
    g_logits_list = []
    g_pkv = None
    g_seqs = input_ids.clone()
    g_attn = attn.clone()
    eos_id = 248047  # Qwen3.5 EOS token ID
    # Get actual eos_id from model config
    eos_id = int(getattr(g_model.config, "eos_token_id", 248047))

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            if step == 0:
                logits, g_pkv = g_model(
                    input_ids=g_seqs, attention_mask=g_attn,
                    multimodal_inputs=mm_inputs if mm_inputs else None,
                    use_cache=True,
                )
            else:
                logits, g_pkv = g_model(
                    input_ids=g_seqs, attention_mask=g_attn,
                    past_key_values=g_pkv, use_cache=True,
                )

        next_token = int(logits[0, -1, :].argmax().item())
        g_tokens.append(next_token)
        g_logits_list.append(logits[0, -1, :].detach().clone())

        g_seqs = torch.tensor([[next_token]], dtype=torch.long, device=device)
        g_attn = torch.cat(
            [g_attn, torch.ones(1, 1, dtype=g_attn.dtype, device=device)], dim=1
        )

        if next_token == eos_id:
            break

    if rank == 0:
        print(f"GRASPO TP=4: {len(g_tokens)} tokens generated", flush=True)
        print(f"  Text: {processor.tokenizer.decode(g_tokens)[:150]!r}", flush=True)

    # --- Run HF decode (rank 0 only) ---
    hf_tokens = []
    hf_logits_list = []
    if rank == 0:
        # HF generate with output_logits
        hf_kw = {}
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            v = mm_inputs.get(k)
            if v is not None:
                hf_kw[k] = v

        with torch.no_grad():
            hf_gen = hf_model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True, output_logits=True,
                **hf_kw,
            )

        hf_completion_ids = hf_gen.sequences[0, input_ids.shape[1]:].tolist()
        hf_tokens = hf_completion_ids

        print(f"HF: {len(hf_tokens)} tokens generated", flush=True)
        print(f"  Text: {processor.tokenizer.decode(hf_tokens)[:150]!r}", flush=True)

        # --- Compare ---
        print(f"\n{'='*60}", flush=True)
        print("TOKEN COMPARISON (HF vs GRASPO TP=4):", flush=True)

        min_len = min(len(hf_tokens), len(g_tokens))
        diverge_step = None
        for s in range(min_len):
            if hf_tokens[s] != g_tokens[s]:
                diverge_step = s
                break
        if diverge_step is None and len(hf_tokens) != len(g_tokens):
            diverge_step = min_len

        if diverge_step is not None:
            print(f"  *** DIVERGE at step {diverge_step}", flush=True)
            if diverge_step < len(hf_tokens):
                print(f"  HF: {processor.tokenizer.decode([hf_tokens[diverge_step]])!r}", flush=True)
            if diverge_step < len(g_tokens):
                print(f"  G:  {processor.tokenizer.decode([g_tokens[diverge_step]])!r}", flush=True)

            # Compare logits at divergence step
            g_step_logit = g_logits_list[diverge_step].float()
            if diverge_step < len(hf_gen.logits):
                hf_step_logit = hf_gen.logits[diverge_step][0].float()
                cos_sim = torch.nn.functional.cosine_similarity(
                    hf_step_logit, g_step_logit, dim=0
                ).item()
                print(f"  cos_sim at divergence: {cos_sim:.6f}", flush=True)
                print(f"  HF top5: {hf_step_logit.topk(5).indices.tolist()}", flush=True)
                print(f"  G  top5: {g_step_logit.topk(5).indices.tolist()}", flush=True)
        else:
            print(f"  ALL {min_len} TOKENS MATCH!", flush=True)

        # Also compare logits at key steps
        print(f"\nLOGIT COMPARISON (cos_sim per step):", flush=True)
        diverge_found = False
        for s in range(min(len(hf_gen.logits), len(g_logits_list))):
            hf_logit = hf_gen.logits[s][0].float()
            g_logit = g_logits_list[s].float()
            cos_sim = torch.nn.functional.cosine_similarity(hf_logit, g_logit, dim=0).item()
            hf_tok = int(hf_logit.argmax().item())
            g_tok = int(g_logit.argmax().item())
            match = "OK" if hf_tok == g_tok else "DIVERGE"

            if s <= 3 or s >= len(hf_gen.logits) - 3 or match == "DIVERGE":
                eos_val_hf = hf_logit[eos_id].item()
                eos_val_g = g_logit[eos_id].item()
                print(f"  Step {s:2d}: {match:7s} cos_sim={cos_sim:.6f} "
                      f"EOS:HF={eos_val_hf:.4f} G={eos_val_g:.4f}", flush=True)
                if match == "DIVERGE":
                    diverge_found = True
            elif s == 4 and not diverge_found:
                print(f"  ... (steps 4-{len(hf_gen.logits)-4} OK, omitted)", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
