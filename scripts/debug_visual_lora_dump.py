#!/usr/bin/env python3
"""Phase 1.1: Dump visual tower LoRA weights per-rank before/after 1 optimizer step.

TP=4 training with real multimodal data.
Verifies whether visual tower LoRA diverges across TP ranks.
"""
import json, os
import torch
import torch.distributed as dist

torch.manual_seed(42)


def _visual_lora_stats(model):
    """Collect per-module lora_a/lora_b stats for visual tower modules."""
    stats = {}
    for name, mod in model.named_modules():
        # Only visual tower modules
        if "visual." not in name:
            continue
        if not hasattr(mod, 'lora_a') or not mod.lora_enabled or mod.lora_a is None:
            continue
        a = mod.lora_a.data
        b = mod.lora_b.data
        stats[name] = {
            "a_norm": float(a.float().norm().item()),
            "b_norm": float(b.float().norm().item()),
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "a_sum": float(a.float().sum().item()),
            "b_sum": float(b.float().sum().item()),
        }
    return stats


def _log_visual_lora_divergence(label, all_rank_stats, rank):
    """Print cross-rank divergence for visual LoRA modules."""
    if rank != 0:
        return
    n_ranks = len(all_rank_stats)
    if n_ranks < 2:
        return

    # Compute cross-rank spread for each module
    all_names = sorted(all_rank_stats[0].keys())
    max_a_spread = 0.0
    max_b_spread = 0.0
    max_a_name = ""
    max_b_name = ""
    total_a_spread = 0.0
    total_b_spread = 0.0
    n_modules = len(all_names)

    for name in all_names:
        a_norms = [all_rank_stats[r][name]["a_norm"] for r in range(n_ranks)]
        b_norms = [all_rank_stats[r][name]["b_norm"] for r in range(n_ranks)]
        a_spread = max(a_norms) - min(a_norms)
        b_spread = max(b_norms) - min(b_norms)
        total_a_spread += a_spread
        total_b_spread += b_spread
        if a_spread > max_a_spread:
            max_a_spread = a_spread
            max_a_name = name
        if b_spread > max_b_spread:
            max_b_spread = b_spread
            max_b_name = name

    avg_a_spread = total_a_spread / max(n_modules, 1)
    avg_b_spread = total_b_spread / max(n_modules, 1)
    print(f"\n[{label}] Visual LoRA cross-rank (n_modules={n_modules}):")
    print(f"  lora_a: max_spread={max_a_spread:.8f} avg_spread={avg_a_spread:.8f} worst={max_a_name}")
    print(f"  lora_b: max_spread={max_b_spread:.8f} avg_spread={avg_b_spread:.8f} worst={max_b_name}")

    # Detailed print for top-5 most divergent modules
    spreads = []
    for name in all_names:
        a_norms = [all_rank_stats[r][name]["a_norm"] for r in range(n_ranks)]
        b_norms = [all_rank_stats[r][name]["b_norm"] for r in range(n_ranks)]
        a_spread = max(a_norms) - min(a_norms)
        b_spread = max(b_norms) - min(b_norms)
        spreads.append((a_spread + b_spread, name, a_norms, b_norms))
    spreads.sort(reverse=True)

    n_show = min(5, len(spreads))
    print(f"  Top-{n_show} most divergent modules:")
    for i, (total_spread, name, a_norms, b_norms) in enumerate(spreads[:n_show]):
        short = name.replace("model.visual.", "")
        print(f"    {short}: a_norms={[f'{v:.6f}' for v in a_norms]} b_norms={[f'{v:.6f}' for v in b_norms]}")

    return {
        "max_a_spread": max_a_spread,
        "max_b_spread": max_b_spread,
        "avg_a_spread": avg_a_spread,
        "avg_b_spread": avg_b_spread,
        "n_modules": n_modules,
    }


def worker():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from graspo.backends.native_tp.tensor_utils import _set_tensor_parallel_group
    _set_tensor_parallel_group(dist.new_group(list(range(world_size))), world_size)

    from graspo.core.schema import GraspoConfig
    config = GraspoConfig.from_dict({
        "backend": "native-tp",
        "model": {
            "model_path": "/workspace/models/Qwen3.5-9B",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "data": {
            "train_path": "/workspace/data/data/elam_graspo_train.jsonl",
            "max_prompt_length": 2048,
        },
        "training": {
            "rollout_group_size": 8,
            "max_new_tokens": 128,
            "temperature": 1.0,
            "top_p": 1.0,
            "optimize_prompt_batch_size": 4,
            "optimize_times_per_step": 1,
            "max_grad_norm": 1.0,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
        "backend_config": {
            "native_tp": {
                "tp_size": world_size,
                "pp_size": 1,
                "placement_strategy": "qwen3_tp",
                "forward_batch_size": 16,
                "use_kv_cache_for_rollout": True,
                "empty_cache_after_rollout_split": True,
            },
        },
    })

    from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
    adapter = QwenNativeTPAdapter(config)
    adapter.setup()
    model = adapter.model

    if rank == 0:
        print(f"TP={world_size}, model ready. Collecting BEFORE stats...")

    # ---- BEFORE stats ----
    before_stats = _visual_lora_stats(model)
    all_before = [None] * world_size
    dist.all_gather_object(all_before, before_stats)
    before_summary = _log_visual_lora_divergence("BEFORE optimizer step", all_before, rank)

    # ---- Count visual LoRA modules ----
    n_vis_lora = len(before_stats)
    if rank == 0:
        print(f"Visual tower LoRA modules found: {n_vis_lora}")
        if n_vis_lora == 0:
            print("ERROR: No visual tower LoRA modules found! Check lora_targets.")

    # ---- One optimizer step with real multimodal data ----
    model.train()
    import json as _json
    from pathlib import Path

    # Load 4 real multimodal samples
    with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
        raw_samples = [_json.loads(line) for line in f if line.strip()][:4]

    # Build messages and apply chat template
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(
        "/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True,
    )

    all_input_ids = []
    all_attention_masks = []
    all_mm_inputs_list = []

    for s in raw_samples:
        msgs = []
        for m in s["messages"]:
            c = m.get("content", "")
            if isinstance(c, list):
                nc = []
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_name = Path(item["image"]).name
                        nc.append({"type": "image", "image": f"/workspace/images/{img_name}"})
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
        if s.get("tools"):
            kwargs["tools"] = s["tools"]
        inputs = proc.apply_chat_template(msgs, **kwargs)

        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])
        mm = {}
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            v = inputs.get(k)
            if v is not None and v.numel() > 0:
                mm[k] = v
        all_mm_inputs_list.append(mm)

    # Pad to the same length
    max_len = max(ids.shape[1] for ids in all_input_ids)
    pad_token_id = 0  # Qwen uses 0 as pad in GRASPO

    padded_ids = []
    padded_masks = []
    for ids, mask in zip(all_input_ids, all_attention_masks):
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            padded_ids.append(torch.cat([
                ids, torch.full((1, pad_len), pad_token_id, dtype=ids.dtype)
            ], dim=1))
            padded_masks.append(torch.cat([
                mask, torch.zeros(1, pad_len, dtype=mask.dtype)
            ], dim=1))
        else:
            padded_ids.append(ids)
            padded_masks.append(mask)

    batch_ids = torch.cat(padded_ids, dim=0).to(device)
    batch_mask = torch.cat(padded_masks, dim=0).to(device)

    # Merge multimodal inputs (concatenate across samples)
    mm_batch = {}
    for k in ("pixel_values", "image_grid_thw"):
        tensors = []
        for mm in all_mm_inputs_list:
            t = mm.get(k)
            if t is not None:
                tensors.append(t.to(device))
        if tensors:
            mm_batch[k] = torch.cat(tensors, dim=0)

    if rank == 0:
        print(f"Batch: ids={batch_ids.shape}, mask={batch_mask.shape}")
        for k, v in mm_batch.items():
            print(f"  {k}: {v.shape}")

    # Forward + backward + optimizer step
    if mm_batch:
        log_probs = model.sequence_log_probs(batch_ids, batch_mask, multimodal_inputs=mm_batch)
    else:
        log_probs = model.sequence_log_probs(batch_ids, batch_mask)

    loss = -log_probs.mean()
    if rank == 0:
        print(f"Loss: {loss.item():.6f}")

    adapter.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0,
    )
    adapter.optimizer.step()

    # Check if any visual LoRA params got gradients
    vis_grad_count = 0
    vis_grad_max = 0.0
    for name, param in model.named_parameters():
        if "visual." in name and "lora_" in name and param.grad is not None:
            vis_grad_count += 1
            gnorm = float(param.grad.float().norm().item())
            if gnorm > vis_grad_max:
                vis_grad_max = gnorm
    if rank == 0:
        print(f"Visual LoRA params with grad: {vis_grad_count}, max grad norm: {vis_grad_max:.6f}")

    # ---- AFTER stats ----
    after_stats = _visual_lora_stats(model)
    all_after = [None] * world_size
    dist.all_gather_object(all_after, after_stats)
    after_summary = _log_visual_lora_divergence("AFTER optimizer step", all_after, rank)

    # ---- DELTA analysis ----
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DELTA ANALYSIS (after - before per-rank)")
        all_names = sorted(all_before[0].keys())
        # For each module, compute per-rank deltas
        for name in all_names[:5]:  # Show first 5
            a_deltas = []
            b_deltas = []
            for r in range(world_size):
                a_d = all_after[r][name]["a_norm"] - all_before[r][name]["a_norm"]
                b_d = all_after[r][name]["b_norm"] - all_before[r][name]["b_norm"]
                a_deltas.append(a_d)
                b_deltas.append(b_d)
            a_spread = max(a_deltas) - min(a_deltas)
            b_spread = max(b_deltas) - min(b_deltas)
            short = name.replace("model.visual.", "")
            print(f"  {short}: a_delta={[f'{d:+.6f}' for d in a_deltas]} spread={a_spread:.6f} | "
                  f"b_delta={[f'{d:+.6f}' for d in b_deltas]} spread={b_spread:.6f}")

        # Summary: count modules with significant divergence
        sig_threshold = 1e-6
        a_diverged = 0
        b_diverged = 0
        for name in all_names:
            a_deltas = [all_after[r][name]["a_norm"] - all_before[r][name]["a_norm"] for r in range(world_size)]
            b_deltas = [all_after[r][name]["b_norm"] - all_before[r][name]["b_norm"] for r in range(world_size)]
            if max(a_deltas) - min(a_deltas) > sig_threshold:
                a_diverged += 1
            if max(b_deltas) - min(b_deltas) > sig_threshold:
                b_diverged += 1

        print(f"\nModules with significant lora_a divergence: {a_diverged}/{n_vis_lora}")
        print(f"Modules with significant lora_b divergence: {b_diverged}/{n_vis_lora}")

        result = {
            "tp_size": world_size,
            "n_visual_lora_modules": n_vis_lora,
            "a_diverged_count": a_diverged,
            "b_diverged_count": b_diverged,
            "before_summary": before_summary,
            "after_summary": after_summary,
        }
        with open("/workspace/data/visual_lora_dump.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to /workspace/data/visual_lora_dump.json")

        if a_diverged > 0 or b_diverged > 0:
            print("\n*** BUG CONFIRMED: Visual tower LoRA DIVERGES across TP ranks! ***")
        else:
            print("\nVisual tower LoRA does NOT diverge across TP ranks. Hypothesis REJECTED.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
