#!/usr/bin/env python3
"""Phase 2: Check if decoder LoRA weights diverge across TP ranks after optimizer step.

Hypothesis: In TP-sharded layers, the non-sharded LoRA matrix receives different
gradients per rank, causing weight divergence. Specifically:
- shard_kind="rows" (q_proj, v_proj): lora_a maps from full input → diverges
- shard="in" (o_proj): lora_b maps to full output → diverges
"""
import json, os
import torch
import torch.distributed as dist

torch.manual_seed(42)


def _lora_full_stats(model):
    """Collect lora_a and lora_b per-module stats for ALL LoRA modules."""
    stats = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, 'lora_a') or not mod.lora_enabled or mod.lora_a is None:
            continue
        a = mod.lora_a.data
        b = mod.lora_b.data
        stats[name] = {
            "a_norm": float(a.float().norm().item()),
            "b_norm": float(b.float().norm().item()),
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "shard_kind": getattr(mod, 'lora_shard_kind', '?'),
        }
    return stats


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
            "temperature": 1.0, "top_p": 1.0,
            "optimize_prompt_batch_size": 4,
            "optimize_times_per_step": 1,
            "max_grad_norm": 1.0,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
        "backend_config": {
            "native_tp": {
                "tp_size": world_size, "pp_size": 1,
                "placement_strategy": "qwen3_tp", "forward_batch_size": 16,
                "use_kv_cache_for_rollout": True,
                "empty_cache_after_rollout_split": True,
            },
        },
    })

    from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
    adapter = QwenNativeTPAdapter(config)
    adapter.setup()
    model = adapter.model

    # ---- BEFORE stats ----
    before_stats = _lora_full_stats(model)
    all_before = [None] * world_size
    dist.all_gather_object(all_before, before_stats)

    if rank == 0:
        n_total = len(before_stats)
        print(f"Total LoRA modules: {n_total}")
        by_shard = {}
        for name, s in before_stats.items():
            sk = s["shard_kind"]
            by_shard[sk] = by_shard.get(sk, 0) + 1
        print(f"By shard_kind: {by_shard}")

    # ---- Run one optimizer step with real multimodal data ----
    model.train()
    from pathlib import Path
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(
        "/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True,
    )

    with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()][:4]

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

    # Pad and batch
    max_len = max(ids.shape[1] for ids in all_input_ids)
    padded_ids = []
    padded_masks = []
    pad_id = 0
    for ids, mask in zip(all_input_ids, all_attention_masks):
        pl = max_len - ids.shape[1]
        if pl > 0:
            padded_ids.append(torch.cat([ids, torch.full((1, pl), pad_id, dtype=ids.dtype)], dim=1))
            padded_masks.append(torch.cat([mask, torch.zeros(1, pl, dtype=mask.dtype)], dim=1))
        else:
            padded_ids.append(ids)
            padded_masks.append(mask)

    batch_ids = torch.cat(padded_ids, dim=0).to(device)
    batch_mask = torch.cat(padded_masks, dim=0).to(device)

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

    # Forward + backward + step
    if mm_batch:
        log_probs = model.sequence_log_probs(batch_ids, batch_mask, multimodal_inputs=mm_batch)
    else:
        log_probs = model.sequence_log_probs(batch_ids, batch_mask)

    loss = -log_probs.mean()
    if rank == 0:
        print(f"Loss: {loss.item():.6f}")

    adapter.optimizer.zero_grad()
    loss.backward()

    # ---- Check gradients BEFORE optimizer step ----
    if rank == 0:
        print(f"\n--- Gradient analysis (before optimizer step) ---")
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and "lora_" in name:
            gnorm = float(param.grad.float().norm().item())
            grad_stats[name] = {"gnorm": gnorm}

    all_grads = [None] * world_size
    dist.all_gather_object(all_grads, grad_stats)

    if rank == 0:
        # Show a few representative modules
        samples = []
        for name in sorted(grad_stats.keys()):
            parts = name.split(".")
            if "full_attn" in name and "layers" in name:
                layer_idx = [p for p in parts if p.isdigit()][0] if any(p.isdigit() for p in parts) else "?"
                samples.append((int(layer_idx) if layer_idx.isdigit() else 99, name))

        samples.sort()
        shown = 0
        for _, name in samples:
            if shown >= 6:
                break
            gnorms = []
            for r in range(world_size):
                gnorms.append(all_grads[r].get(name, {}).get("gnorm", 0))
            spread = max(gnorms) - min(gnorms)
            mean_g = sum(gnorms) / len(gnorms)
            spread_pct = 100 * spread / max(mean_g, 1e-10)
            is_a = ".lora_a" in name
            short = name.replace("model.language_model.layers.", "L")
            print(f"  {short}: norms={[f'{g:.6f}' for g in gnorms]} spread={spread:.6f} ({spread_pct:.1f}%) {'⚠️ A_DIVERGES' if is_a and spread_pct > 1 else ''}")

    # Clip and step
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0,
    )
    adapter.optimizer.step()

    # ---- AFTER stats ----
    after_stats = _lora_full_stats(model)
    all_after = [None] * world_size
    dist.all_gather_object(all_after, after_stats)

    if rank == 0:
        print(f"\n--- AFTER optimizer step: Cross-rank LoRA weight divergence ---")
        all_names = sorted(before_stats.keys())
        a_diverged = 0
        b_diverged = 0
        threshold = 1e-7

        # Show top divergences
        spreads = []
        for name in all_names:
            a_before = [all_before[r][name]["a_norm"] for r in range(world_size)]
            a_after = [all_after[r][name]["a_norm"] for r in range(world_size)]
            b_before = [all_before[r][name]["b_norm"] for r in range(world_size)]
            b_after = [all_after[r][name]["b_norm"] for r in range(world_size)]

            a_deltas = [a_after[r] - a_before[r] for r in range(world_size)]
            b_deltas = [b_after[r] - b_before[r] for r in range(world_size)]

            a_spread = max(a_deltas) - min(a_deltas)
            b_spread = max(b_deltas) - min(b_deltas)

            sk = before_stats[name].get("shard_kind", "?")
            spreads.append((a_spread + b_spread, name, a_spread, b_spread, a_deltas, b_deltas, sk))

        spreads.sort(reverse=True)
        n_show = min(10, len(spreads))
        print(f"Top-{n_show} most divergent modules:")
        for i, (total_spread, name, a_sp, b_sp, a_d, b_d, sk) in enumerate(spreads[:n_show]):
            short = name.replace("model.language_model.", "")
            a_flag = " ⚠️" if a_sp > threshold else ""
            b_flag = " ⚠️" if b_sp > threshold else ""
            print(f"  {short} (shard={sk})")
            print(f"    lora_a deltas: {[f'{v:+.8f}' for v in a_d]} spread={a_sp:.8f}{a_flag}")
            print(f"    lora_b deltas: {[f'{v:+.8f}' for v in b_d]} spread={b_sp:.8f}{b_flag}")

        # Count diverged
        for name in all_names:
            a_before_vals = [all_before[r][name]["a_norm"] for r in range(world_size)]
            a_after_vals = [all_after[r][name]["a_norm"] for r in range(world_size)]
            b_before_vals = [all_before[r][name]["b_norm"] for r in range(world_size)]
            b_after_vals = [all_after[r][name]["b_norm"] for r in range(world_size)]
            a_deltas = [a_after_vals[r] - a_before_vals[r] for r in range(world_size)]
            b_deltas = [b_after_vals[r] - b_before_vals[r] for r in range(world_size)]
            if max(a_deltas) - min(a_deltas) > threshold:
                a_diverged += 1
            if max(b_deltas) - min(b_deltas) > threshold:
                b_diverged += 1

        print(f"\nlora_a diverged: {a_diverged}/{len(all_names)}")
        print(f"lora_b diverged: {b_diverged}/{len(all_names)}")

        if a_diverged > 0 or b_diverged > 0:
            print(f"\n*** CONFIRMED: LoRA weight divergence across TP ranks after optimizer step! ***")
            print(f"*** Root cause: non-sharded LoRA matrix receives different gradients per rank ***")
        else:
            print(f"\nNo LoRA divergence detected. Bug is elsewhere.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
