#!/usr/bin/env python3
"""Phase 2b: Run 2 optimizer steps, track lora_a and lora_b divergence over steps."""
import json, os
import torch
import torch.distributed as dist

torch.manual_seed(42)

def _lora_stats(model):
    stats = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, 'lora_a') or not mod.lora_enabled or mod.lora_a is None:
            continue
        a = mod.lora_a.data
        b = mod.lora_b.data
        stats[name] = {
            "a_norm": float(a.float().norm().item()),
            "b_norm": float(b.float().norm().item()),
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
        "backend": "native-tp", "model": {"model_path": "/workspace/models/Qwen3.5-9B", "trust_remote_code": True, "torch_dtype": "bfloat16", "chat_template_kwargs": {"enable_thinking": False}},
        "data": {"train_path": "/workspace/data/data/elam_graspo_train.jsonl", "max_prompt_length": 2048},
        "training": {"rollout_group_size": 8, "max_new_tokens": 128, "temperature": 1.0, "top_p": 1.0, "optimize_prompt_batch_size": 4, "optimize_times_per_step": 1, "max_grad_norm": 1.0},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
        "backend_config": {"native_tp": {"tp_size": world_size, "pp_size": 1, "placement_strategy": "qwen3_tp", "forward_batch_size": 16, "use_kv_cache_for_rollout": True, "empty_cache_after_rollout_split": True}},
    })
    from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
    adapter = QwenNativeTPAdapter(config)
    adapter.setup()
    model = adapter.model

    # Real multimodal data
    from pathlib import Path
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True)
    with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()][:4]
    all_input_ids, all_attention_masks, all_mm_inputs_list = [], [], []
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
                    else: nc.append(item)
                msgs.append({"role": m["role"], "content": nc})
            else: msgs.append({"role": m["role"], "content": c})
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_dict": True, "return_tensors": "pt", "enable_thinking": False}
        if s.get("tools"): kwargs["tools"] = s["tools"]
        inputs = proc.apply_chat_template(msgs, **kwargs)
        all_input_ids.append(inputs["input_ids"]); all_attention_masks.append(inputs["attention_mask"])
        mm = {}
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            v = inputs.get(k)
            if v is not None and v.numel() > 0: mm[k] = v
        all_mm_inputs_list.append(mm)
    max_len = max(ids.shape[1] for ids in all_input_ids)
    padded_ids, padded_masks = [], []
    for ids, mask in zip(all_input_ids, all_attention_masks):
        pl = max_len - ids.shape[1]
        padded_ids.append(torch.cat([ids, torch.full((1, pl), 0, dtype=ids.dtype)], dim=1) if pl > 0 else ids)
        padded_masks.append(torch.cat([mask, torch.zeros(1, pl, dtype=mask.dtype)], dim=1) if pl > 0 else mask)
    batch_ids = torch.cat(padded_ids, dim=0).to(device)
    batch_mask = torch.cat(padded_masks, dim=0).to(device)
    mm_batch = {}
    for k in ("pixel_values", "image_grid_thw"):
        tensors = [mm.get(k) for mm in all_mm_inputs_list if mm.get(k) is not None]
        if tensors: mm_batch[k] = torch.cat([t.to(device) for t in tensors], dim=0)

    n_steps = 3
    for step in range(n_steps):
        before_stats = _lora_stats(model)
        model.train()
        if mm_batch:
            log_probs = model.sequence_log_probs(batch_ids, batch_mask, multimodal_inputs=mm_batch)
        else:
            log_probs = model.sequence_log_probs(batch_ids, batch_mask)
        loss = -log_probs.mean()
        adapter.optimizer.zero_grad()
        loss.backward()
        # Sync non-sharded LoRA gradients BEFORE optimizer step
        from graspo.backends.native_tp.models.qwen.lora import _sync_nonsharded_lora_grads
        from graspo.backends.native_tp.tensor_utils import _TENSOR_PARALLEL_GROUP
        if _TENSOR_PARALLEL_GROUP is not None:
            _sync_nonsharded_lora_grads(model, _TENSOR_PARALLEL_GROUP)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        adapter.optimizer.step()
        after_stats = _lora_stats(model)

        all_before = [None] * world_size; all_after = [None] * world_size
        dist.all_gather_object(all_before, before_stats); dist.all_gather_object(all_after, after_stats)

        if rank == 0:
            a_div = 0; b_div = 0; total = len(before_stats)
            for name in before_stats:
                a_deltas = [all_after[r][name]["a_norm"] - all_before[r][name]["a_norm"] for r in range(world_size)]
                b_deltas = [all_after[r][name]["b_norm"] - all_before[r][name]["b_norm"] for r in range(world_size)]
                if max(a_deltas) - min(a_deltas) > 1e-8: a_div += 1
                if max(b_deltas) - min(b_deltas) > 1e-8: b_div += 1

            # Show worst divergences
            spreads = []
            for name in before_stats:
                a_deltas = [all_after[r][name]["a_norm"] - all_before[r][name]["a_norm"] for r in range(world_size)]
                b_deltas = [all_after[r][name]["b_norm"] - all_before[r][name]["b_norm"] for r in range(world_size)]
                a_spread = max(a_deltas) - min(a_deltas)
                b_spread = max(b_deltas) - min(b_deltas)
                spreads.append((a_spread + b_spread, name, a_spread, b_spread))
            spreads.sort(reverse=True)
            worst_a = max(s[2] for s in spreads)
            worst_b = max(s[3] for s in spreads)
            print(f"Step {step+1}: loss={loss.item():.4f} lora_a_diverged={a_div}/{total} lora_b_diverged={b_div}/{total} "
                  f"worst_a_spread={worst_a:.8f} worst_b_spread={worst_b:.8f}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    worker()
