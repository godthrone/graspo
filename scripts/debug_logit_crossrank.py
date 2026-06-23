#!/usr/bin/env python3
"""Check: are logits identical across TP ranks after training?"""
import json, os
import torch
import torch.distributed as dist
from pathlib import Path

torch.manual_seed(42)

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

    # Load real multimodal data
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("/workspace/models/Qwen3.5-9B", trust_remote_code=True, local_files_only=True)
    with open("/workspace/data/data/elam_graspo_train.jsonl") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()][:2]

    all_ids, all_masks, all_mm = [], [], []
    for s in raw_samples:
        msgs = []
        for m in s["messages"]:
            c = m.get("content", "")
            if isinstance(c, list):
                nc = []
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "image":
                        nc.append({"type": "image", "image": f"/workspace/images/{Path(item['image']).name}"})
                    else: nc.append(item)
                msgs.append({"role": m["role"], "content": nc})
            else: msgs.append({"role": m["role"], "content": c})
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_dict": True, "return_tensors": "pt", "enable_thinking": False}
        if s.get("tools"): kwargs["tools"] = s["tools"]
        inputs = proc.apply_chat_template(msgs, **kwargs)
        all_ids.append(inputs["input_ids"]); all_masks.append(inputs["attention_mask"])
        mm = {}
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            v = inputs.get(k)
            if v is not None and v.numel() > 0: mm[k] = v
        all_mm.append(mm)

    max_len = max(ids.shape[1] for ids in all_ids)
    padded_ids = [torch.cat([ids, torch.full((1, max_len-ids.shape[1]), 0, dtype=ids.dtype)], dim=1) if max_len > ids.shape[1] else ids for ids in all_ids]
    padded_masks = [torch.cat([mask, torch.zeros(1, max_len-mask.shape[1], dtype=mask.dtype)], dim=1) if max_len > mask.shape[1] else mask for mask in all_masks]
    batch_ids = torch.cat(padded_ids, dim=0).to(device)
    batch_mask = torch.cat(padded_masks, dim=0).to(device)
    mm_batch = {}
    for k in ("pixel_values", "image_grid_thw"):
        tensors = [m.get(k) for m in all_mm if m.get(k) is not None]
        if tensors: mm_batch[k] = torch.cat([t.to(device) for t in tensors], dim=0)

    # ---- Before training: check logit consistency ----
    model.eval()
    with torch.no_grad():
        out = model(batch_ids, attention_mask=batch_mask, multimodal_inputs=mm_batch, use_cache=False)
    logits_before = out[0] if isinstance(out, tuple) else out
    last_pos = batch_mask.sum(dim=1).long() - 1
    logits_before_last = logits_before[torch.arange(len(last_pos)), last_pos, :].float()

    # Gather logits from all ranks
    all_logits_before = [torch.zeros_like(logits_before_last) for _ in range(world_size)]
    dist.all_gather(all_logits_before, logits_before_last)
    if rank == 0:
        diffs = [(all_logits_before[i] - all_logits_before[0]).abs().max().item() for i in range(1, world_size)]
        print(f"BEFORE training - logit cross-rank maxdiff: {diffs}")

    # ---- One optimizer step ----
    model.train()
    log_probs = model.sequence_log_probs(batch_ids, batch_mask, multimodal_inputs=mm_batch)
    loss = -log_probs.mean()
    adapter.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    adapter.optimizer.step()

    # ---- After training: check logit consistency again (eval mode to avoid dropout noise) ----
    model.eval()
    with torch.no_grad():
        out = model(batch_ids, attention_mask=batch_mask, multimodal_inputs=mm_batch, use_cache=False)
    logits_after = out[0] if isinstance(out, tuple) else out
    logits_after_last = logits_after[torch.arange(len(last_pos)), last_pos, :].float()

    all_logits_after = [torch.zeros_like(logits_after_last) for _ in range(world_size)]
    dist.all_gather(all_logits_after, logits_after_last)
    if rank == 0:
        diffs = [(all_logits_after[i] - all_logits_after[0]).abs().max().item() for i in range(1, world_size)]
        print(f"AFTER training - logit cross-rank maxdiff: {diffs}")

        # Also compare before vs after for rank 0
        logit_change = (logits_before_last - logits_after_last).abs().max().item()
        print(f"Rank 0 logit change before→after training: {logit_change:.6f}")

        if all(d < 1e-6 for d in diffs):
            print("Logits are IDENTICAL across ranks → parse errors NOT from TP numerical issues")
        else:
            print(f"Logits DIFFER across ranks → TP all-reduce or LoRA path has a bug")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    worker()
