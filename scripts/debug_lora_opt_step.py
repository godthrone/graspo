#!/usr/bin/env python3
"""Minimal test: forward+backward+optimizer.step() on TP=4, dump LoRA deltas per rank."""
import json, os
import torch
import torch.distributed as dist

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _dump_lora(model):
    modules = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            modules[name] = {
                "a_norm": mod.lora_a.data.float().norm().item(),
                "b_norm": mod.lora_b.data.float().norm().item(),
                "b_sum": mod.lora_b.data.float().sum().item(),
                "a_shape": list(mod.lora_a.shape),
                "b_shape": list(mod.lora_b.shape),
                "shard": getattr(mod, "lora_shard_kind", "?"),
            }
    return modules


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
        "model": {"model_path": "/workspace/models/Qwen3.5-9B", "trust_remote_code": True,
                   "torch_dtype": "bfloat16",
                   "chat_template_kwargs": {"enable_thinking": False}},
        "data": {"train_path": "/workspace/data/data/elam_graspo_train.jsonl",
                 "max_prompt_length": 2048},
        "training": {"rollout_group_size": 8, "max_new_tokens": 128,
                      "temperature": 1.0, "top_p": 1.0,
                      "optimize_prompt_batch_size": 8, "optimize_times_per_step": 1,
                      "policy_ratio_clip_eps": 0.2, "max_grad_norm": 1.0},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
        "backend_config": {"native_tp": {"tp_size": world_size, "pp_size": 1,
                           "placement_strategy": "qwen3_tp", "forward_batch_size": 64,
                           "use_kv_cache_for_rollout": True,
                           "empty_cache_after_rollout_split": False}},
    })

    from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
    adapter = QwenNativeTPAdapter(config)
    adapter.setup()
    model = adapter.model
    model.train()

    # Dump BEFORE
    before = _dump_lora(model)

    # Create a simple input: random token IDs (no images needed for text-only loss test)
    batch_size = 8
    seq_len = 128
    vocab_size = int(model.config.vocab_size)

    # Use actual token IDs in valid range, with padding
    input_ids = torch.randint(0, min(vocab_size, 10000), (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    # Add some random padding
    for i in range(batch_size):
        pad_start = torch.randint(seq_len // 2, seq_len, (1,)).item()
        attention_mask[i, pad_start:] = 0

    # Forward: compute sequence log_probs (text-only, no multimodal)
    try:
        log_probs = model.sequence_log_probs(input_ids, attention_mask)
        # Dummy loss: negative mean of log_probs
        loss = -log_probs.mean()
        if rank == 0:
            print(f"Loss: {loss.item():.4f}", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] FORWARD ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        dist.barrier()
        dist.destroy_process_group()
        return

    # Backward
    try:
        loss.backward()
    except Exception as e:
        print(f"[Rank {rank}] BACKWARD ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        dist.barrier()
        dist.destroy_process_group()
        return

    # Dump gradients BEFORE optimizer step
    grad_info = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            ga = mod.lora_a.grad
            gb = mod.lora_b.grad
            grad_info[name] = {
                "a_grad_norm": ga.float().norm().item() if ga is not None else 0,
                "b_grad_norm": gb.float().norm().item() if gb is not None else 0,
                "b_grad_sum": gb.float().sum().item() if gb is not None else 0,
            }

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0,
    )

    # Optimizer step
    adapter.optimizer.step()

    # Dump AFTER
    after = _dump_lora(model)

    # Compute deltas
    delta_info = {}
    for name in before:
        ba = before[name]["b_norm"]
        aa = after[name]["b_norm"]
        delta_info[name] = {
            "b_norm_before": before[name]["b_norm"],
            "b_norm_after": after[name]["b_norm"],
            "b_norm_delta": after[name]["b_norm"] - before[name]["b_norm"],
            "b_sum_after": after[name]["b_sum"],
            "b_grad_norm": grad_info.get(name, {}).get("b_grad_norm", 0),
            "b_grad_sum": grad_info.get(name, {}).get("b_grad_sum", 0),
            "b_shape": before[name]["b_shape"],
            "shard": before[name]["shard"],
        }

    # Save per rank
    out = {"rank": rank, "deltas": delta_info}
    with open(f"/workspace/data/lora_delta_rank{rank}.json", "w") as f:
        json.dump(out, f, indent=2)

    if rank == 0:
        print(f"\nLoRA weight deltas (rank 0, first 10 modules):")
        for i, (name, info) in enumerate(delta_info.items()):
            if i >= 10:
                break
            print(f"  {name}: b_norm Δ={info['b_norm_delta']:.8f} "
                  f"b_grad_norm={info['b_grad_norm']:.6f} "
                  f"b_shape={info['b_shape']} shard={info['shard']}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
