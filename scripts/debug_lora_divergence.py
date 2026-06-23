#!/usr/bin/env python3
"""Track LoRA cross-rank divergence over multiple optimizer steps (random data)."""
import json, os
import torch
import torch.distributed as dist
import torch.nn.functional as F

torch.manual_seed(42)


def _lora_b_stats(model):
    """Get per-module lora_b norm, first-row, for cross-rank comparison."""
    stats = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            b = mod.lora_b.data
            stats[name] = {
                "b_norm": b.float().norm().item(),
                "b_first_row_norm": b[0].float().norm().item() if b.shape[0] > 0 else 0,
                "b_shape": list(b.shape),
                "shard": getattr(mod, 'lora_shard_kind', '?'),
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
        "model": {"model_path": "/workspace/models/Qwen3.5-9B", "trust_remote_code": True,
                   "torch_dtype": "bfloat16",
                   "chat_template_kwargs": {"enable_thinking": False}},
        "data": {"train_path": "/workspace/data/data/elam_graspo_train.jsonl",
                 "max_prompt_length": 2048},
        "training": {"rollout_group_size": 8, "max_new_tokens": 128,
                      "temperature": 1.0, "top_p": 1.0,
                      "optimize_prompt_batch_size": 8, "optimize_times_per_step": 1,
                      "max_grad_norm": 1.0},
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

    # Use fixed random token IDs (same across ranks via manual seed)
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 128

    all_step_results = []

    num_steps = 20
    if rank == 0:
        print(f"Running {num_steps} optimizer steps, TP={world_size}", flush=True)

    for step in range(num_steps):
        # Generate same random data on all ranks (using cpu generator then moving to GPU)
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        # Dump BEFORE
        before_stats = _lora_b_stats(model)

        # Forward
        log_probs = model.sequence_log_probs(input_ids, attention_mask)
        loss = -log_probs.mean()

        # Backward
        adapter.optimizer.zero_grad()
        loss.backward()

        # Clip and step
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        adapter.optimizer.step()

        # Dump AFTER
        after_stats = _lora_b_stats(model)

        dist.barrier()

        # All-gather stats
        all_before = [None] * world_size
        all_after = [None] * world_size
        dist.all_gather_object(all_before, before_stats)
        dist.all_gather_object(all_after, after_stats)

        if rank == 0:
            # Track cross-rank divergence for key layers
            full_attn_layers = sorted([n for n in before_stats if 'layers.3.' in n or 'layers.31.' in n])
            sample_layers = full_attn_layers[:1] + full_attn_layers[-1:]  # first and last

            for name in sample_layers:
                b_deltas = []
                b_afters = []
                for r in range(world_size):
                    bd = after_stats[name]["b_norm"] - before_stats[name]["b_norm"]
                    b_deltas.append(bd)
                    # Actually we need per-rank values, not just rank 0's

                # Get per-rank values
                per_rank_before = [all_before[r][name]["b_norm"] for r in range(world_size)]
                per_rank_after = [all_after[r][name]["b_norm"] for r in range(world_size)]
                per_rank_delta = [per_rank_after[r] - per_rank_before[r] for r in range(world_size)]
                spread = max(per_rank_delta) - min(per_rank_delta)
                mean_delta = sum(per_rank_delta) / len(per_rank_delta)
                spread_pct = 100 * spread / max(abs(mean_delta), 1e-10)

                b_shape = before_stats[name]["b_shape"]
                print(f"  Step {step:2d} {name}: b={b_shape} delta=[{', '.join(f'{d:.6f}' for d in per_rank_delta)}] "
                      f"spread={spread:.6f} ({spread_pct:.1f}%)",
                      flush=True)

            all_step_results.append({
                "step": step,
                "loss": loss.item(),
                "sample_layer_deltas": {
                    name: {
                        "b_shape": before_stats[name]["b_shape"],
                        "deltas_per_rank": [all_after[r][name]["b_norm"] - all_before[r][name]["b_norm"] for r in range(world_size)],
                    }
                    for name in sample_layers
                }
            })

    # Final analysis on rank 0
    if rank == 0:
        # Check if divergence grows over steps
        print(f"\n--- Divergence trend ---")
        for name in sample_layers:
            spreads = []
            for sr in all_step_results:
                deltas = sr["sample_layer_deltas"][name]["deltas_per_rank"]
                spread = max(deltas) - min(deltas)
                spreads.append(spread)
            print(f"  {name}: spreads over steps: {[f'{s:.6f}' for s in spreads]}")

        with open("/workspace/data/lora_divergence.json", "w") as f:
            json.dump(all_step_results, f, indent=2)
        print("Saved to /workspace/data/lora_divergence.json", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
