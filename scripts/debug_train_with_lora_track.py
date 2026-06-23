#!/usr/bin/env python3
"""Run real TP=4 training with LoRA b-norm tracking per step.

Monkey-patches train_batch to collect lora_b cross-rank stats.
Runs for N steps and reports parse_err + lora_b divergence trend.
"""

import json, os
import torch
import torch.distributed as dist


def _lora_b_norms(model):
    """Get per-module lora_b norm for cross-rank tracking."""
    norms = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            norms[name] = mod.lora_b.data.float().norm().item()
    return norms


def worker():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

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
                      "policy_ratio_clip_eps": 0.2, "max_grad_norm": 1.0,
                      "rollout_max_retry_times": 1,
                      "max_steps": 10},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
        "backend_config": {"native_tp": {"tp_size": world_size, "pp_size": 1,
                           "placement_strategy": "qwen3_tp", "forward_batch_size": 64,
                           "use_kv_cache_for_rollout": True,
                           "empty_cache_after_rollout_split": False}},
        "logging": {"log_dir": "/workspace/data/logs"},
    })

    from graspo.backends.native_tp.trainer import NativeTPGraspoTrainer
    trainer = NativeTPGraspoTrainer(config)
    trainer.runtime.setup()
    adapter = trainer.runtime._require_adapter()

    # Monkey-patch: inject lora_b tracking into train_batch
    orig_train_batch = adapter.train_batch
    step_results = []

    def tracked_train_batch(experiences, **kwargs):
        nonlocal step_results

        # Before optimizer step: dump lora_b norms
        before_norms = _lora_b_norms(adapter.model)

        # Run real train_batch
        result = orig_train_batch(experiences, **kwargs)

        # After: dump again
        after_norms = _lora_b_norms(adapter.model)

        # Gather norms from all ranks
        all_before = [None] * world_size
        all_after = [None] * world_size
        dist.all_gather_object(all_before, before_norms)
        dist.all_gather_object(all_after, after_norms)

        if rank == 0:
            # Compute cross-rank spread for key layers
            key_names = sorted([n for n in before_norms if 'layers.3.' in n or 'layers.31.' in n])
            layer_spreads = {}
            for name in key_names:
                b_deltas = [all_after[r][name] - all_before[r][name] for r in range(world_size)]
                spread = max(b_deltas) - min(b_deltas)
                mean_d = sum(b_deltas) / len(b_deltas)
                layer_spreads[name] = {
                    "deltas": b_deltas,
                    "spread": spread,
                    "spread_pct": 100 * spread / max(abs(mean_d), 1e-10),
                }
            step_results.append({
                "lora_norm_before": result.get("lora_norm_before"),
                "lora_norm_after": result.get("lora_norm_after"),
                "lora_norm_delta": result.get("lora_norm_delta"),
                "loss_mean": result.get("loss_mean"),
                "grad_norm_mean": result.get("grad_norm_mean"),
                "layer_spreads": layer_spreads,
            })

        return result

    adapter.train_batch = tracked_train_batch

    if rank == 0:
        print(f"Starting training with LoRA tracking, TP={world_size}", flush=True)

    # Run training (max_steps=10)
    trainer.train()

    if rank == 0:
        print(f"\n{'='*60}")
        print("LORA DIVERGENCE SUMMARY:")
        for i, sr in enumerate(step_results):
            spreads = []
            for name, info in sr["layer_spreads"].items():
                spreads.append(f"{name.split('.')[1]}.{name.split('.')[-1]}={info['spread_pct']:.1f}%")
            print(f"  Step {i+1}: loss={sr.get('loss_mean', 'N/A')} "
                  f"lora_norm_delta={sr.get('lora_norm_delta', 'N/A')} "
                  f"spreads=[{', '.join(spreads)}]", flush=True)

        with open("/workspace/data/lora_train_track.json", "w") as f:
            json.dump(step_results, f, indent=2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
