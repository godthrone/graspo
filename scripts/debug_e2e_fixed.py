#!/usr/bin/env python3
"""E2E: trainer with LoRA tracking, reduced config to avoid OOM."""
import json, os
import torch
import torch.distributed as dist

torch.manual_seed(42)

def _lora_b_vals(model, tp_group):
    """Gather LoRA b norms from all ranks."""
    norms = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            norms[name] = mod.lora_b.data.float().norm().item()
    all_norms = [None] * dist.get_world_size(tp_group)
    dist.all_gather_object(all_norms, norms)
    return all_norms  # list of dicts, one per rank

def worker():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    from graspo.backends.native_tp.tensor_utils import _set_tensor_parallel_group
    tp_group = dist.new_group(list(range(world_size)))
    _set_tensor_parallel_group(tp_group, world_size)

    from graspo.core.schema import GraspoConfig
    config = GraspoConfig.from_dict({
        'backend': 'native-tp',
        'model': {'model_path': '/workspace/models/Qwen3.5-9B', 'trust_remote_code': True,
                   'torch_dtype': 'bfloat16', 'chat_template_kwargs': {'enable_thinking': False}},
        'data': {'train_path': '/workspace/data/data/elam_graspo_train.jsonl', 'max_prompt_length': 2048},
        'training': {'rollout_group_size': 4, 'max_new_tokens': 64, 'temperature': 1.0, 'top_p': 1.0,
                      'optimize_prompt_batch_size': 4, 'optimize_times_per_step': 1,
                      'policy_ratio_clip_eps': 0.2, 'max_grad_norm': 1.0,
                      'rollout_max_retry_times': 1, 'max_steps': 8},
        'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05, 'target_preset': 'language_safe'},
        'backend_config': {'native_tp': {'tp_size': world_size, 'pp_size': 1,
                           'placement_strategy': 'qwen3_tp', 'forward_batch_size': 16,
                           'use_kv_cache_for_rollout': True, 'empty_cache_after_rollout_split': True}},
    })

    from graspo.backends.native_tp.trainer import NativeTPGraspoTrainer
    trainer = NativeTPGraspoTrainer(config)
    trainer.runtime.setup()
    adapter = trainer.runtime._require_adapter()

    # Track LoRA across steps
    step_data = []

    orig_train_batch = adapter.train_batch
    def tracked_train_batch(experiences, **kwargs):
        result = orig_train_batch(experiences, **kwargs)
        all_norms = _lora_b_vals(adapter.model, tp_group)
        step_data.append(all_norms)
        return result
    adapter.train_batch = tracked_train_batch

    if rank == 0:
        print(f"E2E training: max_steps=8, TP={world_size}, G=4, B=4")

    # Run training
    trainer.train()

    if rank == 0 and step_data:
        print(f"\n--- LORA B CROSS-RANK DIVERGENCE ---")
        # Compute max spread per step
        for step_idx, all_norms in enumerate(step_data):
            max_spread = 0
            max_name = ""
            for name in sorted(all_norms[0].keys()):
                vals = [all_norms[r][name] for r in range(world_size)]
                spread = max(vals) - min(vals)
                if spread > max_spread:
                    max_spread = spread
                    max_name = name
            short = max_name.split(".")[1] + "." + max_name.split(".")[-1]
            print(f"  Step {step_idx+1}: max_spread={max_spread:.6f} ({short})")

        with open("/workspace/data/e2e_final.json", "w") as f:
            json.dump({"num_steps": len(step_data), "num_ranks": world_size}, f)
        print("Saved to /workspace/data/e2e_final.json")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    worker()
