#!/usr/bin/env python3
"""Test: sync lora_b across TP ranks after each optimizer step.

If parse_err disappears → cross-rank divergence IS the root cause.
If parse_err persists → bug is elsewhere.
"""

import os
import torch
import torch.distributed as dist


def _sync_lora_b(model, tp_group):
    """All-reduce (AVG) all lora_b weights across TP ranks."""
    for mod in model.modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            dist.all_reduce(mod.lora_b.data, op=dist.ReduceOp.AVG, group=tp_group)


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
    })

    from graspo.backends.native_tp.trainer import NativeTPGraspoTrainer
    trainer = NativeTPGraspoTrainer(config)
    trainer.runtime.setup()
    adapter = trainer.runtime._require_adapter()

    # Monkey-patch: sync lora_b after each optimizer step
    orig_train_batch = adapter.train_batch

    def synced_train_batch(experiences, **kwargs):
        result = orig_train_batch(experiences, **kwargs)
        _sync_lora_b(adapter.model, tp_group)
        if rank == 0:
            print(f"  [LORA_SYNC] lora_b synced across {world_size} ranks", flush=True)
        return result

    adapter.train_batch = synced_train_batch

    if rank == 0:
        print(f"Starting training with LoRA sync fix, TP={world_size}", flush=True)

    trainer.train()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
