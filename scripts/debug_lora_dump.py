#!/usr/bin/env python3
"""Dump LoRA weights before/after a single optimizer step on TP=4.

Usage (via torchrun --nproc_per_node=4):
    python debug_lora_dump.py
"""

import json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _collect_lora_state(model, label):
    """Collect LoRA weight metadata from model into a dict."""
    state = {"label": label, "modules": {}}
    for name, module in model.named_modules():
        if hasattr(module, "lora_a") and module.lora_enabled and module.lora_a is not None:
            info = {
                "lora_a_norm": module.lora_a.data.float().norm().item(),
                "lora_a_mean": module.lora_a.data.float().mean().item(),
                "lora_a_std": module.lora_a.data.float().std().item(),
                "lora_a_min": module.lora_a.data.float().min().item(),
                "lora_a_max": module.lora_a.data.float().max().item(),
                "lora_b_norm": module.lora_b.data.float().norm().item(),
                "lora_b_mean": module.lora_b.data.float().mean().item(),
                "lora_b_std": module.lora_b.data.float().std().item(),
                "lora_b_min": module.lora_b.data.float().min().item(),
                "lora_b_max": module.lora_b.data.float().max().item(),
                "lora_a_shape": list(module.lora_a.shape),
                "lora_b_shape": list(module.lora_b.shape),
                "shard_kind": getattr(module, "lora_shard_kind", "?"),
                "target_name": getattr(module, "lora_target_name", "?"),
            }
            state["modules"][name] = info
    return state


def worker():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    from graspo.backends.native_tp.tensor_utils import _set_tensor_parallel_group
    ranks = list(range(world_size))
    tp_group = dist.new_group(ranks)
    _set_tensor_parallel_group(tp_group, world_size)

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
            "optimize_prompt_batch_size": 8,
            "optimize_times_per_step": 1,
        },
        "lora": {
            "r": 16, "alpha": 32, "dropout": 0.05,
            "target_preset": "language_safe",
        },
        "backend_config": {
            "native_tp": {
                "tp_size": world_size, "pp_size": 1,
                "placement_strategy": "qwen3_tp",
                "forward_batch_size": 64,
                "use_kv_cache_for_rollout": True,
                "empty_cache_after_rollout_split": False,
            },
        },
    })

    from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
    adapter = QwenNativeTPAdapter(config)
    adapter.setup()

    # Collect LoRA state BEFORE training step
    before_state = _collect_lora_state(adapter.model, "before_optimizer_step")

    # Save BEFORE state per rank
    output_file = f"/workspace/data/lora_dump_rank{rank}_before.json"
    with open(output_file, "w") as f:
        json.dump(before_state, f, indent=2, default=str)

    if rank == 0:
        print(f"BEFORE: {len(before_state['modules'])} LoRA modules captured", flush=True)
        # Print summary of key modules
        for name, info in sorted(before_state["modules"].items()):
            if "full_attn" in name:
                print(f"  {name}: a_shape={info['lora_a_shape']} b_shape={info['lora_b_shape']} "
                      f"a_norm={info['lora_a_norm']:.4f} b_norm={info['lora_b_norm']:.6f} "
                      f"shard={info['shard_kind']}", flush=True)

    # Run ONE training step
    if rank == 0:
        print("\nRunning one training step...", flush=True)

    from graspo.core.data import load_jsonl
    from graspo.core.reward import GraspoReward
    from graspo.core.decision import should_optimize

    all_samples = load_jsonl(config.data.train_path)
    samples = all_samples[:config.training.optimize_prompt_batch_size]

    # Rollout
    generations = adapter.generate_sample_groups(
        samples=samples,
        rollout_group_size=config.training.rollout_group_size,
        max_new_tokens=config.training.max_new_tokens,
        max_prompt_length=config.data.max_prompt_length,
        temperature=config.training.temperature,
        top_p=config.training.top_p,
        chat_template_kwargs={"enable_thinking": False},
    )

    # Compute rewards
    reward_engine = GraspoReward(config)
    reward_results = []
    for sample, gen in zip(samples, generations):
        for comp in gen.completions:
            rr = reward_engine.score(sample, comp, tools=sample.tools)
            reward_results.append(rr)

    # Optimize
    if should_optimize(reward_results):
        if rank == 0:
            print(f"Optimizing with {len(reward_results)} completions...", flush=True)
        adapter.optimize_step(generations, samples, reward_results)
        if rank == 0:
            print("Optimizer step completed.", flush=True)

    dist.barrier()

    # Collect LoRA state AFTER training step
    after_state = _collect_lora_state(adapter.model, "after_optimizer_step")

    output_file = f"/workspace/data/lora_dump_rank{rank}_after.json"
    with open(output_file, "w") as f:
        json.dump(after_state, f, indent=2, default=str)

    if rank == 0:
        print(f"\nAFTER: {len(after_state['modules'])} LoRA modules", flush=True)
        # Compute deltas for key modules
        for name, info in sorted(after_state["modules"].items()):
            if "full_attn" in name:
                before_info = before_state["modules"][name]
                delta_a = info['lora_a_norm'] - before_info['lora_a_norm']
                delta_b = info['lora_b_norm'] - before_info['lora_b_norm']
                print(f"  {name}: a_norm={info['lora_a_norm']:.4f} (Δ={delta_a:.6f}) "
                      f"b_norm={info['lora_b_norm']:.6f} (Δ={delta_b:.6f}) "
                      f"b_mean={info['lora_b_mean']:.8f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
