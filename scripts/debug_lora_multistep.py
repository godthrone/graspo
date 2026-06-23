#!/usr/bin/env python3
"""Multi-step LoRA tracking on TP=4: run N training steps, dump LoRA per-rank at each step.

Captures:
- LoRA a/b norms, per-layer, per-rank, before/after optimizer step
- Parse error rate per step
- Cross-rank lora_b similarity (cosine similarity between rank pairs)
"""

import json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _collect_lora_stats(model):
    """Collect LoRA stats: norms, first row, for cross-rank comparison."""
    stats = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'lora_a') and mod.lora_enabled and mod.lora_a is not None:
            a_data = mod.lora_a.data
            b_data = mod.lora_b.data
            stats[name] = {
                "a_norm": a_data.float().norm().item(),
                "b_norm": b_data.float().norm().item(),
                "b_sum": b_data.float().sum().item(),
                "b_first_row_norm": b_data[0].float().norm().item() if b_data.shape[0] > 0 else 0,
                "a_shape": list(a_data.shape),
                "b_shape": list(b_data.shape),
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
                      "policy_ratio_clip_eps": 0.2, "max_grad_norm": 1.0,
                      "replay_buffer_optimize_threshold": 64,
                      "rollout_max_retry_times": 1},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_preset": "language_safe"},
        "reward": {"reward_model": "compare", "compare_score_scale": 1.0},
        "backend_config": {"native_tp": {"tp_size": world_size, "pp_size": 1,
                           "placement_strategy": "qwen3_tp", "forward_batch_size": 64,
                           "use_kv_cache_for_rollout": True,
                           "empty_cache_after_rollout_split": False}},
    })

    from graspo.backends.native_tp.models.qwen.adapter import QwenNativeTPAdapter
    adapter = QwenNativeTPAdapter(config)
    adapter.setup()

    from graspo.core.data import load_jsonl
    from graspo.backends.native_tp.tool_parser import parse_qwen_tool_completion
    from graspo.core.reward import GraspoReward

    all_samples = load_jsonl(config.data.train_path)
    reward_engine = GraspoReward(config)

    results_per_step = []
    sample_offset = 0

    num_steps = 10
    if rank == 0:
        print(f"Running {num_steps} training steps, TP={world_size}, LoRA r=16", flush=True)

    for step_idx in range(num_steps):
        # Select samples
        batch_size = config.training.optimize_prompt_batch_size
        start = sample_offset
        end = min(start + batch_size, len(all_samples))
        if end - start < batch_size:
            sample_offset = 0
            start = 0
            end = batch_size
        samples = all_samples[start:end]
        sample_offset = end

        if rank == 0:
            print(f"\n--- Step {step_idx+1}/{num_steps}, samples {start}-{end-1} ---", flush=True)

        # Dump BEFORE
        before_stats = _collect_lora_stats(adapter.model)
        dist.barrier()

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

        # Score rewards and check parse errors
        experiences = []
        parse_errs = 0
        total_comps = 0
        for gen, sample in zip(generations, samples):
            for comp in gen.completions:
                parsed = parse_qwen_tool_completion(comp, tools=sample.tools)
                if parsed.parse_errors:
                    parse_errs += 1
                total_comps += 1
                rr = reward_engine.score(sample, comp, tools=sample.tools)
                # Build experience dict matching what trainer expects
                exp = {
                    "sequences": gen.sequences,
                    "attention_mask": gen.attention_mask,
                    "action_mask": gen.action_mask,
                    "completion": comp,
                    "reward": rr.reward,
                    "content_score": rr.content_score,
                    "metadata": gen.metadata,
                }
                experiences.append(exp)

        if rank == 0:
            print(f"  Rollout: {parse_errs}/{total_comps} parse errors", flush=True)

        # Run train_batch
        metrics = adapter.train_batch(
            experiences,
            policy_ratio_clip_eps=config.training.policy_ratio_clip_eps,
            optimize_times_per_step=config.training.optimize_times_per_step,
            max_grad_norm=config.training.max_grad_norm,
        )

        dist.barrier()

        # Dump AFTER
        after_stats = _collect_lora_stats(adapter.model)

        # Compute per-layer deltas and cross-rank info
        # All-gather stats from all ranks for comparison
        before_json = json.dumps({"rank": rank, "stats": before_stats})
        after_json = json.dumps({"rank": rank, "stats": after_stats})

        # Gather all ranks' stats to rank 0
        all_before = [None] * world_size
        all_after = [None] * world_size
        dist.all_gather_object(all_before, before_json)
        dist.all_gather_object(all_after, after_json)

        all_before_parsed = [json.loads(s) for s in all_before]
        all_after_parsed = [json.loads(s) for s in all_after]

        if rank == 0:
            step_result = {
                "step": step_idx + 1,
                "samples": f"{start}-{end-1}",
                "parse_err": parse_errs,
                "total_comps": total_comps,
                "lora_norm_before": metrics.get("lora_norm_before"),
                "lora_norm_after": metrics.get("lora_norm_after"),
                "lora_norm_delta": metrics.get("lora_norm_delta"),
                "loss_mean": metrics.get("loss_mean"),
                "grad_norm_mean": metrics.get("grad_norm_mean"),
                "rank_deltas": {},
            }

            # Compare key layers across ranks
            key_layers = [n for n in before_stats if 'full_attn' in n and 'q_proj' in n]
            key_layers = key_layers[:2] + key_layers[-2:]  # first 2 and last 2

            for name in key_layers:
                layer_info = {"name": name, "b_shape": before_stats[name]["b_shape"]}
                for r in range(world_size):
                    b_before = json.loads(all_before[r])["stats"][name]
                    b_after = json.loads(all_after[r])["stats"][name]
                    layer_info[f"rank{r}_b_before"] = b_before["b_norm"]
                    layer_info[f"rank{r}_b_after"] = b_after["b_norm"]
                    layer_info[f"rank{r}_b_delta"] = b_after["b_norm"] - b_before["b_norm"]
                # Compute max spread across ranks
                b_deltas = [layer_info[f"rank{r}_b_delta"] for r in range(world_size)]
                layer_info["b_delta_spread"] = max(b_deltas) - min(b_deltas)
                layer_info["b_delta_mean"] = sum(b_deltas) / len(b_deltas)
                step_result["rank_deltas"][name] = layer_info

                print(f"  {name}: b={layer_info['b_shape']} "
                      f"b_delta_mean={layer_info['b_delta_mean']:.8f} "
                      f"b_delta_spread={layer_info['b_delta_spread']:.8f} "
                      f"({layer_info['b_delta_spread']/max(layer_info['b_delta_mean'], 1e-10)*100:.1f}% spread)",
                      flush=True)

            results_per_step.append(step_result)

            # Check if parse_err spiked
            if parse_errs > total_comps * 0.1:  # >10% error
                print(f"  *** HIGH PARSE ERROR: {parse_errs}/{total_comps}", flush=True)

        dist.barrier()

    # Final summary on rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY:")
        for r in results_per_step:
            pct = 100 * r["parse_err"] / max(r["total_comps"], 1)
            loss = r.get("loss_mean", "N/A")
            flag = " !!!" if pct > 10 else ""
            print(f"  Step {r['step']:2d}: parse_err={r['parse_err']}/{r['total_comps']} ({pct:.0f}%){flag} "
                  f"loss={loss} lora_norm={r.get('lora_norm_before', '?')}→{r.get('lora_norm_after', '?')}", flush=True)

        # Save detailed results
        with open("/workspace/data/lora_multistep_results.json", "w") as f:
            json.dump(results_per_step, f, indent=2)
        print(f"Results saved to /workspace/data/lora_multistep_results.json", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
