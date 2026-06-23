#!/usr/bin/env python3
"""TP=4 adapter + LoRA (B_init=0) + T=1.0 + batch: test all training conditions except optimizer.

Usage (on 228):
    docker run --rm -e NVIDIA_VISIBLE_DEVICES=4,5,6,7 \
      ... graspo:0.6.0-cuda13.2 \
      torchrun --nproc_per_node=4 debug_tp4_adapter_test.py \
        --model /workspace/models/Qwen3.5-9B \
        --data /workspace/data/data/elam_graspo_train.jsonl \
        --images /workspace/images \
        --start 0 --count 16 --rollout-group-size 8 \
        --max-new-tokens 128 --temperature 1.0
"""

import argparse, json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--rollout-group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def worker():
    args = parse_args()
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
            "model_path": args.model, "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "data": {
            "train_path": args.data,
            "max_prompt_length": 2048,
        },
        "training": {
            "rollout_group_size": args.rollout_group_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "optimize_prompt_batch_size": args.batch_size,
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

    from graspo.core.data import load_jsonl
    all_samples = load_jsonl(args.data)

    # Load the target samples
    end_idx = min(args.start + args.count, len(all_samples))
    test_samples = all_samples[args.start:end_idx]

    from graspo.backends.native_tp.tool_parser import parse_qwen_tool_completion

    # Process in batches of batch_size
    batch_starts = list(range(0, len(test_samples), args.batch_size))
    total_parse_err = 0
    total_completions = 0
    total_eos = 0
    total_truncated = 0

    if rank == 0:
        print(f"Testing samples {args.start}-{end_idx-1} ({len(test_samples)} total) "
              f"in batches of {args.batch_size}", flush=True)
        print(f"TP={world_size}, LoRA r=16, T={args.temperature}, G={args.rollout_group_size}", flush=True)

    for batch_idx, bstart in enumerate(batch_starts):
        bend = min(bstart + args.batch_size, len(test_samples))
        batch = test_samples[bstart:bend]

        generations = adapter.generate_sample_groups(
            samples=batch,
            rollout_group_size=args.rollout_group_size,
            max_new_tokens=args.max_new_tokens,
            max_prompt_length=2048,
            temperature=args.temperature,
            top_p=args.top_p,
            chat_template_kwargs={"enable_thinking": False},
        )

        batch_err = 0
        batch_comps = 0
        batch_eos = 0
        batch_trunc = 0

        for sidx, (sample, gen) in enumerate(zip(batch, generations)):
            actual_idx = args.start + bstart + sidx
            n_err = 0
            n_eos = 0
            n_trunc = 0
            previews = []
            for comp in gen.completions:
                parsed = parse_qwen_tool_completion(comp, tools=sample.tools)
                if parsed.parse_errors:
                    n_err += 1
                    if rank == 0 and n_err <= 2:
                        # Show first couple errors
                        preview = comp[:100].replace('\n', '\\n')
                        previews.append(f"ERR[{len(parsed.parse_errors)}]: {preview}")

            batch_err += n_err
            batch_comps += len(gen.completions)

            if rank == 0:
                flag = "ERR" if n_err > 0 else "OK"
                extra = ""
                if previews:
                    extra = " | " + " | ".join(previews)
                print(f"  [{actual_idx:3d}] {flag} err={n_err}/{len(gen.completions)}{extra}", flush=True)

        total_parse_err += batch_err
        total_completions += batch_comps

        if rank == 0:
            print(f"  Batch {batch_idx}: {batch_err}/{batch_comps} parse errors", flush=True)

        dist.barrier()

    if rank == 0:
        pct = 100 * total_parse_err / max(total_completions, 1)
        print(f"\n{'='*60}", flush=True)
        print(f"TOTAL: {total_parse_err}/{total_completions} parse errors ({pct:.1f}%)", flush=True)
        print(f"Samples: {args.start}-{end_idx-1}, TP={world_size}, LoRA r=16, T={args.temperature}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    worker()
