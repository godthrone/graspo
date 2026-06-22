# CLAUDE.md — GRASPO

GRPO-style LoRA trainer for structured-output RL (JSON, tool calls) with native
tensor parallel (TP) / pipeline parallel (PP) backends.

## Key Architecture

- `src/graspo/core/compare.py` — `dict_compare_score` returns `CompareResult`
  (dataclass with `.dcs`, `.base_dcs`, `.all_right`).  Numeric leaves
  participate in `dcs` (gradient signal) but are stripped for
  `base_dcs`/`all_right` (structural gating).
- `src/graspo/core/reward.py` — `GraspoReward.score()` and `.score_parsed()` use
  `result.all_right` for gating.  `RewardResult` includes `base_content_score`.
  Supports multi-target alternatives; best-scoring target wins.
- `src/graspo/core/schema.py` — `NativeTPConfig.forward_batch_size` (default 8)
  controls rollout batch sizing; old `gpu_memory_utilization`,
  `generation_micro_batch_size`, and `rollout_kv_cache_max_reserved_fraction`
  are removed. `TrainingConfig.rollout_group_size` (default 8) and
  `TrainingConfig.optimize_prompt_batch_size` (default 8) are the core
  algorithm parameters.
- `src/graspo/backends/native_tp/trainer.py` — Multimodal samples are chunked in
  CPU-friendly encoding batches (`_MAX_MULTIMODAL_SAMPLES_PER_CALL = 16`).
  `_is_pure_tool_call_task()` guards JSON-marker debug counts.
- `src/graspo/backends/native_tp/logger.py` — Same guard in
  `group_debug_summary()`.
- `src/graspo/backends/native_tp/qwen_tp_adapter.py` — Multimodal Level 1
  (prompt chunk) + Level 2 (micro-batch) generation with offset-based
  multimodal input slicing for heterogeneous image counts.  Budget estimate
  uses `budget_prompt_len = max(prompt_len, max_prompt_length)` with a 1.5×
  safety factor to prevent OOM at 8K prompt lengths.  `prompt_chunk` is capped
  at 3 to prevent over-ambitious batching.

## Core Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `rollout_group_size` | 8 | Algorithm: completions per prompt (do NOT change lightly) |
| `optimize_prompt_batch_size` | 8 | Prompts per optimize step; replay threshold = G × B |
| `forward_batch_size` | 8 | Rollout micro-batch size (GPU memory trade-off) |
| `empty_cache_after_rollout_split` | true | Frees PyTorch cache between Level 1 chunks |
| `empty_cache_before_train` | false | Frees PyTorch cache before optimize step |

## GPU Memory Estimation

`_kv_cache_batch_fits_budget` checks `max(kv_bytes, act_bytes × 1.5) ≤ total × utilization − reserved`.
The activation estimate accounts for full-attention QK^T materialisation
(`B × local_heads × L² × 2 bytes`) plus residual/MLP intermediates.
For Qwen3.5's hybrid architecture (8 full-attn + 24 linear-attn layers),
the full-attention layers dominate the prefill peak.

## Known Bug: mRoPE ndim=4 in `_apply_rope` / `_apply_rope_partial`

**Fixed in 0.6.0.** See `tests/test_rope_ndim.py`.

## Known Bug: TP all-reduce silently disabled

**Fixed in 0.6.0.** `_TENSOR_PARALLEL_GROUP` and `_TENSOR_PARALLEL_SIZE` were
duplicated in `adapter.py` and `tensor_utils.py`.  `_set_tensor_parallel_group()`
in adapter.py set the adapter.py copies; `_all_reduce_tp()` in tensor_utils.py
read the tensor_utils.py copies — which were always `None`/`1`.  Every TP>1
training run was silently running without cross-rank reduction, producing
gibberish.

**Fix**: moved globals and `_set_tensor_parallel_group()` to `tensor_utils.py`.
adapter.py now imports them from tensor_utils.  See `tests/test_rope_ndim.py`
(`TestTPGlobals`).

## Deploy

```bash
# 228 training (A800 80GB × 4, TP=4, GPUs 4-7)
docker run -d --name graspo_elam_v12_fk \
  -e NVIDIA_VISIBLE_DEVICES=4,5,6,7 \
  -v /home/zhangzy/models/Qwen3.5-9B:/workspace/models/Qwen3.5-9B:ro \
  -v /home/zhangzy/elam_v12_fk:/workspace/data \
  -v /home/zhangzy/elam_v12_fk/images:/workspace/images:ro \
  --ipc=host --shm-size=16g \
  graspo:0.6.0-cuda13.2 \
  python -m graspo launch --config /workspace/data/data/config_docker.yaml

# CRITICAL: Docker 29+ uses CDI mode. Do NOT use --gpus all or --gpus device=X!
# Only use -e NVIDIA_VISIBLE_DEVICES=X,Y,Z to select GPUs.
# The env var alone works because nvidia-container-toolkit CDI mode reads it directly.
```

## Testing

```bash
python -m pytest tests/ -x -q
```
