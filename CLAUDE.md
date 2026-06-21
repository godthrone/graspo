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
- `src/graspo/core/schema.py` — `NativeTPConfig.gpu_memory_utilization` (0-1,
  default 0.90) replaces the old inter-dependent
  `generation_micro_batch_size` / `rollout_kv_cache_max_reserved_fraction`.
  `TrainingConfig.rollout_group_size` is the sole algorithm parameter.
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
| `rollout_group_size` | 8 | Algorithm: completions per sample (do NOT change lightly) |
| `gpu_memory_utilization` | 0.90 | Resource: fraction of GPU memory for rollout (like vLLM's) |
| `empty_cache_after_rollout_split` | true | Frees PyTorch cache between Level 1 chunks |

## GPU Memory Estimation

`_kv_cache_batch_fits_budget` checks `max(kv_bytes, act_bytes × 1.5) ≤ total × utilization − reserved`.
The activation estimate accounts for full-attention QK^T materialisation
(`B × local_heads × L² × 2 bytes`) plus residual/MLP intermediates.
For Qwen3.5's hybrid architecture (8 full-attn + 24 linear-attn layers),
the full-attention layers dominate the prefill peak.

## Known Bug: mRoPE ndim=4 in `_apply_rope` / `_apply_rope_partial`

**Fixed in 0.6.0.** On PyTorch ≥ 2.12 with CUDA 13.2, `_qwen35_mrope_embeddings` returns
cos/sin with ndim=4 (shape `(1, B, S, head_dim)`).  The old code only handled
ndim=2 and ndim=3; ndim=4 fell through to `cos[position_ids].unsqueeze(1)`,
which double-indexed the already position-aware mRoPE cos tensor and caused a
dimension mismatch (`RuntimeError: size 926 ≠ 925 at dim 2`) on the second
decode step of multimodal generation.

**Fix** (`tensor_utils.py`): added `elif cos.ndim == 4: cos.squeeze(0).unsqueeze(1)`
to both `_apply_rope` and `_apply_rope_partial`.  This strips the mRoPE dimension
(always 1) and adds the head dimension, matching the ndim=3 pattern.  See
`tests/test_rope_ndim.py`.

## Deploy

```bash
# 228 training (A800 80GB × 4, TP=4)
docker run -d --name graspo_elam_v11_fk \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -v /home/zhangzy/models/Qwen3.5-9B:/workspace/models/Qwen3.5-9B:ro \
  -v /home/zhangzy/elam_v11_fk:/workspace/data \
  --ipc=host --shm-size=16g \
  graspo:latest \
  python -m graspo launch --config /workspace/data/config_docker.yaml

# CRITICAL: Use --gpus all + CUDA_VISIBLE_DEVICES, NOT --gpus '"device=..."'!
# The latter sets CUDA_VISIBLE_DEVICES=4,5,6,7 which makes
# torch.cuda.is_available() return False inside the container.
```

## Testing

```bash
python -m pytest tests/ -x -q
```
