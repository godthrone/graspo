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
  controls rollout batch sizing. `TrainingConfig.rollout_group_size` (default 8)
  and `TrainingConfig.optimize_prompt_batch_size` (default 8) are the core
  algorithm parameters.
- `src/graspo/backends/native_tp/trainer.py` — Multimodal samples are chunked in
  CPU-friendly encoding batches (`_MAX_MULTIMODAL_SAMPLES_PER_CALL = 16`).
  `_is_pure_tool_call_task()` guards JSON-marker debug counts.
- `src/graspo/backends/native_tp/logger.py` — Same guard in
  `group_debug_summary()`.
- `src/graspo/backends/native_tp/models/qwen/adapter.py` — Multimodal Level 1
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

## TP LoRA Gradient Sync

In TP mode, all ranks process the same data and compute partial gradients.
The full gradient is the SUM of partial gradients (not AVG as in DDP).
`lora_a` (input projection) is non-sharded → gradient must be all-reduced
with SUM across TP ranks.  `lora_b` (output projection) is sharded along
the output dimension → no sync needed for the sharded dimension.

## Known Bug: mRoPE ndim=4 in `_apply_rope` / `_apply_rope_partial`

**Fixed in 0.6.0.** See `tests/test_rope_ndim.py`.

## Known Bug: TP all-reduce silently disabled

**Fixed in 0.6.0.** `_TENSOR_PARALLEL_GROUP` and `_TENSOR_PARALLEL_SIZE` were
duplicated in `adapter.py` and `tensor_utils.py`.  `_set_tensor_parallel_group()`
in adapter.py set the adapter.py copies; `_all_reduce_tp()` in tensor_utils.py
read the tensor_utils.py copies — which were always `None`/`1`.  Every TP>1
training run was silently running without cross-rank reduction.

**Fix**: moved globals and `_set_tensor_parallel_group()` to `tensor_utils.py`.
adapter.py now imports them from tensor_utils.

## Known Bug: `_causal_attention_mask` KV-cache dimension mismatch

**Fixed in 0.6.0.**  `tensor_utils.py:_causal_attention_mask` used the full
`attention_mask` as `key_mask` (`attention_mask[:, None, None, :]`).  During
incremental KV-cache decode the attention mask grows longer than `key_len`,
causing a broadcast mismatch.

**Fix**: truncate `key_mask` to the last `key_len` positions:
`attention_mask[:, None, None, -key_len:]`.

## Testing

```bash
python -m pytest tests/ -x -q
ruff check src/ tests/
mypy src/
```