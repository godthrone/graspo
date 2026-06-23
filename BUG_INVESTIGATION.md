# GRASPO Parse Err Bug Investigation

## Date: 2026-06-22/23

## STATUS: ROOT CAUSE FOUND ✅ — FIXES COMMITTED ✅

### Root Cause 1: Visual tower inv_freq (commit 4ca329e) — SECONDARY

The visual tower's `inv_freq` buffer was cast to bfloat16 by `.to(dtype=torch_dtype)`,
losing ~3 decimal digits.  Fix: recompute in float32 after `load_state_dict`.

### Root Cause 2: LoRA non-sharded matrix TP divergence (commit 87123ea) — PRIMARY

**Mechanism**: In TP-sharded decoder layers (shard_kind="rows" for q_proj/v_proj),
the `lora_a` matrix maps from full input dimension → should be **identical**
across all TP ranks.  However during backward, each rank computes a different
gradient for `lora_a` because:
1. `grad(lora_a) = sum over output dims of (lora_b^T @ dL/d(lora_out))`
2. `lora_b` is sharded (different per rank, correct)
3. `dL/d(lora_out)` is also partial (different per rank, correct)
4. Their product → different lora_a gradient per rank → **lora_a DIVERGES**

**Timeline**:
- Step 1: lora_b=0 (B_init=0) → lora_a gradient=0 → lora_a stays in sync
- Step 1: lora_b gets non-zero gradient → diverges (63/64 modules, expected)
- Step 2+: lora_b is now non-zero → lora_a gets different gradient per rank → DIVERGES (56/64 modules by step 3)

**Fix** (commit 87123ea): After each optimizer step, all-reduce (AVG) the
non-sharded LoRA matrix across TP ranks:
- `shard_kind="rows"/"out"`: all-reduce `lora_a` (maps from full input)
- `shard_kind="in"`: all-reduce `lora_b` (maps to full output)

**Verified**: lora_a divergence drops from 56/64 → 0/64 after fix.

---

## 1. Bug Description

**Symptom**: GRASPO 0.6.0 TP=4 LoRA training on Qwen3.5-9B multimodal (v12_fk_scenes data) produces
17-52% tool call parse errors per training step. Bad steps have ALL 64 sequences running to max 512
tokens without generating EOS. Pure HF transformers on the same samples shows 0/405 parse errors.

**Server**: 10.1.251.228:22022, GPUs 4-7 (A800 80GB × 4)

**Validation standard**: GRASPO with LoRA should produce 0% tool call parse errors, matching pure HF
transformers on the same 405-sample dataset with the same generation parameters.

---

## 2. All Exclusion Experiments Conducted

| # | Hypothesis | Test | Result | Verdict |
|---|-----------|------|--------|---------|
| 1 | mRoPE `cos.ndim==4` bug in `_apply_rope` | Checked code at `tensor_utils.py:402-448` | Already fixed (elif branch present) | ❌ Not root cause |
| 2 | `rope_deltas = None` RESET bug | Checked code at `modeling_hybrid.py:318-319` | Already fixed (reset removed) | ❌ Not root cause |
| 3 | `_causal_attention_mask` KV cache dimension | Checked code at `tensor_utils.py:466` | Already fixed (`-key_len:` truncation) | ❌ Not root cause |
| 4 | TP all-reduce silently disabled | Checked `tensor_utils.py:18-26` | Already fixed (globals consolidated) | ❌ Not root cause |
| 5 | LoRA cross-rank divergence causes parse_err | TP=4 training with LoRA sync fix vs control (8 steps each) | Control: 14/128 (10.9%), Sync: 13/128 (10.2%). No significant difference. | ❌ Not root cause |
| 6 | Samples inherently difficult | HF baseline on all 405 samples | 0/405 parse errors | ❌ Not sample-dependent |
| 7 | TP=1 base model generation correct | Manual forward pass TP=1 vs HF | 50/50 tokens match perfectly | ✅ Base model correct |
| 8 | TP=4 base model generation correct | Manual forward pass TP=4 vs HF | 50/50 tokens match perfectly | ✅ TP forward correct |
| 9 | Adapter with LoRA B_init=0 (no training) correct | TP=4 adapter test, 16 samples, T=1.0 | 0/128 parse errors | ✅ Pre-training adapter correct |
| 10 | Adapter model.forward() correct | Direct model call vs HF generate() | 32/32 tokens match perfectly | ✅ Model forward correct |
| 11 | Adapter tokenization matches manual | Compared input_ids from adapter vs manual | Identical (925 tokens) | ✅ Tokenization correct |
| 12 | Adapter image loading correct | Checked relative vs absolute paths | Both produce same pixel_values | ✅ Image loading correct |
| 13 | Hidden state divergence location | Per-layer hidden state comparison | **L0 maxdiff=4.31**, grows to 63.5 at L31 | 🔍 Divergence originates at layer 0 |
| 14 | Embedding weight match | Direct weight comparison | embed_tokens maxdiff=0.0, equal=True | ✅ Weights correct |
| 15 | Text embedding match | Compare text embed (before visual) | maxdiff=0.0 | ✅ Text embed perfect |
| 16 | **Visual tower output match** | Compare GRASPO vs HF visual features | **maxdiff=3008.0! GRASPO near-zero vs HF normal** | ❌ **ROOT CAUSE** |

---

## 3. Root Cause: Visual Tower Produces Wrong Features

### Evidence

**Text embedding**: GRASPO and HF produce IDENTICAL text embeddings (maxdiff=0.000000).
**Visual tower output**: GRASPO visual features are near-zero while HF produces normal values (maxdiff=3008).

Sample comparison at image token positions 470-472:
```
HF:  [1.25, 3.67, 0.88, -5.66, 0.18]    ← normal feature values
G:   [-0.12, 0.11, 0.15, 0.09, 0.09]    ← near-zero (wrong!)
```

This causes wrong multimodal understanding → model generates wrong tool calls →
with T=1.0 random sampling → enters gibberish loop → runs to max 512 tokens without EOS.

### Propagation chain

1. Visual tower produces wrong features
2. Image tokens in embedding get wrong values
3. Layer 0 hidden state already shows maxdiff=4.31
4. Diff propagates through 32 layers, growing to maxdiff=63.5 at layer 31
5. KV cache accumulates errors during decode
6. By step 32 of greedy decode, argmax flips
7. With T=1.0 sampling, small logit differences amplified → gibberish
8. Entire batch collapses (all 64 sequences → 512 tokens, no EOS)

---

## 4. Next Step: Debug Visual Tower Weight Loading

The visual tower is loaded in: `src/graspo/backends/native_tp/models/qwen/modeling.py`
Function: `_build_qwen35_visual_tower()`

File to check: `/workspace/graspo/src/graspo/backends/native_tp/models/qwen/modeling.py`

The visual tower's forward method is: `g.visual.forward(pixel_values, grid_thw=image_grid_thw).last_hidden_state`

Need to verify:
1. Visual tower weights are correctly loaded from safetensors
2. The weight keys/prefixes match HF's visual tower weight names
3. The visual forward pass matches HF's implementation
4. `pixel_values` dtype/format is compatible (GRASPO receives float32 from processor, model uses bfloat16)

### Quick test to run next

```python
# Compare a few visual tower weight keys
hf_visual_weights = {name: param for name, param in hf.model.visual.named_parameters()}
g_visual_weights = {name: param for name, param in g.visual.named_parameters()}

# Check key sets and values
print("Weight key overlap:", set(hf_visual_weights.keys()) == set(g_visual_weights.keys()))
for name in sorted(hf_visual_weights.keys())[:5]:
    hf_w = hf_visual_weights[name]
    g_w = g_visual_weights[name]
    diff = (hf_w - g_w).abs().max().item()
    print(f"  {name}: shapes HF={tuple(hf_w.shape)} G={tuple(g_w.shape)} maxdiff={diff:.4f}")
```

---

## 5. Key Files Modified/Created During Investigation

### Test scripts (on 228 at `/home/zhangzy/` and in `/workspace/graspo/scripts/`)

| Script | Purpose |
|--------|---------|
| `debug_tp1_compare.py` | TP=1 GRASPO vs HF step-by-step comparison (single sample) |
| `debug_tp1_batch.py` | TP=1 GRASPO vs HF with batched samples |
| `debug_tp4_compare.py` | TP=4 GRASPO vs HF comparison |
| `debug_tp4_adapter_test.py` | TP=4 adapter + LoRA B_init=0 + T=1.0 + batch test |
| `debug_lora_divergence.py` | Track LoRA b-norm cross-rank divergence over optimizer steps |
| `debug_lora_opt_step.py` | Dump LoRA weights before/after optimizer step |
| `debug_train_with_lora_track.py` | Trainer with LoRA tracking monkey-patched |
| `debug_sync_test.py` | LoRA b sync fix vs control comparison |
| `debug_pipeline_trace.py` | Instrument adapter pipeline: log all intermediate values |
| `debug_per_layer_hidden.py` | Per-layer hidden state comparison (GRASPO vs HF) |
| `debug_embed_diff.py` | Compare embedding + visual tower outputs |
| `debug_kv_trace.py` | Per-layer KV cache + first-token logit comparison |
| `hf_full_baseline.py` | HF baseline on all 405 samples |

### Core code examined

| File | Lines | What |
|------|-------|------|
| `tensor_utils.py` | 402-448 | `_apply_rope` / `_apply_rope_partial` (mRoPE ndim=4 fix) |
| `tensor_utils.py` | 451-467 | `_causal_attention_mask` (KV cache truncation) |
| `modeling_hybrid.py` | 289-330 | `compute_multimodal_position_ids` (rope_deltas) |
| `modeling_hybrid.py` | 148-246 | `forward()` / `_forward_hidden()` |
| `modeling_hybrid.py` | 248-287 | `embed_inputs()` — visual feature insertion |
| `layers.py` | 287-354 | `TensorParallelQwen35FullAttention.forward()` |
| `layers.py` | 184-212 | full_attention LoRA weight setup with `_select_head_rows` |
| `layers.py` | 385-450 | linear_attention LoRA weight setup |
| `lora.py` | 126-263 | `LoRALinear` class — init, from_hf, forward |
| `adapter.py` | 900-1008 | `_generate_multimodal_with_kv_cache` |
| `adapter.py` | 1921-2051 | `train_batch` — optimizer step |
| `adapter.py` | 2883-2923 | `_encode_multimodal_rows` |
| `tensor_utils.py` | 648-657 | `_next_token_from_logits` (temperature handling) |

---

## 6. Position Tracking

**Current state**: Root cause identified — visual tower produces wrong features.
**Next action**: Debug `_build_qwen35_visual_tower()` weight loading.
**Hypothesis**: Visual tower weights not correctly loaded from safetensors (wrong keys, missing prefixes, or dtype issues).
