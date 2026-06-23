# GRASPO Parse Err Bug Investigation

## Date: 2026-06-22/23 (ongoing)

## STATUS: Two fixes committed, parse errors scale with batch/group size

### Fix 1: Visual tower inv_freq bfloat16 (commit 4ca329e) — CONFIRMED ✅

`_build_qwen35_visual_tower()` 中的 `.to(dtype=bfloat16)` 将 `inv_freq` buffer
从 float32 转换为 bfloat16，丢失 ~3 位精度。27 层 ViT 累积后 visual output 与 HF
差异 maxdiff=3008。

修复后在 `load_state_dict` 后重新用 float32 计算 `inv_freq`。Visual tower
现在与 HF 完全一致（maxdiff=0.0）。

### Fix 2: LoRA 非切分矩阵 TP rank 间发散 (commit 87123ea) — CONFIRMED ✅

在 TP-sharded decoder 层中（q_proj/v_proj 使用 `shard_kind="rows"`）：
- `lora_a` 映射完整输入维度 → 应该在所有 rank 上**完全相同**
- 但 backward 时每个 rank 计算出不同的 `lora_a` 梯度（因为 lora_b 和 dL/d(lora_out) 都是 per-rank partial 的）
- 3 步训练后 56/64 个 lora_a 模块发散

修复后 lora_a 发散降为 0/64。

---

## Remaining Issue: Parse errors scale with batch/group size

### 参数解释

- **`rollout_group_size`**（简称 G）：每个 prompt 生成多少个 completion。默认 8。
- **`optimize_prompt_batch_size`**（简称 B）：每次 optimizer step 用多少个不同 prompt。默认 8。
- 每步产生 G × B 个 completion。
- `forward_batch_size`：每次 forward 的 micro-batch 大小。

### 观测数据

| 配置 | Parse errors | 备注 |
|------|-------------|------|
| B_init=0, G=8, TP=4, T=1.0 | **0%** (0/128) | 实验 #9，无训练 |
| G=4, B=4, empty_cache=true | **5%** (4/80，5步) | 4/5 步零错误 |
| G=8, B=8, empty_cache=false | **31%** (140/448，7步) | 大部分步都有错误 |
| G=8, B=8, empty_cache=false (无 fix) | **34%** (154/448，7步) | fix 前 baseline |

### 关键线索

1. 问题随 G 和 B 增大而加重（不是 cache 问题）
2. B_init=0 时完全正确 → forward pass 本身没问题
3. LoRA 训练后出现 parse error → optimizer step 引入问题

### 当前假设：Adam optimizer state 跨 rank 发散

当前的 weight sync fix 在 `optimizer.step()` **之后** sync lora_a 权重，
但 Adam 的 `exp_avg` 和 `exp_avg_sq` **没有** sync。

G/B 越大 → 每步 optimizer sub-step 越多 → Adam state 发散越严重 → 问题越明显。

**正确的 fix**：在 `optimizer.step()` **之前** all-reduce lora_a 的 **gradient**
（而非 step 之后 sync 权重）。这样所有 rank 计算相同的 update，Adam state
保持同步。

### 之前排除的假设

| # | 假设 | 结果 |
|---|------|------|
| 1 | mRoPE ndim=4 bug | ✅ 已 fix (0.6.0) |
| 2 | rope_deltas RESET | ✅ 已 fix (0.6.0) |
| 3 | causal_attention_mask KV cache | ✅ 已 fix (0.6.0) |
| 4 | TP all-reduce silently disabled | ✅ 已 fix (0.6.0) |
| 5 | LoRA lora_b sync (B_init=0) | ❌ 无效（只 sync 了 lora_b） |
| 6 | Sample-dependent | ❌ HF baseline: 0/405 |
| 7 | TP=1 base model | ✅ 正确 |
| 8 | TP=4 base model | ✅ 正确 |
| 9 | TP=4 B_init=0 (no training) | ✅ 0/128 错误 |
| 16 | Visual tower output wrong | ✅ 已 fix (inv_freq) |
| 17 | LoRA lora_a weight divergence | ✅ 已 fix (weight sync) |
| 18 | empty_cache 导致 KV cache 问题 | ❓ 用户认为非根因 |

---

## Files Modified

| File | Commit | Change |
|------|--------|--------|
| `models/qwen/modeling.py` | 4ca329e | inv_freq float32 fix |
| `models/qwen/lora.py` | 87123ea | `_sync_nonsharded_lora_weights()` |
| `models/qwen/adapter.py` | 87123ea | Call sync after optimizer.step() |

## Docker Images (on 228)

- `graspo:0.6.0-inv-freq-fix` — Fix 1 only
- `graspo:0.6.0-full-fix` — Fix 1 + Fix 2
