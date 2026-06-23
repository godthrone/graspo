# GRASPO Parse Err Bug Investigation

## Date: 2026-06-22/23

## STATUS: Two TP precision bugs fixed. Remaining ~30% parse errors NOT from TP — likely training dynamics.

---

## Confirmed & Fixed

### Fix 1: Visual tower inv_freq bfloat16 (commit 4ca329e)

`_build_qwen35_visual_tower()` 中的 `.to(dtype=bfloat16)` 将 `inv_freq` buffer
从 float32 静默转换为 bfloat16，丢失 ~3 位精度。27 层 ViT 累积后 visual output
与 HF 差异 maxdiff=3008，pooler output 差异 ~1%。

**修复**：在 `load_state_dict` 后用 float32 重新计算 `inv_freq`。
修复后 visual tower 与 HF 完全一致（maxdiff=0.0）。

**文件**：`src/graspo/backends/native_tp/models/qwen/modeling.py:204-214`

### Fix 2: LoRA 非切分矩阵 TP rank 间梯度发散 (commit 747f867)

在 TP-sharded decoder 层中（`shard_kind="rows"`）：
- `lora_a` 映射完整输入维度 → 应该在所有 rank 上**完全相同**
- 但 backward 时每个 rank 计算出**不同的** `lora_a` 梯度
- Step 1: lora_b=0 (B_init=0) → grad(lora_a)=0 → lora_a 安全
- Step 2+: lora_b 非零且各 rank 不同 → grad(lora_a) 各 rank 不同 → lora_a 发散（56/64 模块 by step 3）

**修复**：在 `optimizer.step()` **之前** all-reduce `lora_a.grad`（AVG），
使所有 rank 计算相同的 weight update，保持 weights 和 Adam state 同步。

**文件**：
- `src/graspo/backends/native_tp/models/qwen/lora.py` — `_sync_nonsharded_lora_grads()`
- `src/graspo/backends/native_tp/models/qwen/adapter.py` — 在 `loss.backward()` 后调用

---

## Remaining Parse Errors

### 关键诊断结果：logits 跨 rank 完全一致

| 测试 | 结果 |
|------|------|
| 训练前，TP=4，logits cross-rank maxdiff | **0.0, 0.0, 0.0** |
| 训练后，TP=4，logits cross-rank maxdiff | **0.0, 0.0, 0.0** |

**→ TP 切分、all-reduce、LoRA sync 全部正确。Parse error 不是 TP 数值 bug。**

### Parse error 数据

| 配置 | Parse errors | 备注 |
|------|-------------|------|
| B_init=0, G=8, TP=4, T=1.0 | **0%** (0/128) | 基础模型完美 |
| G=4, B=4, empty_cache=true, 5步 | **5%** (4/80) | 小 batch，大部分步零错误 |
| G=8, B=8, empty_cache=false, 7步 (no fix) | **34%** (154/448) | baseline |
| G=8, B=8, empty_cache=false, 7步 (with fix 2) | **27-31%** | 改善有限 |

### 剩余问题的可能方向

1. **训练动力学问题**（最可能）：GRPO 训练让模型在某些样本上学到了错误的 tool call
   格式。B_init=0（基础模型）完美，训练后的模型变差 → 不是 TP 的问题，是训练信号
   或优化过程的问题。

2. **G/B 增大效应**：更大的 batch/group 意味着更多样的样本被一起训练，可能某些
   样本的 reward signal 冲突导致学习困难。

3. **需要对比**：TP=1 训练是否也有类似水平的 parse error？如果 TP=1 同样有 ~30%，
   则确认不是 TP 问题。

---

## 所有排除的假设

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
| 16 | Visual tower output wrong | ✅ 已 fix (inv_freq float32) |
| 17 | LoRA lora_a weight divergence | ✅ 已 fix (gradient sync) |
| 18 | Logits 跨 rank 不一致 | ❌ 排除 — logits 完全一致 |

---

## Commits

```
747f867 fix: sync LoRA gradients before optimizer.step() instead of weights after
87123ea fix: sync non-sharded LoRA matrix across TP ranks after optimizer step
4ca329e fix: recompute visual tower inv_freq in float32 to match HF precision
```

## Docker Images (on 228)

- `graspo:0.6.0-inv-freq-fix` — Fix 1 only
- `graspo:0.6.0-full-fix` — Fix 1 + Fix 2 (weight sync, deprecated)
- `graspo:0.6.0-grad-sync` — Fix 1 + Fix 2 (gradient sync, recommended)
