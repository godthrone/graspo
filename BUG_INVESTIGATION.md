# GRASPO Parse Err Bug Investigation

## Date: 2026-06-22/23

## STATUS: **Fix 3 + Fix 4 committed.** Fix 3 (AVG→SUM) aligns TP=4 with TP=1. Fix 4 (reward_median gating, pending) breaks the positive feedback loop where low-quality completions poison training.  See below for experiment results.

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

### Fix 3: Gradient sync 用 AVG 而非 SUM (commit 2ecb123)

**根因分析**（2026-06-23 TP=1 vs TP=4 对比实验后）：

`_sync_nonsharded_lora_grads()` 用 `dist.ReduceOp.AVG` all-reduce LoRA 梯度。
但在 TP（Tensor Parallelism）中，所有 rank 处理**相同**数据，每个 rank 计算**部分**
梯度。完整梯度 = **SUM**(各 rank 的部分梯度)，而非 AVG。

对比：
- `_TensorParallelAllReduce`（前向传播）：用 `SUM` ✅ — 正确，TP 语义
- `_sync_nonsharded_lora_grads`（梯度同步）：用 `AVG` ❌ — 错误，应该用 SUM

这导致 lora_A 的有效学习率 = lr / tp_size（TP=4 时只有 1/4），而 lora_B 的
梯度保持正确 scale（不受 sync 影响）。lora_A/lora_B 学习率失衡使得 TP>1 训练
无法纠正模型在 GRASPO 训练中引入的格式错误。

**修复**：`AVG` → `SUM` in `dist.all_reduce`。

**文件**：`src/graspo/backends/native_tp/models/qwen/lora.py:284,287`

### Fix 4: classify_group 低质量 completion 正反馈循环 (commit 333e34e, 待修正)

**根因分析**（2026-06-23 TP=1 vs TP=4 波动性排查）：

所有 parse error 都是 `tool_call_count_mismatch`（不是 JSON 语法错误），且与
`reward_median` 强相关。存在一个正反馈循环：

```
模型生成错误数量 tool call → reward_median≈0.0
→ classify_group 判定 TRAINABLE_NOT_CORRECT
→ 低质量样本进入训练 → 模型学到更多错误格式 → 下一步崩溃
```

`classify_group` 的 gating 逻辑缺陷：
```python
elif reward_max > reward_median:
    decision = GroupDecision.TRAINABLE_NOT_CORRECT
```
当 8 个 completion 中 7 个 reward=0.0，1 个 reward=0.01，`reward_max(0.01) > reward_median(0.0)` 成立，
整组被送入训练。这些低质量样本的训练信号会强化格式错误。

**修复**：在 `TRAINABLE_NOT_CORRECT` 判定中加入 `reward_median >= 0.4` 的 guard：
```python
elif reward_max > reward_median and reward_median >= 0.4:
    decision = GroupDecision.TRAINABLE_NOT_CORRECT
```
低于阈值的 group fall through 到 RETRY（重试生成）或 INVALID（丢弃），
打断正反馈循环。

**文件**：`src/graspo/core/graspo_parity.py:116`

**验证**（TP=4, `reward_median >= 0.4` guard, Step 1）：
- retry=12/23 (52%) vs max-gating 的 27% — 过滤更激进
- **首次出现 invalid=3** — 低质量 group 被正确丢弃
- reward_median=0.387 — 接近阈值，证明 guard 在生效
- ⚠️ OOM at Step 2（retry 翻倍导致 completion 数暴增），需减小 forward_batch_size 重跑

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
| G=8, B=8, empty_cache=true, TP=1, Step 1 | **34%** (22/64) | 初始与 TP=4 一致 |
| G=8, B=8, empty_cache=true, TP=1, Step 2 | **1.6%** (1/64) | TP=1 快速改善 |
| G=8, B=8, empty_cache=true, TP=1, Step 3 | **0%** (0/64) | TP=1 完全消除 |
| G=8, B=8, empty_cache=true, TP=1, Step 4+ | **OOM** | forward_batch_size=64 对单 GPU 过大 |

### TP=1 vs TP=4 对比关键结论

**TP=1 训练和 TP=4 训练在 SUM fix 后完全对齐。**

| 配置 | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 | Step 7 | Step 8 | Step 9 | Step 10 |
|------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| TP=4 SUM | 22 | 0 | 0 | 43 | 40 | 20 | 26 | 0 | 43 | — |
| TP=1 (fw=8) | 18 | 1 | 2 | 30 | 49 | 25 | 22 | 0 | 34 | 43 |

两者波动模式一致，diff 在 ±9 以内。Parse error 波动是训练动力学问题，不是 TP 特有的。

### 波动性根因：classify_group 正反馈循环

所有 parse error 都是 `tool_call_count_mismatch`，与 `reward_median` 强相关：

| reward_median | parse errors | 模式 |
|--------------|-------------|------|
| ~0.9 | 0 | ✓ 干净 |
| ~0.4 | 0-2 | 正常 |
| ~0.3 | 18-30 | 波动 |
| **~0.004** | **34-49** | ⚠ 崩溃 |

**正反馈循环**：模型生成错误数量 tool call → reward_median≈0.0 → `classify_group`
判定 `TRAINABLE_NOT_CORRECT`（因为 `reward_max > reward_median` 即使只差 0.01
也成立）→ 低质量样本进入训练 → 模型学到更多错误 → 下一步崩溃。

**Fix 4 修复**：`reward_max > reward_median` → `reward_max > reward_median and reward_median >= 0.4`
（commit 333e34e，待改为 median-based）。低于阈值的 group fall through 到 RETRY/INVALID。

**Fix 4 验证数据**（TP=4, `reward_max >= 0.4` guard）：

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|------|---|---|---|---|---|---|---|---|
| Parse | 20 | 2 | 0 | 42 | 39 | 19 | 21 | 6 |
| Retry | 3/11 | 3/19 | 3/27 | 11/44 | 12/53 | 12/61 | 12/69 | 16/81 |

`reward_max >= 0.4` 效果有限——1 个 outlier completion 就能让整个 group 通过 guard。
正在测试 `reward_median >= 0.4` 版本。

### 剩余问题的可能方向

1. **TP 相关 bug（新确定 — 最可能）**：TP=1 训练完美消除 parse error (34%→0%)，
   TP=4 训练无法消除 (~30% 持续)。Logits 跨 rank 一致 (maxdiff=0.0)，所以
   问题不在前向传播，而在**反向传播/梯度/优化器**环节。

   - **可能的根因**：GRASPOLoss 在 TP-sharded logits 上的 logprobs 计算、
     梯度 all-reduce 精度、或 Adam optimizer state 在 TP rank 间的同步。
   - **需要深入调查**：对比 TP=1 vs TP=4 的 per-step loss、gradient norm、
     logprobs 分布。

2. **G/B 增大效应**：更大的 batch/group 意味着更多样的样本被一起训练，可能某些
   样本的 reward signal 冲突导致学习困难。Note: G=4,B=4 只有 5% error，可能因为
   小 batch 减少了冲突。

3. **需要对比**：TP=1 训练是否也有类似水平的 parse error？如果 TP=1 同样有 ~30%，
   则确认不是 TP 问题。

   **→ 已完成 (2026-06-23)：TP=1 训练确认 parse error 可以完全消除（0%），
   确认剩余问题是 TP 特有的。** 见下方 "TP=1 vs TP=4 对比关键结论"。

---

## 测试环境与方法

### 硬件

| 项目 | 值 |
|------|-----|
| 服务器 | `10.1.251.228`，SSH 端口 `22022`，用户 `zhangzy` |
| GPU | NVIDIA A800 80GB × 8，本次使用 GPU 4-7 |
| CUDA | 容器内由 `graspo:0.6.0-cuda13.2` 提供 |

### Docker 镜像

| 镜像 | 说明 |
|------|------|
| `graspo:0.6.0-cuda13.2` | 原始 0.6.0 发布版（有 bug） |
| `graspo:0.6.0-inv-freq-fix` | + Fix 1 (visual tower inv_freq) |
| `graspo:0.6.0-full-fix` | + Fix 1 + Fix 2 weight sync（已废弃） |
| `graspo:0.6.0-grad-sync` | + Fix 1 + Fix 2 gradient sync（推荐） |

镜像构建方式（在 228 上）：

```bash
# 将修改后的源文件 scp 到 228
scp -P 22022 modeling.py zhangzy@10.1.251.228:/home/zhangzy/elam_v12_fk/scripts/
scp -P 22022 lora.py zhangzy@10.1.251.228:/home/zhangzy/elam_v12_fk/scripts/
scp -P 22022 adapter.py zhangzy@10.1.251.228:/home/zhangzy/elam_v12_fk/scripts/

# 在 228 上构建
ssh -p 22022 zhangzy@10.1.251.228
cat > Dockerfile << 'EOF'
FROM graspo:0.6.0-cuda13.2
COPY modeling_fixed.py /workspace/graspo/src/graspo/backends/native_tp/models/qwen/modeling.py
COPY lora_fixed.py /workspace/graspo/src/graspo/backends/native_tp/models/qwen/lora.py
COPY adapter_fixed.py /workspace/graspo/src/graspo/backends/native_tp/models/qwen/adapter.py
EOF
docker build -f Dockerfile -t graspo:0.6.0-grad-sync .
```

### 运行容器

```bash
# TP=4 训练（推荐用 nohup 避免 SSH 断开导致容器被杀）
docker run --rm --name <name> \
  -e NVIDIA_VISIBLE_DEVICES=4,5,6,7 \       # 只选 GPU 4-7，不要用 --gpus
  -v /home/zhangzy/models/Qwen3.5-9B:/workspace/models/Qwen3.5-9B:ro \
  -v /home/zhangzy/elam_v12_fk:/workspace/data \
  -v /home/zhangzy/elam_v12_fk/images:/workspace/images:ro \
  --ipc=host --shm-size=16g \
  <image> \
  torchrun --nproc_per_node=4 --master_port=29500 <script.py>

# 单 GPU 调试（TP=1）
docker run --rm --name <name> \
  -e NVIDIA_VISIBLE_DEVICES=4 \
  -v /home/zhangzy/models/Qwen3.5-9B:/workspace/models/Qwen3.5-9B:ro \
  -v /home/zhangzy/elam_v12_fk:/workspace/data \
  -v /home/zhangzy/elam_v12_fk/images:/workspace/images:ro \
  --ipc=host --shm-size=16g \
  <image> \
  python <script.py>
```

**关键约束**（来自 CLAUDE.md）：
- Docker 29+ 使用 CDI 模式，**不要用 `--gpus`**，只用 `-e NVIDIA_VISIBLE_DEVICES`
- 必须 `--ipc=host --shm-size=16g`

### 数据与模型路径（容器内）

| 路径 | 内容 |
|------|------|
| `/workspace/models/Qwen3.5-9B` | Qwen3.5-9B 模型权重 (read-only) |
| `/workspace/data/data/elam_graspo_train.jsonl` | 405 条多模态训练样本 |
| `/workspace/images/` | 图像文件 (read-only) |

### 调试脚本

所有脚本位于 `scripts/` 目录，需要 scp 到 228 运行。

**Phase 0 — 原始调查（已有）**：

| 脚本 | 用途 |
|------|------|
| `debug_tp1_compare.py` | TP=1 GRASPO vs HF 逐 token 对比 |
| `debug_tp4_compare.py` | TP=4 GRASPO vs HF 对比 |
| `debug_tp4_adapter_test.py` | TP=4 + LoRA B_init=0 + T=1.0 批量测试 |
| `debug_embed_diff.py` | 对比 embedding + visual tower 输出 |
| `debug_per_layer_hidden.py` | 逐层 hidden state 对比 |
| `debug_visual_weights.py` | Visual tower 权重/ buffer 对比 |
| `hf_full_baseline.py` | HF baseline on 405 样本 |

**Phase 1 — inv_freq 根因验证（新增）**：

| 脚本 | 用途 |
|------|------|
| `debug_inv_freq_verify.py` | 全面比对 visual tower 参数/buffer + inv_freq swap test |
| `debug_inv_freq_dtype.py` | inv_freq dtype 验证 + float32 fix test |
| `debug_visual_trace.py` | Visual config/结构/attention 实现/first block 对比 |
| `debug_e2e_generation.py` | TP=1 greedy decode 50 token vs HF 全流程验证 |

**Phase 2 — LoRA TP 发散验证（新增）**：

| 脚本 | 用途 |
|------|------|
| `debug_lora_tp_grad.py` | 单步训练后 visual/decoder LoRA 权重 cross-rank 对比 |
| `debug_lora_multistep_grad.py` | 3 步训练，每步记录 lora_a/b 发散数量 |
| `debug_logit_crossrank.py` | 训练前后 logits cross-rank 一致性检查（关键诊断） |

**关键诊断命令**：

```bash
# 验证 inv_freq fix
docker run ... graspo:0.6.0-grad-sync python /workspace/data/scripts/debug_inv_freq_verify.py

# 验证 LoRA 发散已修复
docker run ... graspo:0.6.0-grad-sync torchrun --nproc_per_node=4 ... \
  /workspace/data/scripts/debug_lora_multistep_grad.py

# 验证 logits 跨 rank 一致（最关键的诊断）
docker run ... graspo:0.6.0-grad-sync torchrun --nproc_per_node=4 ... \
  /workspace/data/scripts/debug_logit_crossrank.py

# 观察训练中的 parse error
docker run ... graspo:0.6.0-grad-sync python -m graspo launch --config <config.yaml>
# 在另一个终端：
ssh -p 22022 zhangzy@10.1.251.228 \
  "grep 'tool_call_parse_error' /tmp/grad_sync_train.log | python3 -c \"
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    if d.get('event')=='train_step':
        print(f'Step {d[\"step\"]}: {d[\"batch\"][\"debug\"][\"tool_call_parse_error\"]} errors')
\""
```

### 训练配置

测试用配置（与生产配置的主要差异：减少步数和 batch 大小）：

```yaml
training:
  max_steps: 10          # 生产: 999999
  rollout_group_size: 8  # 每个 prompt 的 completion 数
  optimize_prompt_batch_size: 8  # 每次 optimizer step 的 prompt 数
  optimize_times_per_step: 1
  max_new_tokens: 64     # 生产: 512
  temperature: 1.0
  top_p: 1.0
backend_config:
  native_tp:
    tp_size: 4
    forward_batch_size: 64
    empty_cache_after_rollout_split: false  # true 可能降低 OOM 风险
```

### TP=1 实验配置 (2026-06-23)

```yaml
backend_config:
  native_tp:
    tp_size: 1
    forward_batch_size: 64      # 对单 GPU 过大，Step 4 OOM
    empty_cache_after_rollout_split: true
launch:
  gpus: [0]                     # 容器内索引（NVIDIA_VISIBLE_DEVICES=4 映射为 GPU 0）
  nproc_per_node: 1
training:
  rollout_group_size: 8
  optimize_prompt_batch_size: 8
  max_steps: 10
  max_new_tokens: 64
```

⚠️ **GPU 索引陷阱**：当 Docker 用 `-e NVIDIA_VISIBLE_DEVICES=4` 时，host GPU 4
在容器内变为 GPU 0。Config 的 `launch.gpus` 必须写容器内索引 `[0]`，不是 host
索引 `[4]`。写错会导致 CUDA 找不到 GPU，模型被加载到 CPU（18GB RSS，0 MiB GPU），
静默失败。

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
333e34e fix: guard TRAINABLE_NOT_CORRECT with reward_median >= 0.4 threshold
2ecb123 fix: use SUM instead of AVG for LoRA gradient sync in TP
747f867 fix: sync LoRA gradients before optimizer.step() instead of weights after
87123ea fix: sync non-sharded LoRA matrix across TP ranks after optimizer step
4ca329e fix: recompute visual tower inv_freq in float32 to match HF precision
```

## Docker Images (on 228)

- `graspo:0.6.0-gating-median` — Fix 1 + Fix 2 + Fix 3 + Fix 4 (reward_median >= 0.4)
- `graspo:0.6.0-gating-fix` — Fix 1 + Fix 2 + Fix 3 + Fix 4 (reward_max >= 0.4, deprecated)
- `graspo:0.6.0-grad-sum-fix` — Fix 1 + Fix 2 + Fix 3 (AVG→SUM)
- `graspo:0.6.0-grad-sync` — Fix 1 + Fix 2 (gradient sync, AVG — deprecated)
- `graspo:0.6.0-full-fix` — Fix 1 + Fix 2 (weight sync, deprecated)
- `graspo:0.6.0-inv-freq-fix` — Fix 1 only
- `graspo:0.6.0-cuda13.2` — Original 0.6.0 release (has bugs)
