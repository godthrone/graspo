# ELAM GRASPO 训练实验记录

> 目标：通过 GRASPO 强化学习训练 Qwen3.5-9B 模型，使其学会 ELAM 机械臂场景的原子动作序列。
> 数据：405 条训练样本，`/data/v12_fk_scenes/data/elam_graspo_train_and.jsonl`。
> 环境：GPU 服务器 10.1.252.121，GPU 6,7（TP=2）。

---

## 一、动作空间

ELAM 场景共 5 个原子动作：

| 动作 | 参数 |
|------|------|
| 伸长手臂 | `distance_cm` |
| 收缩手臂 | `distance_cm` |
| 降低 | `distance_cm` |
| 顺时针旋转 | `angle_deg` |
| 逆时针旋转 | `angle_deg` |

每条样本的 target 包含 1-3 个 `robot_atomic_control` 工具调用。大多数样本是 3 个（81.6%），少数 2 个（18.4%）。

---

## 二、Reward 设计

```
reward = 100 × content_score + 10 × marker_reward + 1 × anti_useless_str
```

- `content_score`：`dict_compare_score(model_output, target)`，比较工具调用结构（action_type、参数名）和数值（distance_cm、angle_deg）
- `base_content_score`：剥离数值字段后的 content_score，只比较结构
- `marker_reward`：XML 标记存在性（`<tool_call>` 等）
- `anti_useless_str`：惩罚无用文本长度

**`numeric_tolerance` 机制**（v0.10.10 引入）：

```python
score = 1.0 / (1.0 + max(0, relative_error - numeric_tolerance))
```

- `tolerance=0.0`：原始公式，任何误差都扣分
- `tolerance=0.2`：20% 以内满分，超出后从 20% 边界平滑衰减
- `tolerance=10.0`：所有数值字段满分，`content_score == base_content_score`（方向-only 训练）

**`classify_group` 决策逻辑：**

| 判定 | 条件 |
|------|------|
| `perfect_skip` | `reward_median >= 1.0`（全组满分） |
| `trainable_max_correct` | `reward_max >= 1.0` 且 `reward_median < 1.0` |
| `trainable_not_correct` | `reward_max > reward_median` 且 `reward_max < 1.0` |
| `invalid_no_preference_gap` | `reward_max == reward_median`（无差异） |
| `invalid` | `is_invalid_group`（无 reward 方差或 uniform partial content） |
| `retry` | 未达 terminal 条件且未超过 `rollout_max_retries` |

---

## 三、实验版本历史

### 3.1 v0.10.6 — 初始版本

**配置：** Qwen3.5-9B base, r=128, lr=5e-6, SFT checkpoint `sft_elam_v2_merged`

**结果：** 训练崩溃。

**根因：** reward 函数对"多输出 tool call"没有惩罚。模型学到输出 5 个 tool call 来"碰运气"，导致 reward 膨胀。

**修复：** `too many tool calls` 标记为 parse_error，视为格式错误。

---

### 3.2 v0.10.7 — 修复多 tool call 惩罚

**结果：** 仍崩溃，但原因不同。

---

### 3.3 v0.10.8 — CUDA OOM

**症状：** Step 12 时 CUDA OOM，35GB reserved but unallocated。

**根因：** 内存碎片化。PyTorch 的默认 allocator 在 rollout → optimize 交替中产生大量碎片。

**修复：** 添加 `ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

### 3.4 v0.10.9 — 内存修复后 Loss 爆炸

**配置：** Qwen3.5-9B base, r=128, `expandable_segments:True`。

**症状：** Step 35-40 loss 从 +0.004 跳变到 +1098。

**根因分析：**

1. 某些组中所有 completion 的工具调用结构完全正确（`base_content_score=1.0`），但数值参数有微小差异
2. 数值差异来自浮点噪声（如 0.9676 vs 0.9681，gap=0.0005）
3. 这个 fake gap 使组被判定为 `trainable_not_correct` 而非 `invalid_no_preference_gap`
4. 进入 replay buffer 后，微小 advantage 被 PPO 放大，污染整个 buffer
5. 正反馈循环：污染 buffer → 极端策略 → 更多污染 → 爆炸

**修复：** 引入 `numeric_tolerance=0.2`，20% 误差内给满分。

---

### 3.5 v0.10.10 — numeric_tolerance=0.2

**配置：** Qwen3.5-9B base, r=128, `numeric_tolerance=0.2`。

**症状：** 仍爆炸，Step 41-62 loss 从 +2.0 到 +1.7 万亿。

**根因分析：**

1. `numeric_tolerance=0.2` 只在数值字段上生效，但 `action_type` 是字符串字段
2. Qwen3.5-9B base 的 action_type 命中率约 16%
3. 模型从未产生 `max_correct` 组，reward 信号始终是噪声
4. 模型坍缩到"对所有 prompt 输出相同错误答案"的模式
5. 大量 `invalid` 组被过滤，剩余 trainable 组是退化信号 → buffer 污染 → 爆炸

**关键数据：** 53 步，0 个 `max_correct` 组。

---

### 3.6 v4 — numeric_tolerance=10.0（方向-only）

**配置：** Qwen3.5-9B base, r=128, `numeric_tolerance=10.0`。

**设计理念：** 数值完全不参与 reward，GRASPO 只学"选什么工具"，不学"参数填多少"。参数值由预训练知识自然提供。

**结果：** 9.5 小时，53 步，3 epoch，`max_correct=0` 全程。

**关键发现：**

| 指标 | 值 |
|------|-----|
| 决策流 | retry 78.2%, invalid 12.6%, trainable 8.1%, inv_gap 1.1% |
| Action-type 匹配率 | 16.2%（base 模型） |
| TC 数量匹配率 | 55.7% |
| epoch 0-2 content mean | 0.768 → 0.773 → 0.801（几乎无变化） |
| max_correct | 0 |

**根本原因：** Qwen3.5-9B base 不会输出"收缩手臂"这个动作，倾向于输出"伸长手臂"、"顺时针旋转"等高频动作。5 个 action_type 的搜索空间对纯 RL 来说太大，模型永远碰不到正确答案。

**Gap 分析：**

```
gap 直方图 (427 trainable groups):
├── 0-0.001:   5 (1.2%)   ← 超小 gap
├── 0.01-0.02: 77 (18.0%)  ← 1 个 completion 偏了 TC 数
├── 0.02-0.05: 94 (22.0%)  ← 2 个偏了
├── 0.1-0.4:    0          ← 完全空洞！
└── 0.4-1.0:  239 (56.0%)  ← 巨大 gap，有人碰巧输出正确 TC 数
```

**双峰分布证明：** 不存在"接近正确"的中间状态——要么 TC 数对了，要么不对。没有渐进梯度信号。

**tool call 数量错配：** 75.4% 的 trainable 组 tool call 数量不对。模型倾向输出 1-2 个 tool call（84% 太少），而目标要求 3 个。

---

### 3.7 v5 — SFT checkpoint + numeric_tolerance=10.0

**配置：** SFT merged HF 模型 (`sft_elam_v2/final` → merged-hf), r=16, alpha=32, `numeric_tolerance=10.0`

**预期：** SFT 已教会模型 5 个 action_type，第一步就出 `max_correct`。

**结果：** 模型质量低于 base。

| 指标 | v4 (base) | v5 (SFT merged) |
|------|-----------|-----------------|
| TC 数量匹配 | 55.7% | 35.2% |
| 动作类型匹配 | 16.2% | 3.2% |

**根因：** 
1. SFT 训练 loss 只降到 0.12（好的 SFT 应该 <0.01），模型未充分学会 ELAM 格式
2. merge 操作（native checkpoint → merged HF）可能引入精度损失
3. LoRA 参数从 r=16 合并到完整权重，信息密度被稀释

**结论：** 当前的 SFT checkpoint 不能作为 GRASPO 的初始化。

---

## 四、核心发现

### 4.1 GRASPO 需要正确答案才能学习

GRASPO 通过组内相对优势（max vs median）学习。如果 8 个 completion 在本组内没有正确者，它只能选"错得没那么离谱的"。当全组都是不同方向的错误时，优势信号是噪声。

### 4.2 搜索空间太大导致永远碰不到正确答案

- 5 个 action_type，每个样本 1-3 个 tool call
- 随机猜中全部动作类型的概率 ≈ 1/125
- Base 模型有预训练偏向，实际命中率 ≈ 16%
- 53 步 × 8 组 × 8 completion = 3392 个尝试，零命中

### 4.3 数值噪声 vs 动作类型噪声

- `numeric_tolerance=10.0` 成功消除了数值噪声（验证通过）
- 但 `action_type` 是字符串字段，tolerance 对它无效
- 真正的噪声源是 tool call 数量错配和动作类型错配

### 4.4 SFT checkpoint 质量至关重要

- 当前 SFT（loss=0.12, r=16）质量不够
- 好的 SFT 应该 loss < 0.01
- 需要大 r、多 epoch 的 SFT 先充分学会 ELAM 格式

---

## 五、未来方向

### 5.1 重新训练 SFT

- 使用更大的 LoRA rank（r=128）
- 目标 loss < 0.01
- 确保 merged model 的 action_type 匹配率 > 90%

### 5.2 改进 reward 设计

- 对 tool call 数量错配使用更细粒度的 reward（而非二元正确/错误）
- 允许 action_type 的模糊匹配（如编辑距离）
- 或者：将 GRASPO 目标缩减为"只优化 tool call 数量，不优化 action_type"

### 5.3 改进 classify_group

- 增加 `gap` 阈值：当 `reward_max - reward_median < threshold` 时直接标记为 `invalid_no_preference_gap`
- 当前代码中 `reward_max == reward_median` 太严格，浮点精度导致很多该过滤的组没有被过滤

### 5.4 课程学习

- 先用简单样本（1-2 个 tool call）训练，逐步增加难度
- 或者先用 SFT 覆盖 80% 的样本，剩余 20% 用 GRASPO 精调

---

## 六、代码变更记录

| Commit | 描述 |
|--------|------|
| `88a8d3c` | fix: add accelerate to Dockerfile |
| `680b112` | fix: penalize too-many-tool-calls as format error |
| `04ba356` | refactor: remove launch.gpus, auto-derive nproc_per_node |
| `32c9b0a` | fix: add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True |
| `e78a473` | feat: add numeric_tolerance to reward scoring |
| `1324108` | chore: add numeric_tolerance to all example configs |
| `ace31ce` | feat: numeric_tolerance=10.0, direction-only, clean output v4 |
| `146ba82` | feat: SFT checkpoint + numeric_tolerance=10.0, r=16, clean output v5 |
| `3c7c549` | feat: v5 w/ SFT merged HF model + numeric_tolerance=10.0 |