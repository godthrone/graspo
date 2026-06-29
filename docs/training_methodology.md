# GRASPO 训练方法论

## 背景：GRPO 与结构化输出

GRPO（Group Relative Policy Optimization）的核心思想是：对同一个 prompt 采样多条 completion，在组内比较 reward，用组内相对优势信号（而非绝对 reward）来更新策略。这天然适合结构化输出任务——JSON 生成、工具调用、信息抽取——因为这些任务的答案可以**自动校验**，不需要人工标注 reward。

GRASPO 在 GRPO 基础上针对结构化输出场景做了以下改进：

1. **三层奖励体系**：结构性标记 + 内容正确性 + 反冗余惩罚，每个维度独立可调
2. **多目标最优选择**：支持每个 prompt 配多个 acceptable targets，取最佳匹配
3. **组智能过滤**：自动识别并过滤/重试无训练价值的 group
4. **Perfect-skip 机制**：已稳定答对的 prompt 不再消耗 optimizer budget
5. **ReplayBuffer**：跨 step 的经验回放，提升样本效率

## 训练循环

每个训练 step 包含以下阶段：

```
┌──────────────────────────────────────────────────────────┐
│  Phase 1: Rollout 生成                                    │
│  prompt batch → 每个 prompt 采样 G 个 completion          │
│  G = rollout_group_size (默认 8)                         │
├──────────────────────────────────────────────────────────┤
│  Phase 2: Reward 评分                                     │
│  每个 completion → GraspoReward.score() → 结构化 reward  │
│  提取 JSON/tool-call → 与 targets 比较 → 计算分数        │
├──────────────────────────────────────────────────────────┤
│  Phase 3: 组决策                                         │
│  每个 group 根据 reward 分布分类：                         │
│  · median_reward >= perfect_skip_threshold → perfect_skip│
│  · 有 reward 方差 → trainable (max_correct/not_correct)  │
│  · 无方差 → invalid_no_preference_gap                    │
│  · 格式损坏 → invalid → retry (最多 5 次)                 │
├──────────────────────────────────────────────────────────┤
│  Phase 4: 优化                                           │
│  对 trainable group：                                    │
│  · group_advantages() → 组内相对优势                      │
│  · 将 Experience 加入 ReplayBuffer                       │
│  · 重复 optimize_times_per_step 次                       │
│  · 从 ReplayBuffer 采样 optimize_prompt_batch_size 个     │
│  · PPO-style policy ratio clipping + LoRA 更新           │
└──────────────────────────────────────────────────────────┘
```

## 三层奖励体系

### 1. 结构性标记奖励（Marker Reward）

检查 completion 是否包含预期的格式标记：

- ````json ...  ```` fence 是否存在且闭合
- `<think>...</think>` 标签（可选，通过 `check_think` 开关）
- tool-call 结构（可选，通过 `check_tool_call` 开关）

权重：`marker_reward_weight`（默认 10）。格式正确但内容错误的 completion 也会得到基础分。

### 2. 内容正确性奖励（Content Reward）

从 completion 中提取 JSON/tool-call payload，与 targets 进行结构化比较：

- `dict_compare_score()`：递归比较 JSON 结构，叶子节点数值参与梯度信号（`dcs`），非数值叶子仅参与结构门控（`base_dcs`/`all_right`）
- 支持 list order optional：默认不要求列表顺序
- 支持多 target alternatives：每个 target 独立打分，取最佳匹配（`best_scoring target wins`）
- `all_right` 额外加分：内容完全匹配时额外加 `content_reward_weight`

权重：`content_reward_weight`（默认 100）。内容正确的重要性远高于格式正确。

### 3. 反冗余惩罚（Anti-Useless Reward）

对 filler text（非 JSON、非结构性内容）进行指数衰减惩罚：

```
penalty = anti_useless_str_reward_weight / 2^(len(useless_text) / half_reward_len)
```

- `anti_useless_str_reward_weight`：最大惩罚值（默认 1）
- `anti_useless_str_half_reward_len`：半衰期（默认 100 字符）

效果：简洁回答得分更高，冗长填充文本被自动抑制。

### 归一化

```python
normalized_reward = raw_score / max_score
```

完美答案的 raw_score 可能略超 1.0（反冗余奖励在分母计算后加上），这是故意设计——给完美答案一个"卓越"标记。

## 组决策体系

每个 rollout group（同一个 prompt 的 G 个 completion）根据其 reward 分布进入不同处理路径：

| 决策 | 条件 | 行为 |
|------|------|------|
| **perfect_skip** | `median(rewards) >= perfect_skip_reward_threshold` | 跳过训练，节省计算 |
| **trainable_max_correct** | 有 reward 方差 + best completion 正确（`all_right=True`） | 正常训练 |
| **trainable_not_correct** | 有 reward 方差 + best completion 不完全正确 | 正常训练 |
| **invalid** | 最好 completion 有 parse error 或 tool-call count mismatch | 先 retry（最多 5 次），仍失败则丢弃 |
| **invalid_no_preference_gap** | reward 完全相同（无组内方差） | 丢弃（无偏好信号） |
| **retry** | rollout 失败（如内存不足） | 重试（最多 5 次） |

**防线角色**：`reject_unparseable_groups = True`（默认）是防线——格式损坏的 group 被拦截在训练边界之外。这是宪法 §2.3（边界校验即防呆）的具体体现。

## ReplayBuffer

`ReplayBuffer` 保存 completion 级别的训练经验（tokens + old_log_probs + advantages）。核心机制：

- 写入：每次 rollout 的 trainable group 转化为 Experience 并追加
- 读取：optimize step 从 buffer 采样 `optimize_prompt_batch_size` 个 prompt 的经验
- 容量控制：通过 `replay_buffer_optimize_threshold = G × B`（默认 8 × 8 = 64）限制训练前的最低经验积累
- 触发条件：buffer 中 trainable completion 数 ≥ threshold 时才开始训练

这确保 optimzer 有足够的样本多样性才开始更新。

## 效果评估方法

GRASPO 内置多层次的训练过程监控和效果评估体系：

### 1. 组内分类统计（Group Classification Stats）

每个 step 统计各决策类型的数量和比例：

```
perfect_skip:    已稳定答对，无需训练
trainable:       有训练价值（max_correct / not_correct）
invalid:         格式错误被丢弃
invalid_no_pref: 无偏好信号被丢弃
retry:           rollout 失败后重试
```

**关键趋势信号**：
- `perfect_skip` 比例持续上升 → 模型逐渐掌握任务
- `trainable_max_correct` 比例上升 → 模型正在学会"正确+简洁"
- `invalid` 持续高位 → 格式解析有问题，检查 reward config
- `invalid_no_pref` 为 0 → 所有 group 都有区分度（理想状态）

### 2. Reward 趋势指标

| 指标 | 含义 |
|------|------|
| `reward_mean` | batch 内平均 reward |
| `reward_median` | batch 内中位 reward |
| `reward_max_median_gap_mean` | 每个 group 内 max - median 的 batch 均值（组内区分度） |
| `reward_nonzero_range_rate` | 有 reward 方差的 group 占比 |
| `nonzero_range_rate` (window) | 滑动窗口内 reward 方差 > 0 的比例 |

**关键趋势信号**：
- `reward_mean` + `reward_median` 上升 → 整体训练有效
- `reward_max_median_gap_mean` 保持 >0 → 组内仍然有偏好信号（不要过早下降）
- `nonzero_range_rate` 接近 0 → 模型可能过拟合，所有 completion 都接近完美

### 3. Content Score 趋势

| 指标 | 含义 |
|------|------|
| `content_mean` | batch 内平均内容得分 |
| `content_all_zero_rate` (window) | 滑动窗口内全部 group 内容得分为 0 的比例 |
| `content_all_one_rate` (window) | 滑动窗口内全部 group 内容得分为 1 的比例 |

**关键趋势信号**：
- `content_all_zero_rate` 持续 ≥ 0.8 → 模型无法产生正确内容，检查：数据质量、温度设置、prompt 设计
- `content_all_one_rate` 上升 → 模型趋于完美（配合 perfect_skip 比例判断）

### 4. Training Health 综合判断

`training_health()` 基于多个信号给出训练是否健康的综合判断：

| 异常信号 | 含义 |
|----------|------|
| `nonfinite_loss_or_grad` | loss/grad 出现 inf/nan，需要降低学习率或检查数据 |
| `zero_lora_delta` | LoRA 权重无变化，可能梯度消失 |
| `batch_reward_all_zero` | 本 batch 所有 reward = 0，reward 函数可能配置不当 |
| `batch_json_truncation_detected` | JSON 截断（生成长度不够或模型出错） |
| `reward_all_zero_window` | 近期 10+ 步 reward 全部为 0 |
| `no_group_reward_variance_window` | 近期无组内区分度，所有 completion 相同 |
| `content_score_all_zero_window` | 近期 ≥80% 为内容零分 |

当 `early_stop_recommended = True` 时，训练可能需要干预。

### 5. Timing 分析

每个 step 输出详细计时分解：`rollout_sec`、`reward_cpu_sec`、`decision_sec`、`old_logprob_sec`、`optimize_sec`、`checkpoint_sec`。子分解包括 `prefill_sec`、`decode_sec`、`sampling_sec`、`micro_batch_forward_sec`、`backward_sec`、`optimizer_step_sec`。这些可以精确定位性能瓶颈。

### 6. Debug 信号

| 指标 | 含义 |
|------|------|
| `missing_json_marker` | 无 JSON fence 的 completion 数 |
| `unclosed_json_fence` | JSON fence 未闭合的 completion 数 |
| `invalid_json` | JSON 解析失败的 completion 数 |
| `truncated_json` | 疑似 JSON 截断（上下文可能不够长） |
| `tool_call_parse_error` | tool-call 解析失败数 |
| `tool_call_count_mismatch` | tool call 数量与 target 不匹配 |

这些是模型输出质量和 `max_new_tokens` 设置是否合理的直接反馈。

## 与原始 GRPO 的关键差异

| 维度 | 原始 GRPO | GRASPO |
|------|----------|--------|
| 奖励 | 外部 reward model / 单一分数 | 内置三层结构化 reward，可审计 |
| 目标 | 通用文本生成 | 结构化输出（JSON、tool-call） |
| 组过滤 | 简单丢弃 | 四级分类 + retry + perfect-skip |
| 训练 | 全参数 | LoRA only，单卡 80G 可训 9B |
| 数据 | 对话偏好 | JSONL targets（自动比较） |
| 评估 | 外部 eval | 内置多维统计 + health 判断 |
