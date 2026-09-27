# Ripple — 算法层

Ripple（涟漪）是 GRASPO 的算法层，负责所有训练方法论相关的**纯计算**逻辑。零设施依赖——可以在单线程 CPU 上运行、可以单独测试、可以在任何环境复现。

命名由来：在强化学习中，一个 token 的即时奖励并非孤立——它通过 advantage 计算对前后 token 的梯度传播产生涟漪效应（ripple effect），信用分配如波浪般回传。Flow 承载数据流动，Ripple 捕获这一波浪式回传的微观动态。命名出处的落地实现是 **GRASPO-Ripple token 级奖励算法**（`annotation/advantages.py`，v0.20.0 标注驱动）——见 `src/graspo/ripple/__init__.py:1-12`。

## 边界

- **入**：配置参数 + 原始数据（completion 文本、targets、tokenizer）
- **出**：reward 分数、token 级 advantage、group 决策、loss 值
- **不碰**：GPU 设备、分布式通信、网络、文件 IO、Docker

边界不是口号，是**可核查的导入约束**（`src/graspo/ripple/__init__.py:9-12`）：

- 允许：标准库、pydantic、torch 纯张量计算（CPU 可单测）
- 禁止：GPU 设备调用（`torch.cuda`）、分布式、网络、文件 IO
- **不依赖 `graspo.flow` 的任何模块**（`src/graspo/ripple/algorithm.py:6`；全包 `graspo.flow` 只出现在注释里，无 import）

反向依赖是允许且真实的：flow 消费 ripple 的算法核心与监控摘要——`GraspoAlgorithmCore` 被 msswift 训练器注入（`src/graspo/flow/msswift/trainer.py:110-112`），`classify_group` 被两侧训练器调用（`src/graspo/flow/msswift/trainer.py:355`），`monitoring` 被 native 训练器与 optimize 消费（`src/graspo/flow/trainer/trainer.py:35`、`src/graspo/flow/trainer/optimize.py:11`）。

## 模块清单（23 个模块 + 6 个包 `__init__`）

旧的"五大模块"章节按"一条 completion 的处理链路 + 支持模块"来划分，与当前包的物理结构（按**域**分目录：annotation / reward / parsing / multimodal / monitoring，加 4 个顶层模块）严重不符，且漏掉了 5 个既有模块。现按代码结构重排并逐项对应到文件：

| 域 | 模块 | 职责 | 证据 |
|----|------|------|------|
| 顶层 | `algorithm.py` | `GraspoAlgorithmCore`——后端无关的算法核心，组合下面全部模块 | `algorithm.py:32-55` |
| 顶层 | `loss.py` | `GRASPORippleLoss` + RL/SFT 共享的 log-prob 实现 + `masked_mean` | `loss.py:16-80` |
| 顶层 | `group_decision.py` | 组分类决策 + replay 阈值 + reward 统计工具 | `group_decision.py:80-190` |
| 顶层 | `buffer.py` | `Experience` 数据容器 + `ReplayBuffer` | `buffer.py:8-46` |
| 顶层 | `data.py` | 数据加载与样本构建（`Sample`、SFT tokenize、多模态 deferred） | `data.py:26-492` |
| annotation | `char_tag.py` | `CharTag` 枚举（S/V/T/W/E/D，含 `trainable` 属性） | `char_tag.py:13-40` |
| annotation | `labeler.py` | `annotate()` 主入口 + `AnnotationInput` / `AnnotationResult` | `labeler.py:12-52` |
| annotation | `json_labeler.py` | JSON 逐字符标注：围栏提取、状态机扫描、字段名拼错/多余检测 | `json_labeler.py:115-295` |
| annotation | `tool_call_labeler.py` | Qwen XML tool call 标注：期望 mark 序列逐字符比对、参数无序集合匹配 | `tool_call_labeler.py:196-375` |
| annotation | `tokenize.py` | 字符标注 → token 标注（offset_mapping，E 优先合并） | `tokenize.py:42-75` |
| annotation | `advantages.py` | 标注 → per-token advantage（字段级相对化 + 末尾 EOS 惩罚） | `advantages.py:152-295` |
| reward | `reward.py` | `GraspoReward` 三层加权评分 + `RewardResult` + 注册表 | `reward.py:53-355` |
| reward | `compare.py` | 结构化 dict 比较：`dcs` / `base_dcs` / `all_right` | `compare.py:8-278` |
| reward | `normalize.py` | target 归一化与 JSON 校验、`TargetScore` | `normalize.py:13-100` |
| parsing | `qwen_tool_parser.py` | Qwen XML tool call 严格解析（规范化 + ElementTree） | `qwen_tool_parser.py:1-278` |
| parsing | `json_tool_parser.py` | 跨模型通用 JSON tool call 解析 | `json_tool_parser.py:16-49` |
| parsing | `completion.py` | `ParsedCompletion` 数据模型（reward 前置） | `completion.py:6-30` |
| parsing | `xml.py` | Qwen XML 构建（SFT target text / tool_calls → XML） | `xml.py:14-77` |
| parsing | `classification.py` | 输出/任务分类判定（logger 与 monitoring 共用） | `classification.py:14-71` |
| multimodal | `rows.py` | 多模态 rows 的纯数据操作（构建/读写/路径解析） | `rows.py:19-146` |
| multimodal | `contract.py` | 多模态防呆契约：缺图即硬失败 | `contract.py:52-124` |
| monitoring | `stats.py` | 训练/epoch 统计的数据模型与序列化 | `stats.py:10-137` |
| monitoring | `summary.py` | 组监控、reward 窗口/批次摘要、训练健康度 | `summary.py:36-545` |

> 数模块的主人是 `ripple/__init__.py` 的 docstring（`__init__.py:1-12`），不是本表；本表只做导航。

## 奖励体系

奖励是 ripple 的核心输出。奖励由 **`GraspoReward` 类**统一实现（`reward.py:53`），并通过 `REWARD_REGISTRY` 注册/扩展。注册表经 `graspo.core.discovery` 的 entry-point 机制装配（生产真相源是 `pyproject.toml` 的 `[project.entry-points."graspo.rewards"]`，开发回退表在 `discovery.py:22-24`）——当前内置 `graspo` 一种，未来可注册新的 reward 类而无需改动调用端（算法层插件化）：

- 注册入口：`pyproject.toml:63-64`；开发回退：`src/graspo/core/discovery.py:22-24`
- 实例化与未知实现报错：`reward.py:345-353`
- 配置侧校验（`RewardConfig.kind` 必须在注册表内）：`src/graspo/core/schema.py:243-252`

### 标注与 reward 的分工（核心创新）

GRASPO 的核心是把"**对格式的强识别能力**"和"**对 value 的自由定义 reward 能力**"分开：

- 底层是 **`annotation/` 字符级标注模块**（核心创新）：对整条 completion 的**每个字符**打角色标注（`CharTag`：S/V/T/W/E/D），据此赋予**每个 token 级 advantage**。它天然 tokenizer 无关，对格式本身有极强的结构化判断能力（逐字符比对结构模板）。
- reward 只是**给标注中的某个 value 赋一个分**：既可以是给某个字段（JSON 的 value / tool-call 参数）赋值打分，也可以是**把整段 JSON 序列视作一个 value 赋一个分**（因为 JSON 整体是一个 value）。这样保留了用户**自由定义 value reward** 的能力。
- **扩展性**：未来要增加更多打分能力（某类字段的领域规则、质量评分等），**只需要改 reward 模块**，底层的标注模块与 token 级梯度训练**都不用动**。

### 三层加权评分与归一化

旧文档把评分描述为"三个独立维度"（结构性标记 / 内容正确性 / 反冗余惩罚）**并列相加**。"并列"与代码不符：三个维度是**同一 `raw_score` 上的加权累加项**，最后统一除以 `max_score` 归一化（`reward.py:65-66,151-157`）。维度与权重如下：

| 维度 | 权重配置（默认） | 实现 |
|------|------------------|------|
| marker（结构标记） | `marker_reward_weight = 10.0` | 逐 mark 查找命中即加分（`reward.py:75-97`）；think 标记成对出现加 2 倍（`reward.py:110`、`_think_marker_score` `reward.py:321-328`） |
| content（内容正确性） | `content_reward_weight = 100.0` | 见下节 |
| anti-useless（反冗余） | `anti_useless_str_reward_weight = 1.0`，半衰长度 `anti_useless_str_half_reward_len = 100` | 指数衰减：`w / 2^(len/half)`（`reward.py:330-334`） |
| 归一化 | `numeric_tolerance = 0.2`；`check_list_order = False`；`check_think = False`；`check_json_markdown = True` | `normalized_reward = raw_score / max_score`（`reward.py:157`） |

默认值证据：`src/graspo/core/schema.py:233-241`。`max_score` 由 check list 结构推出（`reward.py:291-304`）。

一条被如实记录的实现细节：反冗余加分在 `max_score` 分母算出**之后**才累加，因此一个"干净且全对"的答案的 reward 可以**略高于 1.0**（`reward.py:154-157` 的注释明确说明这是与原 GRASPO 实现对齐的结果）。

### 内容正确性

从 completion 中提取 JSON/tool-call payload，与 targets 做结构化比较（JSON 路径 `reward.py:99-150`，tool-call 路径 `reward.py:200-246`）：

- `dict_compare_score()`：递归比较 JSON 结构，产出 `dcs`（含数值叶子）与 `base_dcs`（剥离数值叶子，只看非数值结构），`all_right = base_total > 0 and base_total == base_check`（`compare.py:8-29`）
- **数值叶子在 `base_dcs` 中被剥离**：剥离的是**两侧**（模型输出与 GT），因此 `base_dcs` 只门控结构正确性、不要求数值精确匹配（`compare.py:11-17,41-61`）
- 支持多 target alternatives：每个 target 独立打分，取 `dcs` 最高者（`reward.py:117-137`）
- `all_right` 额外加分：`base_content_score >= 1.0` 时用 `content_score` 加权；若 `all_right` 再加一份 `content_reward_weight`（`reward.py:141-149`）
- tool-call 路径的额外格式门：模型生成的 tool call 数**多于**任何 target 时视为格式错（追加 `"too many tool calls"` 到 `parse_errors`）；**少于**不惩罚（`reward.py:201-214,306-319`）

### 反冗余惩罚

对 filler text（非结构化内容）进行指数衰减惩罚，抑制冗长输出（`reward.py:330-334`）。超过半衰长度阈值时还会跳过 content 比较（JSON 路径 `reward.py:111-112`；tool-call 路径 `reward.py:211-214`）。

## 字符级标注（annotation/）

Rollout 完成后对每条 completion 做**字符级结构标注**，作为 token 级 advantage 的唯一输入。

### 设计哲学

- **字符级标注** `List[CharTag]`（S/V/T/W/E/D），长度恒等于 `len(completion)`，只标角色不打分（`annotation/__init__.py:3-7`、`labeler.py:24-30`）
- **逐字符比对**：期望 mark 序列与模型输出逐字符比对，首个不匹配字符标 E。标注定义在字符级，token 由 offset_mapping 派生，天然 tokenizer 无关（`tool_call_labeler.py:13-17`、`tokenize.py:1-9`）
- **不做 FORMAT/CONTENT 分类**：每个 token 只关注"在当前前缀条件下与模板结构的对齐状态"（`char_tag.py:3-7`）
- **严格对齐截断**：E 之后全部 D（不训练）；值错误不触发 E（下游相似度打分），仅结构/类型错误触发 E（`char_tag.py:31-35`、`tool_call_labeler.py:7-11`）
- **与 reward 层语义一致**：只有语义错误标 E（缺字段/多余字段/拼错/类型错）

### CharTag 枚举

| 枚举 | 含义 | Advantage |
|------|------|-----------|
| S (`STRUCTURE`) | 结构字符，与期望模板匹配 | +1.0 |
| V (`VALUE`) | 值字符（参数值 / JSON value） | 字段相似度 − μ_f |
| T (`THINK`) | think 内容 | 0（不训练） |
| W (`WASTE`) | 前导/尾随/夹缝文本 | 0（不训练） |
| E (`ERROR`) | 首个与期望不匹配字符 | −1.0 |
| D (`DROPPED`) | E 之后所有字符 | 0（不训练） |

定义与逐项语义：`char_tag.py:13-40`。`trainable` 属性为 `S / V / E` 三者（`char_tag.py:37-40`）。

> **旧别名已不保留**：旧文档使用的全小写名（`structure` / `value` / `think` / `waste` / `error` / `dropped`）在代码中是**兼容别名常量**（`char_tag.py:43-49`），不是枚举成员；文档统一只写正式名 S/V/T/W/E/D。

### 两个标注器的差异（旧文档未区分）

| | tool call（`tool_call_labeler.py:196-375`） | JSON（`json_labeler.py:115-295`） |
|---|---|---|
| 结构识别 | think 标记 → S（可选，`check_think=True`）；`<tool_call>` 开标记 → S（含其后换行） | 围栏 \`\`\`json → S（含其后换行，`check_json_markdown=True`）；**围栏缺失但结构存在（首非空白是 JSON 起始字符）→ 首非空白字符 E + 其后 D**（`json_labeler.py:126-131`） |
| 主体扫描 | 期望 mark 序列逐字符比对 | 状态机扫描：栈跟踪对象/数组嵌套，配平行的 GT 上下文栈（`json_labeler.py:143-160`） |
| 参数顺序 | **无序集合匹配**：参数顺序颠倒不标 E（`tool_call_labeler.py:248-251`）；多余参数在**参数名首字符**标 E | 字段名按 GT 检测拼错 / 多余 |
| 残缺/未知标签 | E 落在"应出现 `>` 的位置"（参数名后首字符），`<parameter=` 与参数名本身是正确结构 → S（`tool_call_labeler.py:261-279`） | — |

### 模块结构

见上文「模块清单」的 annotation 行；每行的职责与行号证据在表内。

### 测试数据集

`tests/data/annotation_testset_v3.jsonl`（**65 条：tool call 44 + JSON 21**，字段为 `id/type/case/completion/ground_truth/annotation/correct/error_pos/notes`）作为标注模块回归基准，覆盖完美输出、值错误、前导文本、拼错、双调用、多余字段、缺字段、顺序颠倒、截断、乱码、嵌套结构等场景。

断言口径（`tests/ripple/annotation/test_annotation_testset.py:72-95`）：逐字符标注与期望一致、长度等于 `len(completion)`、E 之后全部为 D、`error_pos` 等于首个 E 下标；另有全枚举合法性、E→D 不变量、全样本 `correct` 标记等附加断言（同文件 `:101-131`）。测试集自身的覆盖矩阵说明见该文件 docstring（`test_annotation_testset.py:1-26`）。

## Token 级 Advantage

标注模块产出字符级 `tags` + 并行 `fields` 数组后，advantage 层（`advantages.py`）将其转换为 token 级训练信号。完整链路（`advantages.py:152-241`）：

1. **末尾 EOS 惩罚**：结构不完整（tool_call 缺 `</tool_call>`；JSON 去围栏后无法 `json.loads`）且非 `max_new_tokens` 硬截断，且尚无 E → 末尾非空白字符标 E（`advantages.py:29-74`）
2. **字符 → token 映射**：`char_to_token_labels` 走 `offset_mapping`，合并语义为 **E > V > S > T > D > W**（`tokenize.py:27-39`）
3. **V span 提取**：同一 field 的连续 V 字符合并为一个 span（`advantages.py:77-98`）
4. **字段级 raw 分**：`field_score` 把整个 value span 与 GT 值比较——GT 为 list 时做**存在性匹配**（命中任一元素即 1.0），标量走 `leaf_compare_score`（数字容差 / 字符串精确）（`advantages.py:116-134`）；值解析对带引号字符串保持字符串，避免全数字字符串（如 IMSI）被 `float` 误转（`advantages.py:101-113`）
5. **μ_f 相对化**：`μ_f` = 组内该 field 的 raw 均值（**有该字段 V 段的 completion 才参与**），`adv(V) = raw − μ_f`（`advantages.py:208-213,230-235`）
6. **标签 → 数值**：`S→+1.0 / V→raw−μ_f / T·W·D→0 / E→−1.0`（`advantages.py:224-240`）

补充事实：

- **no group-level normalization**（v0.20.0）：advantage 不做组内 z-score 或整条标量加权（`advantages.py:249-254`）
- **ragged → 张量对齐**：prompt 位置 0、生成区填 ragged advantage、尾部 padding 0（`advantages.py:279-295`）
- **n=1 安全零**：单条 completion 时 μ_f 等于自身 raw，V 的 advantage 为 0

## Group 决策体系

每个 rollout group（同一个 prompt 的 G 个 completion）根据 reward 分布进入不同处理路径。`GroupDecision` 共 6 个取值（`group_decision.py:16-22`），`classify_group` 按 **if/elif 优先级**判定（`group_decision.py:80-150`）：

| 决策 | 条件 | 行为 |
|------|------|------|
| **perfect_skip** | `reward_median ≥ perfect_skip_reward_threshold`（默认 1.0） | 跳过训练，节省计算 |
| **trainable_max_correct** | `reward_max ≥ 阈值` | 正常训练 |
| **retry**（① parse error） | `reject_unparseable_groups=True` 且 best completion 有 parse error，且 `retry_count < rollout_max_retries` | 重试 |
| **invalid**（① 的重试耗尽分支） | 同上但 `retry_count ≥ rollout_max_retries` | 丢弃 |
| **trainable_not_correct** | `reward_max > reward_median` | 正常训练 |
| **retry**（② 低分） | `reward_max < 阈值` 且 `retry_count < rollout_max_retries` | 重试 |
| **invalid**（空 rewards / 无方差 / 部分一致性） | `not has_reward_variance(rewards)` 或 `is_uniform_partial_content(content_scores)` | 丢弃 |
| **invalid_no_preference_gap** | `reward_max == reward_median` | 丢弃（无偏好信号） |

三点旧文档未写清的语义（均以代码为准）：

1. **判定顺序**：`perfect_skip` / `trainable_*` 的阈值判定**先于** parse-error 与 invalid 判定（`group_decision.py:121-127`）；旧表按"条件"并列列出，容易被读成互斥独立判据。
2. **`retry` 有两个来源**：parse-error 重试与"全组低于阈值"重试；旧表只写了后者。两个来源都受同一个 `retry_count < rollout_max_retries` 约束。
3. **默认值**：`rollout_max_retries = 5`、`perfect_skip_reward_threshold = 1.0`、`reject_unparseable_groups = True` 定义在 `src/graspo/core/schema.py:327,346-347`。

**防线角色**：`reject_unparseable_groups = True` 是防线（fail-closed），不是退路——格式损坏的 group 被拦截在训练边界之外，因为不可解析的数据不可能产出有意义的训练信号（`group_decision.py:92-98`）。

**replay 触发阈值**（旧文档完全缺失）：`replay_buffer_optimize_threshold = rollout_queue_batch_size × rollout_group_size`——恰好一个 rollout queue 触发一次 optimize，使**队列节奏**与**训练微批大小**解耦（`group_decision.py:152-177`）；`ReplayBuffer` 为定长 FIFO，超限保留**最后** `limit` 条（`buffer.py:26-46`）。

## PPO Loss

`loss.py` 实现 GRPO 风格的 PPO-clip loss（`loss.py:61-80`）：

- `ratio = exp(log_probs − old_log_probs)`，先 `clamp(0.1, 10.0)` 防止极端比值（`loss.py:73-74`）
- `surr1 = ratio × A`，`surr2 = clamp(ratio, 1−ε, 1+ε) × A`，`loss = −min(surr1, surr2)`（ε = `policy_ratio_clip_eps`，默认 0.2，`loss.py:62`）
- 逐 token 计算后按 `action_mask` 做 masked mean，再 batch 内取均值（`loss.py:80`、`masked_mean` `loss.py:16-22`）

> **口径修正（重要）**：旧文档写"核心公式 A：`advantage_i = z_i × max(rewards)^p`（p=2），质量加权后全对组获得满推力"。**该公式已不存在于代码中**——它是 v0.14.x 的"整条 completion 质量加权 z-score advantage"，已于 v0.20.0 被字符级标注驱动的 token 级 advantage 取代，并在 commit `bf0e100`（2026-08-23）删除。现在 advantage 的**唯一**来源是 `annotation/advantages.py`，且**不做任何组级归一化**（`group_decision.py:1-8` 明确记录了这次取代及其原因：整条 completion 只能给一个标量推力，无法在错误的 token 上给负梯度）。

`loss.py` 同时承载 **RL 与 SFT 共用的 log-prob 唯一实现** `masked_token_log_probs_from_hidden`（分块 logsumexp，不物化 `(B,S,V)`，峰值张量与 TP 规模无关）——旧文档未提及；历史教训是该实现曾因物化完整 logits 导致 SFT OOM（`loss.py:25-58`）。

## 多模态契约（multimodal/）

多模态行构建的**单一真相源**在 `ripple/multimodal/rows.py`，`MULTIMODAL_ROWS_KEY = "_multimodal_rows"` 是全项目唯一键名，唯一写入方是 `attach_rows`，读取方是 `rows_from_metadata`（`rows.py:17-19,6-11`）。

**防线是两道，不是三道**（修正旧文档）：

1. **RL 训练防线**：`assert_rl_training_has_multimodal`——sequences 含 `image_token_id` 但 metadata 无 rows → `RuntimeError`，在 `sequence_log_probs` / `train_batch` 的 forward 前调用（`contract.py:52-91`；调用点 `src/graspo/flow/trainer/rollout.py:290`、`src/graspo/flow/adapters/models/qwen35_36/training.py:101`）
2. **SFT 批次防线**：`assert_sft_batch_has_multimodal`——样本含 media 但 batch 中无 `multimodal_inputs` → `RuntimeError`，在 SFT collate 后、forward 前调用（`contract.py:94-124`；调用点 `src/graspo/flow/adapters/models/qwen35_36/training_sft.py:434`）

**为什么这两道是硬失败**：多模态数据丢失是"静默吞功能"——丢了图却继续训练就是纯浪费算力。历史事故是 v13：`<image>` 占位 token 存在但 metadata 从未携带 rows，模型把图像占位符当普通 token 嵌入，训练 19.5 小时无任何告警（`contract.py:74-76`）。

**启动预检不属 ripple 层**：`encode → attach → resolve` 链路校验、视觉 LoRA 参数注册校验、fake 前向梯度非零校验都在 **flow 层**的 `run_multimodal_preflight`（`src/graspo/flow/trainer/preflight.py:257-342`，其中第 2/3/4 步在 `:313,324,336`）。旧文档把这一层算作 ripple 契约的"第一道防线"，属于层边界归错：ripple 是零设施依赖的算法层，任何"加载模型 / 跑 fake forward"的检查必然在 flow。

## 解析（parsing/）

- **`qwen_tool_parser.py` 是标签格式常量的单一真相源**：`TOOL_CALL_OPEN/CLOSE`、`FUNCTION_OPEN/CLOSE`、`PARAMETER_OPEN/CLOSE`、`THINK_OPEN/CLOSE` 定义于此（`qwen_tool_parser.py:38-45`）；**标注模块**从这里 import 同一组常量（`annotation/tool_call_labeler.py:24-48`），避免两套格式理解漂移。
- **reward 模块并不引用这些常量**：reward 的 marker 识别走 `RewardConfig` 的 check list（`check_think` 开时用字面量 `" thinking"` / `"</think>"`，`reward.py:276-289,321-328`）。旧文档"标注模块和 reward 模块都引用它的常量"与代码不符。
- **解析路径**：`parse_qwen_tool_completion` 是主解析入口（`qwen_tool_parser.py:216`）；先规范化 `<function=NAME>` / `<parameter=NAME>` 为合法 XML（元素名不允许 `=`，`qwen_tool_parser.py:51-59`）再用 `ElementTree` **严格解析**，未配对标签/嵌套标签/杂散文本即 parse error；随后对照 tools schema 校验 required 参数（`qwen_tool_parser.py:11-22`）。防呆动机是 v0.14 训练崩溃：旧正则解析会被双 `<tool_call>` 开标记的"作弊模板"绕过，模型学会省略数值参数拿 0.886 分（reward hacking）。
- **`json_tool_parser.py` 只解析 `<tool_call>` 的 body**（跨模型标准的 `{"name":..., "arguments":{...}}`），**不做围栏提取**——围栏处理在标注侧 `json_labeler.py`（`json_tool_parser.py:1-46`；旧文档"处理 JSON fence 内的内容提取"与代码不符）。
- **`xml.py` 是构建侧**：`build_sft_target_text` / `tool_calls_to_xml` / `format_xml_param_value`（`xml.py:14-77`），服务 SFT target 生成与奖励侧，不参与解析。
- **`classification.py` 是旧文档完全缺失的模块**：`is_pure_tool_call_task` / `summarize_json_markers` / `likely_truncated_json` / `tool_call_count_mismatch_count` 是 logger（flow）与 monitoring（ripple）**共用**的输出分类判定；历史上存在两份近似但不一致的实现，已归并到本模块（`classification.py:1-8,14-71`）。

## 后端约束：禁用 DeepSpeed（2026-09-27 用户指令）

Ripple 层本身是后端无关的算法核心——它**不 import** 任何后端模块（`algorithm.py:6`），上面的奖赏/advantage/组决策/loss 全在 CPU 上可独立测试。因此"禁用 DeepSpeed"落地为**对后端与并行/优化器相关设计的硬约束**，而不是 ripple 内部的代码改动：

**规则**：今后不得使用 DeepSpeed；后端与并行/优化器相关能力改用 **ms-swift 的 Megatron / FSDP 模块**，或**开发 native 后端**替代。

**理由（梯度范数守卫/计数在 DeepSpeed 路径下不可信）**：

- `accelerate` 的 `clip_grad_norm_` 对 DeepSpeed 有专属分支，**不做裁剪、返回的也不是本步的 norm**，而是 `engine.get_global_grad_norm()`：
  `accelerate/accelerator.py:2982-2986`（`return` 在 `:2985`；`clip_grad_norm_` 定义于 `:2946`）。
  **一手证据（可在本工作区直接复读）**：`.local/hb-workspace/20260923-analysis/task-msswift-naninf-study/src/accelerate_accelerator.py:2982-2986`——该文件是从镜像内**原地抽出**的 `accelerate/accelerator.py` 副本。
  ⚠ **本工位独立复核发现的出入**：任务书与先前工位引用的一段 `accelerate/accelerator.py` 行号 **在该副本中并非该分支**——该范围落在 `clip_grad_value_`（def `:3009`）的 **docstring 示例**里，该函数的 "DeepSpeed and FSDP do not support `clip_grad_value_`" 异常实际在 **`:3031-3032`**（`raise` 在 `:3032`）。与本条款有关的正确位置是 **`:2982-2986`**（与 `task-msswift-naninf-study/report.md:159,396` 一致）。本文以**自己复读副本的结果**为准，记 `:2982-2986`。
  （行号依据：调研者裁决 + 本工位复读；副本 sha256 `47088e0ab3bf21eec97e16afa14595e1db511f6ead9ab85c4eaa5f6f66fe5e61`，4359 行，accelerate **1.14.0**。）
- DeepSpeed 侧 `_global_grad_norm` 在 `_take_model_step()` 里、**`optimizer.step()` 之后**才赋值（`deepspeed/runtime/engine.py:2707-2711`；取值器 `:743-753`；初值 `None` 在 `:256`）⇒ step 前读到的是**陈旧值**；ZeRO-1/2 的 `step()` 在 `overflow` 为真时提前 `return`（`deepspeed/runtime/zero/stage_1_and_2.py:2110-2135`），**根本不计算**。
  依据：`.local/hb-workspace/20260923-analysis/task-msswift-naninf-study/report.md:160-161,412-418`。（**注意**：该工位只抽取了 `swift`/`transformers`/`accelerate` 四份副本，**未抽取 DeepSpeed 源码**，故这三处行号目前只有报告转述、无本地副本可复读。）
- 上游复现：ms-swift issue #3930——用户 `JingMog` 报告 *"the grad clip seems not work. I use deepspeed zero2"*；维护者 `hjh0119` 回"NaN 梯度问题已修复"并链接 `swift/trainers/mixin.py#L264-L281`。
  来源：https://github.com/modelscope/ms-swift/issues/3930 ，引自 `.local/hb-workspace/20260923-analysis/task-msswift-naninf-study/report.md:162,180`。
- 本项目实测印证：T006（ZeRO-2 + bf16）新臂 6 条 `log_history` 全部 `count=0 / coverage=0`；由此 12 个 DS 档的梯度守卫/计数被判为**不可判**，设计上如实产出 `coverage=0` 而**绝不伪造 0**

**已废弃**：上述 DeepSpeed 路径下的任何"梯度范数守卫 / 计数可信"假设**已废弃**。文档不再把它当作可用的守卫手段。

**当前仓库的迁移状态（如实记录，不是已完成）**：DeepSpeed 仍是 schema 中存在的字段，且被若干校验消息引用，尚未从代码删除——

- `MsswiftConfig.deepspeed` 字段仍存在（`src/graspo/core/schema.py:757`），另有 `deepspeed_autotp_size`（`:761`）与 `teacher_deepspeed`（`:531`）；Megatron 段在同文件（`:692,696,739,799`），FSDP 在 `:763`
- 校验消息仍以 DeepSpeed AutoTP / `zero2_offload` 作为替代建议（`src/graspo/core/schema.py:153-172,216`）
- 后端选择入口的说明仍写 "ms-swift infrastructure (Megatron/DeepSpeed/vLLM)"（`src/graspo/flow/backend_selection.py:36`）
- 新增能力的方向是 Megatron 通道：`megatron_passthrough_argv` 与 `use_megatron_fsdp`（`src/graspo/flow/msswift/_config_mapping.py:196-209,323-345`）

> 也就是说：**规则已生效、迁移未完成**。Ripple 层的接口在两种后端下必须一致（`GraspoAlgorithmCore` 同时被 native 训练器与 msswift 训练器组合：`src/graspo/flow/msswift/trainer.py:110-112`），这正是"禁用 DS"不必改动 ripple 内部实现、只需保证接口口径一致的原因。

## 不可核实项（明确标注，不猜）

### 本工位的核实能力边界（为什么某些引用不是"一手"）

本工位**无外网**、且**当前环境未安装 `accelerate` / `deepspeed` / `ms-swift`**，因此不能访问任何上游仓库或 issue 页面。能核实的只有**文件系统上已有的副本**。据此，上一节的引用分三档如实标注：

| 档 | 含义 | 本节是否列入"不可核实" |
|----|------|------------------------|
| **A. 一手副本可复读** | 工作区内有原地抽取的源码副本，我已亲自 `sed` 复读并核对行号 | 否（已在上节引用；含一处出入更正） |
| **B. 工作区调研报告转述（附 URL）** | 报告记录了一手取证过程与链接，但我无法复读被引源码、也无法联网 | 否（已在上节引用并标明转述性质） |
| **C. 无任何本地证据** | 无副本、无报告、无可读产物 | 是 |

**A 档已核实 1 项**：
- `accelerate/accelerator.py` 的 DeepSpeed 分支 = `:2982-2986`（`.local/hb-workspace/20260923-analysis/task-msswift-naninf-study/src/accelerate_accelerator.py:2982-2986`）。
  复核同时**更正**了任务书/先前工位所引的一段错误行号——它落在 `clip_grad_value_`（def `:3009`）的 docstring 示例里，该函数的"不支持 `clip_grad_value_`"异常在 **`:3031-3032`**，与 DeepSpeed 梯度范数条款无关。
  行号依据：**调研者裁决 + 本工位复读**；副本 sha256 `47088e0ab3bf21eec97e16afa14595e1db511f6ead9ab85c4eaa5f6f66fe5e61`，4359 行，accelerate **1.14.0**（与 228 镜像逐字节一致）。

**B 档已引用 2 项**（不在"不可核实"之列，但保留"未能独立联网/无安装"的说明）：
- ms-swift issue #3930 的内容与维护者回复：引自 `task-msswift-naninf-study/report.md:162,180` + URL `https://github.com/modelscope/ms-swift/issues/3930`。
- DeepSpeed `_global_grad_norm` 的赋值时机 / ZeRO overflow 提前 return：引自 `task-msswift-naninf-study/report.md:160-161,412-418`。
  （同档可一并追溯：HF transformers PR #13619、ms-swift PR #3465 / #3469、DeepSpeed issue #8415，均在 `report.md:172-182,426-436`；本文件只引用与禁用 DS 直接相关者。）

### C 档：无可核实来源（1 条）

1. **禁用 DeepSpeed 的完整迁移是否被排期**：`.local/hb-workspace/20260923-analysis/task-ds-exit-study/` 的 `evidence/` 与 `src/` 为空目录，但**同目录存在 `report.md`（468 行 / 44,594 B，本次已复读）**——它给出的是**全换 FSDP2** 的推荐结论（`.local/hb-workspace/20260923-analysis/task-ds-exit-study/report.md:351`）与**两阶段技术顺序 + 工作量估算**（`:363` 阶段 1「0.5–1 人天」、`:374` 阶段 2「1–2 人天」），并设有 U1–U10 不确定项清单（`:407-420`）；但它**未给排期**（无日期/责任/承诺，全文 `排期` 0 命中），也未把迁移排期列入任何可核实结论。**排期仍属未定事项**，本文只记录"规则已生效、schema 字段仍在"这一可核实现状（见上节"当前仓库的迁移状态"）。
