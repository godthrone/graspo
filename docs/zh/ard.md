# Anchor Replay Distillation

Anchor Replay Distillation，简称 ARD，是 GRASPO 的抗遗忘增强流程。它不是一个独立替代 GRASPO 的强化学习算法，而是在困难样本 SFT 之间加入一组由基础模型生成的 anchor bank，用低权重 replay 或可选 KL 蒸馏约束模型，尽量保留通用问答、代码、推理、解释等基础能力。

## 为什么需要 ARD

工业 Agent 的结构化任务经常只有少量高质量标注数据。这些数据通常来自现场工程师、运维人员或业务专家，采集和复核成本高，很难靠简单加大数据量解决问题。

GRASPO 本身适合小规模高质量业务数据。但如果在 GRASPO 之后只对困难样本做 SFT，模型可能过度贴近业务格式，出现通用能力遗忘。ARD 的目标是在强化业务结构化输出能力时，用尽量少的通用 anchor 样本给参数提供“锚点”。

## 标准流程

```text
1. 离线生成 anchor bank
2. 启动 GRASPO
3. 使用 hard samples + anchor bank 做 ARD-SFT
4. 回到第 2 步继续 GRASPO
```

Anchor bank 在第一轮训练前生成，训练过程中复用。只有更换 base model、prompt 模板、ontology，或者发现领域覆盖不足时，才需要重新生成。

## 数据产物

```text
anchor_bank/<base_model_id>/
  anchor_prompts.jsonl
  anchor_answered.jsonl
  anchor_filtered.jsonl
  anchor_train.jsonl
  anchor_eval.jsonl
  manifest.json
```

`manifest.json` 记录 teacher model、生成配置、随机种子、领域/任务/语言覆盖，以及过滤统计。

## 生成 anchor bank

```bash
MODEL_PATH=/data/models/base-model \
BASE_MODEL_ID=base-model \
GPU_COUNT=1 \
bash scripts/run_anchor_bank.sh
```

也可以分步执行：

```bash
graspo anchor-generate --config configs/anchor_generation.yaml --output anchor_prompts.jsonl
graspo anchor-answer --model-path /data/models/base-model --input anchor_prompts.jsonl --output anchor_answered.jsonl
graspo anchor-filter --input anchor_answered.jsonl --output anchor_filtered.jsonl --manifest-output manifest.json
graspo anchor-split --input anchor_filtered.jsonl --train-output anchor_train.jsonl --eval-output anchor_eval.jsonl
```

## ARD-SFT

Hard sample 使用普通 SFT CE loss，anchor sample 默认使用低权重 CE replay。可选开启 KL distillation，让 student 在 anchor prompt 上贴近训练前 base teacher 的分布。

```bash
MODEL_PATH=/data/models/base-model \
HARD_DATA_PATH=/data/graspo/hard_samples.jsonl \
ANCHOR_DATA_PATH=/data/graspo/anchor_bank/base-model/anchor_train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/ard-sft \
GPU_COUNT=8 \
bash scripts/run_sft_ard.sh
```

默认从 `hard:anchor = 1:1` 和较低 anchor loss 权重开始。业务能力下降时减少 anchor 影响；通用能力遗忘明显时增加 anchor 比例或权重。

## 评估

每轮迭代至少同时看两类指标：

- 业务 eval：结构化字段准确率、invalid rate、tool call 参数正确率。
- anchor eval：teacher answer 保持程度、输出长度变化、明显拒答或格式崩坏样本。

`eval-forgetting` 提供一个轻量本地比较工具，适合先做 smoke，不替代正式通用能力评估。
