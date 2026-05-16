# Anchor Replay Distillation

Anchor Replay Distillation, or ARD, is an anti-forgetting workflow for GRASPO. It is not a replacement for the GRASPO rollout algorithm. Instead, it adds a base-model-generated anchor bank between hard-sample SFT stages, using low-weight replay or optional KL distillation to preserve general QA, coding, reasoning, and explanation skills.

## Why ARD

Industrial Agent tasks often have only a small amount of high-quality labeled data. These labels may come from field engineers, operations teams, or business experts, so collecting and reviewing more data is expensive.

GRASPO is designed for this kind of high-value structured-output data. However, hard-sample-only SFT after GRASPO can overfit the model toward business formats and cause general capability forgetting. ARD gives the model a compact set of anchors while it learns the hard business cases.

## Standard Workflow

```text
1. Generate the anchor bank offline
2. Run GRASPO
3. Run ARD-SFT with hard samples + anchor bank
4. Return to step 2 and continue GRASPO
```

The anchor bank is generated before the first training round and reused during training. Regenerate it only when changing the base model, prompt template, ontology, or coverage requirements.

## Artifacts

```text
anchor_bank/<base_model_id>/
  anchor_prompts.jsonl
  anchor_answered.jsonl
  anchor_filtered.jsonl
  anchor_train.jsonl
  anchor_eval.jsonl
  manifest.json
```

`manifest.json` records the teacher model, generation config, random seed, domain/task/language coverage, and filter statistics.

## Generate an Anchor Bank

```bash
MODEL_PATH=/data/models/base-model \
BASE_MODEL_ID=base-model \
GPU_COUNT=1 \
bash scripts/run_anchor_bank.sh
```

You can also run the steps manually:

```bash
graspo anchor-generate --config configs/anchor_generation.yaml --output anchor_prompts.jsonl
graspo anchor-answer --model-path /data/models/base-model --input anchor_prompts.jsonl --output anchor_answered.jsonl
graspo anchor-filter --input anchor_answered.jsonl --output anchor_filtered.jsonl --manifest-output manifest.json
graspo anchor-split --input anchor_filtered.jsonl --train-output anchor_train.jsonl --eval-output anchor_eval.jsonl
```

## ARD-SFT

Hard samples use normal SFT CE loss. Anchor samples use low-weight CE replay by default. Optional KL distillation can make the student stay closer to the pre-training base teacher distribution on anchor prompts.

```bash
MODEL_PATH=/data/models/base-model \
HARD_DATA_PATH=/data/graspo/hard_samples.jsonl \
ANCHOR_DATA_PATH=/data/graspo/anchor_bank/base-model/anchor_train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/ard-sft \
GPU_COUNT=8 \
bash scripts/run_sft_ard.sh
```

Start with `hard:anchor = 1:1` and a low anchor loss weight. Reduce the anchor influence if business accuracy regresses; increase the anchor ratio or weight if general forgetting is visible.

## Evaluation

Track two groups of metrics for every iteration:

- Business eval: structured-field accuracy, invalid rate, and tool-call argument correctness.
- Anchor eval: teacher answer retention, output length drift, obvious refusals, and format collapse.

`eval-forgetting` is a lightweight local comparison tool for smoke checks. It does not replace a full general-capability evaluation suite.
