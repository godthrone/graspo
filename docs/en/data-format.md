# Data Format

GRASPO uses JSONL for training data.

## Standard format

```jsonl
{"prompt": "User task text", "ground_truth": {"key": "value"}}
```

## Messages format

```jsonl
{"messages": [{"role": "user", "content": "Task"}, {"role": "assistant", "content": "{\"key\":\"value\"}"}]}
```

The last assistant message is used as ground truth.

## Excel conversion

Excel rows should contain:

- `instruction`
- `input`
- `output`

Run:

```bash
python -m graspo prepare-data --input dataset.xlsx --output data/train.jsonl
```

## ARD-SFT Format

Hard sample:

```json
{"sample_type": "hard", "messages": [{"role": "user", "content": "..."}], "target": "..."}
```

Anchor sample:

```json
{"sample_type": "anchor", "messages": [{"role": "user", "content": "..."}], "teacher_answer": "...", "teacher_model": "...", "anchor_meta": {}}
```
