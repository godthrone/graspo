# Data Format

GRASPO uses JSONL for training data.

## Standard Format

```jsonl
{"prompt": "User task text", "ground_truth": {"key": "value"}}
```

## Messages Format

```jsonl
{"messages": [{"role": "user", "content": "Task"}, {"role": "assistant", "content": "{\"key\":\"value\"}"}]}
```

The last assistant message is used as ground truth for standard supervised data.

## Excel Conversion

Excel rows should contain:

- `instruction`
- `input`
- `output`

Run:

```bash
python -m graspo prepare-data --input dataset.xlsx --output data/train.jsonl
```
