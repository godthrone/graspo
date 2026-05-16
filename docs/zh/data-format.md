# 数据格式

GRASPO 使用 JSONL 作为训练数据格式。

## 标准格式

```jsonl
{"prompt": "用户任务文本", "ground_truth": {"key": "value"}}
```

## Messages 格式

```jsonl
{"messages": [{"role": "user", "content": "任务"}, {"role": "assistant", "content": "{\"key\":\"value\"}"}]}
```

最后一条 assistant 消息会作为标准答案。

## Excel 转换

Excel 行建议包含：

- `instruction`
- `input`
- `output`

执行：

```bash
python -m graspo prepare-data --input dataset.xlsx --output data/train.jsonl
```

## ARD-SFT 格式

困难样本：

```json
{"sample_type": "hard", "messages": [{"role": "user", "content": "..."}], "target": "..."}
```

Anchor 样本：

```json
{"sample_type": "anchor", "messages": [{"role": "user", "content": "..."}], "teacher_answer": "...", "teacher_model": "...", "anchor_meta": {}}
```
