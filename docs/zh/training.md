# 训练

## 服务器启动命令

```bash
MODEL_PATH=/data/models/your-base-model \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/run-001 \
GPU_COUNT=8 \
bash scripts/run_train.sh
```

## 多卡后端

v0.1 使用：

- Hugging Face Accelerate
- PyTorch FSDP `FULL_SHARD`
- PEFT LoRA

基础模型运行时挂载进容器，不会被打包进 Docker 镜像。

## Checkpoint

GRASPO 默认保存 LoRA adapter 到输出目录。v0.1 不强制自动合并 adapter 到基础模型。

## GRASPO + ARD 迭代

推荐流程：

```text
anchor bank -> GRASPO -> hard sample mining -> ARD-SFT -> GRASPO
```

ARD-SFT 启动命令：

```bash
MODEL_PATH=/data/models/your-base-model \
HARD_DATA_PATH=/data/graspo/hard_samples.jsonl \
ANCHOR_DATA_PATH=/data/graspo/anchor_bank/base-model/anchor_train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/ard-sft-001 \
GPU_COUNT=8 \
bash scripts/run_sft_ard.sh
```

详见 [Anchor Replay Distillation](ard.md)。

## 第一次服务器验证

建议先跑短任务：

```yaml
training:
  max_steps: 1
  save_steps: 1
```

先确认 rollout、logprob、反向传播和 adapter checkpoint 保存都正常，再跑长任务。
