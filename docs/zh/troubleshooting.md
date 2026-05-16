# 排障

## LoRA target 自动识别失败

手动设置 `lora.target_modules`：

```yaml
lora:
  auto_target_modules: false
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## 多卡训练卡住

FSDP 下生成阶段要求所有 rank 同步进入 generation。先用小模型，并设置 `max_steps: 1`、`prompts_per_rank: 1` 来定位问题。

## Tokenizer 没有 pad token

GRASPO 会优先把 `pad_token` 设置成 `eos_token`。如果两者都没有，需要先在 tokenizer 里定义。

## Docker 看不到 GPU

检查：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

