# 配置说明

主训练配置：

```text
configs/fsdp_lora_graspo.yaml
```

默认 8 卡 FSDP 启动配置：

```text
configs/accelerate_fsdp_8gpu.yaml
```

ARD 相关配置：

```text
configs/anchor_generation.yaml
configs/ard_sft_lora.yaml
```

## 模型

通过环境变量或 YAML 指定基础模型：

```bash
export MODEL_PATH=/data/models/your-base-model
```

模型需要能被 Hugging Face `AutoModelForCausalLM` 加载。

## LoRA

默认会自动识别常见 LoRA target modules。也可以手动覆盖：

```yaml
lora:
  auto_target_modules: false
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## 奖励

默认奖励要求模型输出 Markdown JSON 代码块：

```yaml
reward:
  check_think: false
  check_json_markdown: true
  check_list_order: false
```

## 训练参数

关键参数：

- `group_size`：每个 prompt 生成多少条回答
- `max_retry`：自适应重试次数
- `train_batch_size`：优化阶段 micro-batch
- `epochs_per_step`：ReplayBuffer 同一批经验复用多少轮
- `max_new_tokens`：生成 token 上限

## ARD 参数

- `training.anchor_ce_weight`：anchor replay 的 CE loss 权重
- `kl_distillation.enabled`：是否开启 teacher KL distillation
- `data.hard_train_path`：困难样本 SFT 数据
- `data.anchor_train_path`：离线 anchor bank 训练切分
