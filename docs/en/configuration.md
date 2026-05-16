# Configuration

The main training config is:

```text
configs/fsdp_lora_graspo.yaml
```

The default 8-GPU FSDP launcher config is:

```text
configs/accelerate_fsdp_8gpu.yaml
```

ARD-related configs:

```text
configs/anchor_generation.yaml
configs/ard_sft_lora.yaml
```

## Model

Set the base model by environment variable or by editing YAML:

```bash
export MODEL_PATH=/data/models/your-base-model
```

The model must be loadable by Hugging Face `AutoModelForCausalLM`.

## LoRA

By default GRASPO auto-detects common LoRA target modules. To override:

```yaml
lora:
  auto_target_modules: false
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Reward

Default reward settings expect markdown JSON fences:

```yaml
reward:
  check_think: false
  check_json_markdown: true
  check_list_order: false
```

## Training

Important knobs:

- `group_size`: generations per prompt
- `max_retry`: adaptive retry count
- `train_batch_size`: optimization micro-batch size
- `epochs_per_step`: reuse each replay buffer batch for multiple updates
- `max_new_tokens`: generation budget

## ARD Parameters

- `training.anchor_ce_weight`: CE loss weight for anchor replay
- `kl_distillation.enabled`: whether to enable teacher KL distillation
- `data.hard_train_path`: hard-sample SFT data
- `data.anchor_train_path`: offline anchor bank training split
