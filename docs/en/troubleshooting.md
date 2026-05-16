# Troubleshooting

## LoRA target detection fails

Set `lora.target_modules` explicitly:

```yaml
lora:
  auto_target_modules: false
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Training hangs on multiple GPUs

FSDP generation requires all ranks to enter generation together. Start with
`max_steps: 1`, `prompts_per_rank: 1`, and a small model to isolate the issue.

## Tokenizer has no pad token

GRASPO sets `pad_token` to `eos_token` when possible. If both are missing, define
them in the model tokenizer before training.

## Docker cannot see GPUs

Check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

