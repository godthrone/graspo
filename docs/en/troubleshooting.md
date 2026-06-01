# Troubleshooting

## Backend Auto-Selection Fails

Print the selected backend without starting training:

```bash
python -m graspo train --config configs/graspo.yaml --print-backend
```

For large models, `backend: auto` fails early if Megatron-LM/Core is missing.
Install open-source Megatron-LM/Core on the training server first.

## Native TP Smoke Does Not Start

Check the launch environment:

```bash
echo "$RANK $WORLD_SIZE $LOCAL_RANK"
python - <<'PY'
import megatron.core
print("megatron.core ok")
PY
```

`WORLD_SIZE` must equal `backend_config.megatron_native.tensor_model_parallel_size`.

## No Train Step Appears

Start smaller:

```bash
TP_SIZE=2 CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml bash scripts/run_train.sh
```

If rollout succeeds but no train step appears, inspect `rollouts.readable.jsonl`
for perfect skips or invalid groups; ReplayBuffer only trains on accepted
trainable groups.

## LoRA Did Not Update

Treat update as valid only when train-step metrics show finite loss,
`nonzero_grad_count > 0`, and `lora_norm_delta` is non-zero. Check that
`target_modules` match the Qwen projection names.

## Tokenizer Has No Pad Token

GRASPO sets `pad_token` to `eos_token` when possible. If both are missing,
define them in the tokenizer before training.
