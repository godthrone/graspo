# 排障

## Backend 自动选择失败

只打印后端选择结果，不启动训练：

```bash
python -m graspo train --config configs/graspo.yaml --print-backend
```

大模型在缺少 Megatron-LM/Core 时会提前失败。请先在训练服务器安装开源
Megatron-LM/Core。

## Native TP 冒烟无法启动

检查启动环境：

```bash
echo "$RANK $WORLD_SIZE $LOCAL_RANK"
python - <<'PY'
import megatron.core
print("megatron.core ok")
PY
```

`WORLD_SIZE` 必须等于 `backend_config.megatron_native.tensor_model_parallel_size`。

## 没有 train step

先缩小任务：

```bash
TP_SIZE=2 CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml bash scripts/run_train.sh
```

如果 rollout 成功但没有训练，检查 `rollouts.readable.jsonl`：perfect skip 或
invalid group 不会进入 ReplayBuffer。

## LoRA 没有更新

只有同时看到 finite loss、`nonzero_grad_count > 0`、`lora_norm_delta` 非零，
才认为 LoRA 确实更新。也要确认 `target_modules` 与 Qwen projection 名称一致。

## Tokenizer 没有 pad token

GRASPO 会优先把 `pad_token` 设为 `eos_token`。如果两者都没有，需要先在
tokenizer 中定义。
