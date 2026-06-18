# 228 ELAM v11 FK 长训运维

端午节 100-epoch 训练，Qwen3.5-9B + 405 条多模态 tool-call 数据。

## 服务器

```bash
ssh -p 22022 zhangzy@10.1.251.228
```

GPU: 8× A800 80GB，当前用卡 4-7，TP=4。

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `gpu_memory_utilization` | 0.70~0.80 | 显存利用率，越高越快但 OOM 风险越大 |
| `rollout_group_size` | 8 | 算法参数，不动 |
| `training_epoch_count` | 100 | 长训 |
| `max_prompt_length` | 8192 | 多模态 prompt 较长 |
| `max_new_tokens` | 512 | tool call 很短 |
| `check_tool_call` | true | 工具调用评分 |

## 启动/重启

```bash
docker run -d --name graspo_elam_v11_fk \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -v /home/zhangzy/models/Qwen3.5-9B:/workspace/models/Qwen3.5-9B:ro \
  -v /home/zhangzy/elam_v11_fk:/workspace/data \
  --ipc=host --shm-size=16g \
  graspo:v11-final \
  python -m graspo launch --config /workspace/data/config_docker.yaml
```

> ⚠️ 必须 `--gpus all` + `CUDA_VISIBLE_DEVICES`。用 `--gpus '"device=4,5,6,7"'` 会导致 `torch.cuda.is_available() == False`，模型全部跑在 CPU 上。

数据路径：`~/elam_v11_fk/data/train_docker.jsonl`（图片用绝对路径 `/workspace/data/images/`）。

## 监控

### 实时状态

```bash
# GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 容器
docker ps --filter name=graspo_elam_v11_fk
docker logs -f graspo_elam_v11_fk 2>&1 | grep '"event"'
```

### 奖励

```bash
tail -f ~/elam_v11_fk/outputs/elam_v11_fk_graspo_v1/rollouts.readable.jsonl \
  | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    g = d.get('group_stats', {})
    print(f\"step={d.get('step')} reward={g.get('reward_mean',0):.3f} content={g.get('content_mean',0):.3f} all_right_any={g.get('all_right_any')} decision={d.get('decision')}\")
"
```

### 训练指标

```bash
tail -f ~/elam_v11_fk/outputs/elam_v11_fk_graspo_v1/train_batches.readable.jsonl \
  | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    print(f\"step={d.get('step')} loss={d.get('loss',0):.4f} grad_norm={d.get('grad_norm',0):.2f} lr={d.get('learning_rate')}\")
"
```

### 直观一句话

```bash
# 正常状态：4 张卡显存 ~60-80GB，利用率 80-100%，retry_count 在 0-3 之间
watch -n 5 'echo "--- GPU ---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E "^4,|^5,|^6,|^7,"; echo "--- Rollouts ---"; wc -l ~/elam_v11_fk/outputs/elam_v11_fk_graspo_v1/rollouts.readable.jsonl 2>/dev/null; echo "--- Train batches ---"; wc -l ~/elam_v11_fk/outputs/elam_v11_fk_graspo_v1/train_batches.readable.jsonl 2>/dev/null'
```

## 调参

### 改 gpu_memory_utilization

```bash
vim ~/elam_v11_fk/config_docker.yaml   # 改 gpu_memory_utilization
docker restart graspo_elam_v11_fk       # 重启即可生效
```

| gmu | 峰值显存 | prompt_chunk | 吞吐 | 风险 |
|-----|---------|-------------|------|------|
| 0.65 | ~60 GB | 2 | 低 | 最安全 |
| 0.70 | ~63 GB | 3 | 中 | 安全 |
| 0.78 | ~75 GB | 3 | 中高 | 略紧张 |
| 0.80 | ~80 GB | 4 | 高 | 可能 OOM |

### OOM 了怎么办

1. 看日志确认是哪个阶段 OOM：`docker logs graspo_elam_v11_fk | grep OutOfMemory`
2. 降 `gpu_memory_utilization` 0.05~0.10
3. 重启
4. 如果反复 OOM，检查代码是否有修改（`_kv_cache_batch_fits_budget` 中的安全因子）

## 文件位置

| 内容 | 路径 |
|------|------|
| 训练数据 | `~/elam_v11_fk/data/train_docker.jsonl` |
| 图片 | `~/elam_v11_fk/images/` (810 张，640×360) |
| 配置 | `~/elam_v11_fk/config_docker.yaml` |
| 输出 | `~/elam_v11_fk/outputs/elam_v11_fk_graspo_v1/` |
| rollouts 日志 | `.../rollouts.readable.jsonl` |
| 训练日志 | `.../train_batches.readable.jsonl` |
| timing 日志 | `.../timing_events.jsonl` |
| 模型权重 | `/home/zhangzy/models/Qwen3.5-9B/` (只读挂载) |
| 镜像 | `graspo:v11-final` |
| 容器 | `graspo_elam_v11_fk` |

## 预期指标

| 阶段 | reward_mean | content_score | perfect_skip | trainable_max_correct |
|------|-------------|---------------|-------------|----------------------|
| 早期 (step 0-50) | 0.2~0.5 | 0.3~0.6 | 很少 | 少量 |
| 中期 (step 50-200) | 0.5~0.7 | 0.6~0.8 | 开始出现 | 增加 |
| 后期 (step 200+) | 0.7~0.9 | 0.8~0.95 | 频繁 | 多数 |

正常信号：reward 持续上升，trainable_max_correct 增加，loss 平稳下降。  
危险信号：reward 长期不涨、all_right 始终为零、loss NaN/爆炸。
