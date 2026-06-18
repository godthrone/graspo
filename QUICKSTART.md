# GRASPO 快速上手

GRPO-style LoRA 训练器，支持多模态（图像+文本）结构化输出 RL（JSON、tool call）。

## 环境

- **Python 3.11+**, PyTorch 2.5+, CUDA 12.4
- **模型**: Qwen3-8B, Qwen3.5-9B, Qwen3.6-27B
- **GPU**: 推荐 A800 80GB × N，支持 TP（Tensor Parallel）和 PP（Pipeline Parallel）

## 快速启动

```bash
# 1. 安装
pip install graspo

# 2. 准备数据（JSONL 格式）
# 每行: {"messages": [...], "targets": [{"id": "...", "output": {...}}], "tools": [...]}

# 3. 写配置（或复制 config_example.yaml 修改）
# 4. 启动训练
python -m graspo launch --config config_example.yaml
```

## 核心概念

### 两个正交参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `rollout_group_size` | **算法参数**：每个样本生成几条 completion 算 advantage | 8 |
| `gpu_memory_utilization` | **资源参数**：用多少 GPU 显存做 rollout generation（0~1） | 0.90 |

只需调 `gpu_memory_utilization` 控制吞吐。越高 → batch 越大 → 吞吐越高，但 OOM 风险越大。其他正交参数（如 `optimize_completion_batch_size`）一般不动。

### Reward 机制

1. 解析模型输出 → 提取 JSON / tool call
2. 与 `targets` 列表比较，取最高分
3. 评分 = 标记分（marker）+ 结构分（content）+ 完全匹配 bonus - 多余文本惩罚
4. **`all_right`**：只看非数值字段是否完全匹配（数值字段只影响 `content_score` 梯度，不影响 perfect_skip 决策）

## 多模态训练

数据格式：messages 中嵌入 `<image>` 类型 content，tools 声明工具定义。示例见 `data/sample_multimodal.jsonl` 和 `data/sample_tool_call.jsonl`。

关键配置：
```yaml
check_tool_call: true         # tool call 格式用 score_parsed
check_json_markdown: false    # tool call 不需要 fenced JSON
max_prompt_length: 8192       # 多模态 prompt 较长
gpu_memory_utilization: 0.70  # 8K prompt 用保守值
```

## Docker 部署（228 示例）

```bash
docker run -d --name graspo_training \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -v /path/to/model:/workspace/models/model:ro \
  -v /path/to/data:/workspace/data \
  --ipc=host --shm-size=16g \
  graspo:latest \
  python -m graspo launch --config /workspace/data/config_docker.yaml
```

> ⚠️ **必须用 `--gpus all` + `CUDA_VISIBLE_DEVICES`**，不能用 `--gpus '"device=4,5,6,7"'`。后者会导致 `torch.cuda.is_available() == False`。

## 常用故障排查

### OOM（显存溢出）

| 症状 | 原因 | 解决 |
|------|------|------|
| 启动即 OOM | `gpu_memory_utilization` 太高 | 降到 0.65~0.70 |
| 跑几轮后 OOM | 显存积累未释放 | 确认 `empty_cache_after_rollout_split: true` |
| 持续 OOM | 估计过于乐观 | 增大安全因子（`_kv_cache_batch_fits_budget` 中的 `peak_bytes * 1.5`） |

### 训练不收敛

- 检查 reward 是否合理：看 `rollouts.readable.jsonl` 中 `reward_mean` 趋势
- 确认 `all_right` 不要过于严格（数值字段应只影响 content_score，不阻塞 all_right）
- 加 `check_think: false` 简化任务

### 多模态训练慢

- 降低 `_MAX_MULTIMODAL_SAMPLES_PER_CALL`（trainer.py）减少 CPU 编码批次
- 确认 `gpu_memory_utilization` 合理（`nvidia-smi` 看生成阶段显存 > 60%）

## 监控

```bash
# GPU 显存/利用率
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 实时训练事件
docker logs -f <container> | grep '"event"'

# 奖励趋势
cat outputs/xxx/rollouts.readable.jsonl | jq '.group_stats.reward_mean'

# 训练指标
cat outputs/xxx/train_batches.readable.jsonl | jq '{step, loss, grad_norm}'
```

## 导出模型

```bash
python -m graspo export --format peft-adapter --output-dir ./exported
```

## 目录结构

```
src/graspo/
├── core/           # reward, compare, schema, advantage, buffer
├── backends/
│   └── native_tp/  # 自研 TP/PP 后端（无 vLLM/Megatron 依赖）
│       ├── qwen_tp_adapter.py  # ★ 主力：generation + training + LoRA
│       ├── trainer.py          # GRASPO 训练循环 + retry + replay buffer
│       ├── runtime.py          # runtime 协议 + 参数校验
│       └── logger.py           # rollout / timing / train_batch 日志
└── trainer/
    └── lora.py     # LoRA target module 解析
```
