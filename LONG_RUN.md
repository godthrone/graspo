# GRASPO TP=4 长训跟踪 — Qwen3.5-9B on 228

## 运行信息

| 项目 | 值 |
|------|-----|
| 镜像 | `graspo:0.7.0` (0.7.0-cuda13.2) |
| 容器名 | `graspo_tp4_longrun` |
| 服务器 | `10.1.251.228:22022` |
| GPU | 4-7 (4× A800 80GB) |
| 启动时间 | 2026-06-24 14:47 CST |
| 配置 | `config_tp4_v0.7.0_longrun.yaml` |
| 输出目录 | `outputs/tp4_v0.7.0_longrun/` |

## 训练参数

| 参数 | 值 |
|------|-----|
| training_epoch_count | **100** |
| max_steps | -1 (不限) |
| rollout_group_size | 8 |
| optimize_prompt_batch_size | 4 |
| optimize_times_per_step | **3** (v0.7.0 默认) |
| rollout_max_retry_times | 5 |
| forward_batch_size | 32 |
| max_new_tokens | 512 |
| empty_cache_after_rollout_split | **false** (v0.7.0 默认) |
| save_epoch_checkpoint | **true** (v0.7.0 新功能) |
| reject_unparseable_groups | true |
| perfect_skip_reward_threshold | 1.0 |

## 与上次对比 (v0.6.0-skip-fmt2, Epoch 0-4)

| 参数 | v0.6.0 旧训练 | v0.7.0 新训练 |
|------|-------------|-------------|
| 镜像 | graspo:0.6.0-skip-fmt2 | graspo:0.7.0 |
| optimize_times_per_step | 4 | 3 |
| empty_cache_after_rollout_split | true | false |
| save_steps | 999 | -1 |
| save_epoch_checkpoint | 无 | true |
| training_epoch_count | 10 | 100 |
| 旧日志归档 | — | `outputs/tp4_v0.6.0_epoch4/` |

## v0.6.0 旧训练基线 (4 epoch 完整 + 0.7 epoch 未完成)

| Epoch | reward_mean | content_mean | max_correct |
|-------|-------------|--------------|-------------|
| 0 | 0.684 | 0.687 | 5 |
| 1 | 0.793 | 0.737 | 6 |
| 2 | 0.805 | 0.740 | 8 |
| 3 | 0.817 | 0.747 | 9 |
| 4 (69.6%) | 0.827 | 0.756 | — |

## 监控命令

```bash
# 最新 step
ssh -p 22022 zhangzy@10.1.251.228 "docker logs graspo_tp4_longrun 2>&1 | grep 'train_step' | tail -1"

# Epoch 汇总
ssh -p 22022 zhangzy@10.1.251.228 "docker logs graspo_tp4_longrun 2>&1 | grep 'epoch_summary'"

# 检查 checkpoint
ssh -p 22022 zhangzy@10.1.251.228 "ls -la /home/zhangzy/elam_v12_fk/outputs/tp4_v0.7.0_longrun/epoch_*"

# 实时日志
ssh -p 22022 zhangzy@10.1.251.228 "docker logs -f graspo_tp4_longrun"
```

## 预计时间

- 405 样本 × 100 epochs / 4 samples/step = 10,125 steps
- 预计 ~110s/step (optimize_times=3, no empty_cache)
- 总计 ~12.9 天