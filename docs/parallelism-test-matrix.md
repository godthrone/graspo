# GRASPO 五位一体并行测试矩阵（`docs/parallelism-test-matrix.md`）

> 目的：**证明 GRASPO 的"五位一体"（TP+DP+PP+SP+Checkpoint）在 SFT 与 RL 两种模式下、在 9B（Qwen3.5-9B）与 27B（Qwen3.8-27B）上全部可用。**
> 这是本文档的**唯一且充分必要条件**：只有把下面这张表**全部实测通过**，才证明五位一体架构实现成立。本文件是权威的测试定义 + 实测记录（测试参数 + 性能指标），每次跑完一行就更新状态与指标。

---

## 1. 测试设计（Python 生成的全组合表）

在单机 4×A800（80GB）上，对 `{TP, DP, PP, SP, GC}` 的可变维度做正交组合，得到一张覆盖全部**4 卡排列组合**的矩阵，再乘以 `{9B(Qwen3.5-9B), 27B(Qwen3.8-27B)} × {SFT, RL}`。固定约束：

- `world_size = dp_size × tp_size × pp_size`，且用满 4 张卡（`world_size = 4`）；
- `GC`（gradient_checkpointing）默认 `true`，作为独立维度验证其与 TP/DP/PP/SP 正交；
- `SP`（sequence_parallel）仅在 `tp_size ≥ 2` 时可选、按组合开关；
- 每种组合跑 `run.sh <cfg> --smoke`（跑 1 步验证环境）+ 短程训练（记录首步 loss / batch_sec / 峰值显存）。

## 2. 环境（对外公开版）

| 项 | 值 |
|---|---|
| 机器 | 单节点 4× NVIDIA A800 80GB（GPU 拓扑可能是 NVLink mesh 或 PCIe pair；`run.sh` 自动判定 `NCCL_P2P_DISABLE`） |
| 模型 | `Qwen3.5-9B`（9B）、`Qwen3.8-27B`（27B）—— 本地 Hugging Face 模型目录或 model id |
| 数据 | 多模态 SFT / RL JSONL（统一 OpenAI 兼容 `messages`/`tools` + `targets` 格式） |
| 运行 | `bash run.sh <cfg> --gpus 0,1,2,3 [--smoke]`（`run.sh` 是唯一受支持入口） |
| 示例配置 | `samples/configs/a800x4_qwen35_9b_*.yaml`、`samples/configs/a800x4_qwen35_27b_*.yaml` |

> 用于本地/内部复跑时可保留具体路径与拓扑；对外发布版已去除内部节点名/IP/代理。

## 3. 完整矩阵（4 卡 × 9B/27B × SFT/RL）

状态图例：✅ 实测通过 ｜ 🟡 已实现/需复测 ｜ ❌ 失败/已知问题 ｜ — 未测

| # | DP | TP | PP | SP | 描述 | SFT@9B | RL@9B | SFT@27B | RL@27B |
|---|:--:|:--:|:--:|:--:|------|:--:|:--:|:--:|:--:|
| 1 | 1 | 1 | 4 | off | PP=4 | ✅ | 🟡 (已实现，复测) | ✅ | 🟡 (已实现，复测) |
| 2 | 1 | 2 | 2 | off | TP=2+PP=2 | ✅ | 🟡 | ✅ | 🟡 |
| 3 | 1 | 2 | 2 | on | TP=2+PP=2+SP | ✅ | 🟡 | ✅ | 🟡 |
| 4 | 1 | 4 | 1 | off | TP=4 | ✅ | ✅ | ✅ | 🟡 |
| 5 | 1 | 4 | 1 | on | TP=4+SP | ✅ | ✅ | ✅ | 🟡 |
| 6 | 2 | 1 | 2 | off | DP=2+PP=2 | ✅ | 🟡 | ✅ | 🟡 |
| 7 | 2 | 2 | 1 | off | DP=2+TP=2 | ✅ | ✅ | ✅ | 🟡 |
| 8 | 2 | 2 | 1 | on | DP=2+TP=2+SP | ✅ | ✅ | ✅ | 🟡 |
| 9 | 4 | 1 | 1 | off | DP=4 | ✅ | 🟡 | ✅ | 🟡 |

> 说明：RL 含 PP 的组合（#1/#2/#3/#6）的生成路径**已有实现**，并曾在 **Qwen3.6-27B、8 卡 PP、1F1B** 上成功训练验证（约 1–2 月前 git 历史）。此前记录的"RL PP 被模型级 CUDA assert 阻塞"经复核为**误判**——这些格子应在本矩阵重新实测，而非标为不可用。RL 的 `optimize/backward`（1F1B backward+optimizer）也已在 `training.py` 实现。

## 4. 运行方法（每格）

```bash
# 拷贝对应示例配置到本机，修改 model_path / train_path / output_dir，再跑：
bash run.sh samples/configs/<该组合>.yaml --gpus 0,1,2,3 --smoke   # 环境冒烟
bash run.sh samples/configs/<该组合>.yaml --gpus 0,1,2,3            # 短程记录指标
```

每格需记录：首步 loss（与单卡基线对比）、`batch_sec`、逐卡峰值显存（`rank_metrics.rank_*.jsonl`）、是否 `final/` 落盘、是否无 hang/OOM。多卡组合还应额外验证数值等价性（见 §5 判据）。

## 5. 通过判据（证明"可用"≠"能跑"）

**单纯的 `exit 0` 只证明管道连接，不能证明正确性。** 每格必须满足：

1. **单卡基线存在**：先跑 `dp=1,tp=1,pp=1,sp=off`（可用 `samples/configs/sft_example.yaml` / `rl_example.yaml`）作为参照。
2. **数值等价**：同 seed + 同数据下，跨并行配置的首步 loss / per-token log-prob 相对单卡基线在容差内（bf16 单样本 ~1–2%），且 `_reduce_scatter_sp`/`_all_gather_sp` 往返能复原张量（TP≥2 时）。
3. **逐卡显存/负载均衡**：`rank_metrics.rank_*.jsonl` 文件数 == `world_size`（**一进程/卡**）；每 rank 绑定不同 `device_id`；`max(peak_memory) − min(peak_memory) < 5% of mean`。
4. **无 hang / OOM / NaN**：训练正常推进，`error.log` 为空或仅有预期告警。
5. **可复现**：同 config + seed 两次运行、落盘 loss 可比（排除不可控随机性），并写出 `config.yaml` 快照。

## 6. 性能指标（每格记录）

| 指标 | 来源 | 说明 |
|---|---|---|
| 首步 loss | `training.log` | 跨组合数值等价比较 |
| `batch_sec` | `training.log` / `rank_metrics` | 每批训练墙钟 |
| 峰值显存 / rank | `logs/rank_metrics.rank_*.jsonl` | 逐卡对称性 |
| 吞吐 tok/s | `analyze-profile` 的 `analysis_perf.jsonl` | 单卡→多卡加速比 |
| 气泡比（PP） | PP 调度/通信 debug | 1F1B vs 未来策略 |

`analyze-profile <run_dir>` 会输出六份分析文件（`analysis_profile.json` / `analysis_steps.jsonl` / `analysis_epochs.json` / `analysis_errors.jsonl` / `analysis_attribution.json` / `analysis_perf.jsonl`）供脚本化消费。

## 7. 结果记录（随跑随更）

| 日期 | #组合 | 模型 | 模式 | 结果 | 首步loss | batch_sec | 峰值显存/rank(GiB) | 备注 |
|---|---|---|---|---|---|---|---|---|
| 2026-08-23 | 1–9 | 9B | SFT | ✅ | — | — | ~40（DP=4 已实测） | 9/9 全过 |
| 2026-08-23 | 1–9 | 27B | SFT | ✅ | — | — | — | 9/9 全过 |
| 待测 | 1–9 | 9B / 27B | RL | — | — | — | — | 需补齐 RL 全模式实测（含 PP 复测） |

> 本文件为对外/长期权威版，不引用内部 run log（含内部节点/IP/路径的实测明细仅在内部维护，不入库）。
