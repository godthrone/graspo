# GRASPO 并行测试矩阵（`docs/parallelism-test-matrix.md`）

> 目的：**证明 GRASPO 的 TP+DP+PP+SP+GC 五位一体在 1/2/4 卡、9B/27B 上全部可用。**
> 这是本文档的**唯一且充分必要条件**：只有把下面这张表**全部实测通过**，才证明并行架构实现成立。
> 本文件是权威的测试定义 + 实测记录，每次跑完一行就更新状态与指标。

---

## 1. 数学全排列

对 `world_size ∈ {1, 2, 4}`，枚举所有正整数三元组 `(tp, dp, pp)` 满足 `tp × dp × pp = world_size`。`SP ∈ {on, off}` 仅在 `tp ≥ 2` 时两态，否则仅 `off`。

不预先排除任何组合——即使某些组合（如 PP=2 on 2 GPUs）性能差或可能 OOM，也要实测记录，摸清框架功能全貌。

| # | W | TP | DP | PP | SP | 描述 |
|---|:--:|:--:|:--:|:--:|:--:|------|
| 1 | 1 | 1 | 1 | 1 | off | 单卡 baseline |
| 2 | 2 | 1 | 1 | 2 | off | PP=2 |
| 3 | 2 | 1 | 2 | 1 | off | DP=2 |
| 4 | 2 | 2 | 1 | 1 | off | TP=2 |
| 5 | 2 | 2 | 1 | 1 | on | TP=2+SP |
| 6 | 4 | 1 | 1 | 4 | off | PP=4 |
| 7 | 4 | 1 | 2 | 2 | off | DP=2+PP=2 |
| 8 | 4 | 1 | 4 | 1 | off | DP=4 |
| 9 | 4 | 2 | 1 | 2 | off | TP=2+PP=2 |
| 10 | 4 | 2 | 1 | 2 | on | TP=2+PP=2+SP |
| 11 | 4 | 2 | 2 | 1 | off | DP=2+TP=2 |
| 12 | 4 | 2 | 2 | 1 | on | DP=2+TP=2+SP |
| 13 | 4 | 4 | 1 | 1 | off | TP=4 |
| 14 | 4 | 4 | 1 | 1 | on | TP=4+SP |

**14 组合 × 2 模型（9B/27B）× 2 方法（RL/SFT）= 56 格。**

## 2. 环境

| 项 | 值 |
|---|---|
| 机器 | 单节点 4× NVIDIA A800 80GB PCIe（GPU 0-3） |
| 模型 | `Qwen3.5-9B`（`/data/zhangzy/models/Qwen3.5-9B/`）、`Qwen3.8-27B`（`/data/zhangzy/vllm/Qwen3.8-27B/`） |
| 数据 | 统一多模态数据集：228 v4 分层抽样 2 链 × 8 类别 = 37 行、78 图，`samples/data/tool_call_mm/train.jsonl` |
| 运行 | `bash tests/e2e/run_matrix.sh <output_dir>`（输出目录必填） |
| 超时 | 1200s/格 |
| 配置 | `samples/configs/matrix/*.yaml`（56 个，由 `tests/e2e/generate_matrix.py` 生成） |
| 镜像 | `graspo:0.29.0` |

> 多卡自动 `NCCL_P2P_DISABLE=1`（A800 PCIe 拓扑必需）。

## 3. 完整矩阵（56 格）

状态图例：✅ 通过 ｜ ⚠️ 超时/部分完成 ｜ ❌ 失败 ｜ — 未测

| # | W | TP | DP | PP | SP | 9B RL | 27B RL | 9B SFT | 27B SFT | 备注 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|------|
| 1 | 1 | 1 | 1 | 1 | off | ✅ | ⏰ | ❌ | ✅ | RL: single-GPU OK; SFT 9B OOM / 27B OK |
| 2 | 2 | 1 | 1 | 2 | off | ❌ | ❌ | ❌ | ❌ | RL: cross-PG deadlock; SFT: OOM/rc=1 |
| 3 | 2 | 1 | 2 | 1 | off | ❌ | ❌ | ❌ | ❌ | RL: cross-PG deadlock; SFT: SIGABRT |
| 4 | 2 | 2 | 1 | 1 | off | ❌ | ❌ | ✅ | ❌ | RL: deadlock; SFT 9B OK / 27B rc=1 |
| 5 | 2 | 2 | 1 | 1 | on | ❌ | ❌ | ✅ | ❌ | RL: elastic error; SFT 9B OK / 27B rc=1 |
| 6 | 4 | 1 | 1 | 4 | off | ❌ | ❌ | ✅ | ❌ | RL: deadlock; SFT 9B OK / 27B rc=1 |
| 7 | 4 | 1 | 2 | 2 | off | ❌ | ❌ | ❌ | ❌ | RL: deadlock; SFT: rc=1 |
| 8 | 4 | 1 | 4 | 1 | off | ❌ | ❌ | ❌ | ❌ | RL: deadlock; SFT: SIGABRT |
| 9 | 4 | 2 | 1 | 2 | off | ❌ | ❌ | ✅ | ✅ | RL: deadlock; SFT: OK |
| 10 | 4 | 2 | 1 | 2 | on | ⏰ | ⏰ | ✅ | ✅ | RL: timeout; SFT: OK |
| 11 | 4 | 2 | 2 | 1 | off | ❌ | ❌ | ❌ | ❌ | RL: deadlock; SFT: SIGABRT |
| 12 | 4 | 2 | 2 | 1 | on | ❌ | ❌ | ❌ | ❌ | RL: elastic; SFT: rc=1 |
| 13 | 4 | 4 | 1 | 1 | off | ❌ | ❌ | ✅ | ✅ | RL: deadlock; SFT: OK |
| 14 | 4 | 4 | 1 | 1 | on | ❌ | ❌ | ✅ | ✅ | RL: elastic; SFT: OK |

## 4. 运行方法

```bash
# 生成 configs 和 run 脚本（开发机）
python3 tests/e2e/generate_matrix.py

# 同步到 121 后运行（输出目录必填）
bash tests/e2e/run_matrix.sh /data/zhangzy/e2e-results
```

每格在 `<output_dir>/<name>/` 下保存 `log.txt`（完整日志）和 `outputs/`（训练产物）。汇总摘要写入 `<output_dir>/summary.txt`。

## 5. 通过判据

1. **单卡 baseline 存在**：#1 通过，作为能力对齐参照。
2. **能力对齐**：同 seed + 同数据、同一 epoch 下，跨并行配置的 loss / reward 与单卡 baseline 对齐（容差内）。权重差异可接受。
3. **逐卡显存/负载均衡**：`max(peak_memory) − min(peak_memory) < 5% of mean`。
4. **无 hang / OOM / NaN**：训练正常推进，`error.log` 为空或仅有预期告警。
5. **可复现**：同 config + seed 两次运行落盘 loss 可比。

## 6. 结果记录

| 日期 | # | 模型 | 结果 | 耗时(s) | loss | reward | 备注 |
|------|---|------|------|---------|------|--------|------|
| 08-25 | 1 | 9B | ✅ | ~900 | -0.19 | 0.44 | 单卡 baseline，唯一通过 |
| 08-25 | 2-4,6-9 | 9B | ❌ | — | — | — | 多卡 SIGABRT（cross-PG deadlock） |
| 08-25 | 5 | 9B | ❌ | — | — | — | SP 变体，torch elastic error |
| 08-25 | 10 | 9B | ⏰ | 1200+ | — | — | SP 变体，训练卡死超时 |
| 08-25 | 11-14 | 9B | — | — | — | — | 运行中/待测 |
| 08-25 | 1-14 | 27B | — | — | — | — | 待测 |

> 记录方法：每格先跑单卡 baseline（#1）得到该 epoch 的 loss/reward，再跑相应并行组合，填入同一 epoch 的指标用于能力对齐判定。

## 7. 已知问题

### 7.1 多卡 cross-PG deadlock（#2-4, #6-9）

**症状**：所有 2+ 卡配置（DP/TP/PP > 1）在训练中期 SIGABRT 崩溃，exit code -6。

**根因**：跨进程组死锁。rank 0 和 rank 1 在不同 PG 上互相等待——一个在 PG 1 等 all-reduce，另一个在 PG 0 等 barrier。训练跑了 ~36,000 次 NCCL 操作后触发，非初始化问题。

**排查结论**：
- `NCCL_P2P_DISABLE=1` 已设置，不是 P2P hang
- GPU 0↔1 是 NVLink（NV8），不是 PCIe 拓扑
- 这是训练代码的条件分支发散导致的确定性 bug（工作日志中记录的 "DP=4 post-epoch hang"）
- **P2 阶段修复范围**

## 8. 配置生成

所有 config 和 run 脚本由 `tests/e2e/generate_matrix.py` 自动生成，不手工维护。修改测试参数（超时、数据路径、模型路径等）只需改脚本中的常量，重新运行即可。

生成的 config 位于 `samples/configs/matrix/`，run 脚本位于 `tests/e2e/run_matrix.sh`。