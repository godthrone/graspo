# `samples/configs/` — 开箱样例配置目录

> **本目录是什么**：一组**可直接拿来跑**的训练/评测配置样例（YAML），按「单机单卡 → 多机多卡」
> 与「SFT / RL / 评测 / ms-swift」铺开，供新用户照抄改参数。
>
> **本目录不是什么**：**不是**任何实验台账或取数口径的落点；本仓库不承载实验跑次记录。
> 本文件只讲**样例怎么用**。
>
> 落位：2026-09-30（R13 仓库范围收敛后重写；旧版是实验取数口径说明，其主体已按宪法 §16.1 移出仓库）。

## 目录内容（36 份 YAML）

| 类别 | 件数 | 文件 | 说明 |
|---|---:|---|---|
| 最小示例 | 2 | `config_example.yaml`、`config_example_msswift.yaml` | 字段最少的可跑配置；后者需 `pip install graspo[msswift]` |
| 按模式的最小示例 | 3 | `sft_example.yaml`、`rl_example.yaml`、`eval_example.yaml` | 单卡 SFT / 单卡 RL / 评测 |
| 单卡样例 | 2 | `a800x1_qwen35_9b_rl.yaml`、`a800x1_qwen35_27b_rl.yaml` | 1×A800 |
| 单机多卡样例 | 21 | `a800x4_qwen35_*.yaml` | 4×A800；覆盖 DP / TP / PP / SP 与 RL 组合 |
| 8 卡样例 | 1 | `a800x8_qwen35_9b_tp1_dp8_pp1.yaml` | 8×A800 |
| 端到端冒烟 | 7 | `e2e_*.yaml` | 极小规模、用于快速验证环境是否通（含 `e2e_smoke.yaml`） |

## 怎么用

```bash
# 单卡冒烟（先验证环境）
bash run.sh samples/configs/sft_example.yaml --smoke

# 多卡（把配置里的并行度/卡数按机器改好）
bash run.sh samples/configs/a800x8_qwen35_9b_tp1_dp8_pp1.yaml --gpus 0,1,2,3,4,5,6,7
```

- 配置字段的完整定义见 `src/graspo/core/schema.py`（**唯一真相源**，§1.4）；
- 框架总览与安装见仓库根 `README.md` / `README.zh-CN.md`；
- 架构与分层见 `docs/architecture.md`；算法层见 `docs/ripple.md`；设施层见 `docs/flow.md`。

## 纪律

- 这些 YAML 是**样例**，不是冻结件：按自己的机器与模型路径改即可。
- 想新增样例：放本目录、命名自解释，并在**本文件**的表格里补一行（单一真相源就在本文件）。
