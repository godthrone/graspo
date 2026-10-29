# scripts/host/ —— 主机侧 harness

本目录存放**只在评测/训练主机上直接执行、不进容器**的脚本（与 `scripts/` 下的 Python 工具分开，
避免"主机脚手架"和"可移植代码"混在一起，也让这些入口进入版本控制、跑次可复现）。

## `eval_a800.sh` —— 档级评测入口（chroot 版，无 docker）

```bash
# 用法
bash scripts/host/eval_a800.sh <TIER> <FORM:adapter|full> <CKPT> <SERVE_GPUS> [SIDE:both|after]

# 例 1｜评底座（FORM=full，CKPT 直接给底座模型目录）
bash scripts/host/eval_a800.sh M016 full /models/Qwen3.5-9B 2,3 both

# 例 2｜评某个 LoRA checkpoint（先 merge 成完整 HF，再由 vLLM 服务）
bash scripts/host/eval_a800.sh M016 adapter /runs/M016-pdb2-diag/v0-XXXX/checkpoint-400 2,3 both
```

| 入参 | 含义 |
|---|---|
| `<TIER>` | 档号（如 `M016`），决定数据集、评测配置与输出目录 `$GRASPO_EVAL_HOST_ROOT/eval/<TIER>` |
| `<FORM>` | `adapter` = 先 merge LoRA 再评；`full` = 零 merge，直接评给定目录 |
| `<CKPT>` | **容器内路径**（如 `/runs/...`）或模型目录（`/models/...`） |
| `<SERVE_GPUS>` | 给 vLLM 的卡，如 `2,3` |
| `[SIDE]` | `both`（默认，先 after 再 base，产出 Δ）或 `after`（只评 after） |

**环境变量（可选，默认值 = 原先硬编码值，行为零变化）**

| 变量 | 默认 | 说明 |
|---|---|---|
| `GRASPO_EVAL_HOST_ROOT` | `/data/remote` | 主机侧数据根（rootfs、eval 目录、mounts 清单都在其下） |
| `GRASPO_EVAL_PORT` | `8103` | vLLM 端口，须与内部 `gen_eval --port` 同源 |

**产出**：`$GRASPO_EVAL_HOST_ROOT/eval/<TIER>/` 下的 `judge4.json`
（`base_acc` / `after_acc` / `delta_pp` / `parse_rate` / `served_by_vllm`）+ `evidence/` 各步日志。

**执行前提**（缺一不可）：`$GRASPO_EVAL_HOST_ROOT` 下有 `rootfs/{graspo,vllm}`、`mnt-eval.json`、`run_in_rootfs.py`，
且 `eval/<TIER>` 的数据子集与 `mnt-eval.json` 的挂载点一致；vLLM 由本脚本自行起停。

**已知边界**

- `delta` 的 `passed` 字段由工具按固定阈值（如判据② 的 `+5.0 pp`）**无条件套算** ⇒ 只对**等量训练**的矩阵跑有效；
  诊断跑/短档跑引用 `passed` 属口径错误（本次 M016-pdb2 诊断跑即为此踩坑点）。
- `accuracy_percent_excluding_overlap` 在**训练集自评（fit 口径）**时退化：全部样本都在 overlap 里 ⇒ 该字段恒为 0，不可引用。
- 历史变更：2026-10-29「仅挪 delta 顺序」——原顺序把 `graspo eval delta` 排在报告副本拷贝之前，
  delta 读 `$EV/.local/reports/` 副本必然 `not found`（`delta rc=1`），**计算逻辑与阈值未改**。
  原主机版本备份为 `eval_a800.sh.bak-20261029-1740`（md5 `0da546a54332b21a28ed1fb1b024ef8b`）。
