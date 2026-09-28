# `samples/configs/` — 配置目录用途与取数口径

> 本文件只说明**各目录是什么、谁用于取数**；**不改变任何既有配置文件的内容**。
> 落位：2026-09-28（主席收尾清单第 5 项「样例配置核查」）。

## 取数口径（唯一权威）

**本轮能力矩阵 `docs/capability-matrix.html` 的取数配置目录 = `samples/configs/matrix54-v2-runasrun/`**（54 件：51 份 `T0NN.yaml` + 3 份 `T05{2,3,4}.not_applicable.md`）。

该目录是**合并目录**：

| 来源 | 覆盖档 | 说明 |
|---|---|---|
| `samples/configs/matrix54-v2/T0NN.yaml` | 其余 **42 档**（逐字节相同） | DeepSpeed 从未出现在这些档（ms-swift 非 DS 组 / native 组） |
| `samples/configs/matrix54-fsdp/T0NN.yaml` | T004/T005/T006/T019/T020/T021（**逐字节相同**） | FSDP2 阶段 1/2（4 卡无 offload；1/2 卡 + CPU offload） |
| `samples/configs/matrix54-fsdp/T0NN-noreshard.yaml` | T037/T038/T039/T049/T051（逐字节相同） | RL6 派生件（`reshard_after_forward: false`） |
| `samples/configs/matrix54-fsdp/T050-noreshard-noteacheroffload.yaml` | T050（逐字节相同） | E2/RB-12 派生件（另 `distill.offload_teacher_model: false`） |

⇒ **12 档**与冻结基线不同，差异**仅在后端键**：`msswift.deepspeed: zero2[_offload]` → `msswift.fsdp: samples/configs/fsdp/…fsdp2_full_state_dict[_offload].json`。
（T050 另多一行 `distill.offload_teacher_model: true → false`，理由见该文件尾部的 E2/RB-12 专段。）

合并配方与逐档替换清单见 `_tools/LEDGER-ENTRY-multi-root.final.md` §36.2。

## `matrix54-v2/` = 冻结历史基线，**不再用于取数**

`samples/configs/matrix54-v2/` 是 **2026-09-23 冻结的 v2 定稿**（总指纹
`7a230bad535a79cbd9718c7169a1fbabac156948b5bb991a762ee3bcca6f0e47`，54 件）。
其 12 档仍写 `msswift.deepspeed: zero2[_offload]`，**与实际运行配方（FSDP2）不符** ⇒

- **保留**：作为历史基线与可比性维度；
- **不再用于取数**（HTML 配置列的权威 `--config-dir` 是 `matrix54-v2-runasrun/`）；
- **一字未改**：**不得**在该目录内新增、修改或删除任何文件 —— 该目录的总指纹按「目录内全部文件」计算
  （`rig/fp_v2.py`；`scripts/env_snapshot.py::collect_config`），**加一个文件也会改变指纹**，
  而该指纹是被引用的冻结维度（「变则本轮档不可比」）。

## 复现命令（照抄，只读）

```bash
cd /home/<user>/Gitlab/graspo

# ① 冻结基线的总指纹（应为 7a230bad535a79cbd9718c7169a1fbabac156948b5bb991a762ee3bcca6f0e47）
.venv/bin/python rig/fp_v2.py samples/configs/matrix54-v2

# ② 取数目录的目录指纹（**现行**应为 58c999ddca048ef583d8a72f78db56f1bb0a4d4a9babe5f92b6d8f6dfebe45b4）
#    （沿革：8a146cfa…567e7 是建目录当时值；2026-09-28 对本目录 T050.yaml 做了一处**注释级**更正
#      —— 更正 `:10` 与权威键 `:42` / 实跑 args.json 的矛盾，**未改任何键值** —— 之后即为现行值）
cd samples/configs/matrix54-v2-runasrun && ls | sort | xargs sha256sum | sha256sum && cd -

# ③ 取数目录 ↔ HTML 配置列逐格机核（rc=0 为通过）
.venv/bin/python tests/e2e/verify_matrix_html.py \
  --html docs/capability-matrix.html \
  --config-dir samples/configs/matrix54-v2-runasrun
```

## 其它目录（本轮口径未变）

| 目录 | 用途 |
|---|---|
| `samples/configs/matrix54/` | v1 基线（历史） |
| `samples/configs/matrix54-fsdp/` | FSDP2 派生件的**源目录**（由 `tests/e2e/generate_matrix_fsdp.py` 与 `_tools/make_rl6_derivatives.py` 生成，手改无效） |
| `samples/configs/matrix/` | 早期 28 网格并行矩阵 |
| `samples/configs/fsdp/` | FSDP2 配方 JSON（`fsdp2_full_state_dict[_offload].json`，RL6 版在 `fsdp/rl6/`） |
| 其余 `*.yaml` | 单机/多机开箱样例（`README.md` / `README.zh-CN.md` 指引），与能力矩阵无关 |
