# native 后端两处训练链缺陷修复（N1 / N2）

> 本文件是**唯一真相源台账**；工位副本 `.local/hb-workspace/20261022-baseline-r1/task-fix-native-adapter/DELIVERY.md`
> 与它**逐字节相同**（`cmp` 通过；工位副本 = 本文件的 `cp`）。
> 范围声明：本轮**只修 bug**——判据／期望值／奖励口径／矩阵数值**一字未动**，零签字。

## 1. 结论

| 缺陷 | 一句话 | 状态 |
|---|---|---|
| **N1** | native 建模型时不按 `tuner_type` 分支，**全参档也去解析 LoRA 正则** ⇒ `ValueError: lora.target_modules is required` | ✅ 已修 ＋ 单测红→绿 ＋ 228 真机 9B 单卡全参跨 35 步并落 3 个 ckpt |
| **N2** | `pp_size>1` 时归属豁免只覆盖视觉塔一层，**语言侧层归属完全没豁免** ⇒ 别的 stage 的层被当成"本机未实现" | ✅ 已修 ＋ 单测红→绿 ＋ 228 真机 9B/LoRA/pp=2 自然跑完 100 步（RC=0） |

命中面（原生 native 后端）：**N1 = 8 档**（全参+native：M011 M014 M015 M026 M029 M030 M051 M062）；
**N2 = 22 档**（native+pp>1：M004 M005 M006 M009 M014 M015 M019 M020 M021 M024 M029 M030 M044 M045 M046 M049 M051 M055 M056 M057 M060 M062）。

## 2. 缺陷 N1：全参档无条件解析 LoRA 目标

### 2.1 现象（228 实测，2026-10-22）

M011（9B / SFT / **全参** / native / 1 卡）334 s 崩溃，`runs/M011/stdout.log:20` 逐字：

```
ValueError: lora.target_modules is required (a single real HF module-name regex; both
backends share it). lora.target_preset no longer selects LoRA targets …
```

抛出点 `src/graspo/core/lora.py:169-170`（`if target_modules is None: raise ValueError(...)`）。

### 2.2 真因（file:line）

| 位置 | 事实 |
|---|---|
| `src/graspo/flow/backend/native/models/qwen35_36/model_adapter.py:61-66`（HEAD） | `resolve_lora_target_modules(self.config.lora.target_modules, …)` **无条件**调用 |
| 同文件 `:78`（HEAD） | `full_param = self.config.effective_tuner_type == "full"` —— 判定**在解析之后**才发生 |
| `src/graspo/flow/backend/native/models/qwen3/model_adapter.py:47-50`（HEAD） | 同一处缺陷（同族 dense 适配器） |

**事实更正（行号级）**：任务书写「调用来源 `models/qwen3/model_adapter.py:61`」。HEAD 上该文件 `:61`
是 `build_placement_plan(...)` 里的 `if self.config.native.layer_ranges is not None`，**无条件解析在 `:47`**；
行号 `:61` 精确对应的是 **`models/qwen35_36/model_adapter.py:61`**。9B = `Qwen3.5-9B` ⇒ `model_type=qwen3_5_text`
⇒ 走 `Qwen35ModelAdapter`（`models/common/model_builders.py:34-41`）。缺陷本体与描述一致，仅目录名/行号需按上述对齐。

### 2.3 修法

`*_load_model()` 里把 `full_param` 判定**提到解析之前**，全参档不解引用：

```python
full_param = self.config.effective_tuner_type == "full"
lora_targets = (
    None if full_param
    else resolve_lora_target_modules(self.config.lora.target_modules,
                                     available=native_qwen_lora_available_targets(hf_config))
)
…
lora_targets=set(lora_targets.resolved) if lora_targets is not None else set(),
```

- 全参档：**不读、不校验、不编译** `lora.target_modules`；传给模型的 target 集合为空（`lora_r=0` 语义不变，
  `models/qwen35_36/model.py:64`、`models/common/model_builders.py:167`）。
- LoRA 档：**逐字不变**（同一正则、同一 `available` 域、同一 `resolved` 集合）——由 golden 用例锁死。

## 3. 缺陷 N2：`pp_size>1` 时归属豁免缺失

### 3.1 被推翻的旧诊断

旧诊断「9B LoRA 注入空集」**不成立**：实测 `pp=1` 的 native 档能训（M007 400 步、M008 200 步、27B M041 通过），
而报同一句错的档**全部 `pp>1`**。

### 3.2 真因（file:line）

| 位置 | 事实 |
|---|---|
| `src/graspo/flow/backend/native/parallel/placement_plan.py:15,74-85,126-136` | 本 stage 持有层的**唯一真相源**：`local_layer_indices` / `include_embeddings` / `include_lm_head`；`is_pipeline = pp_size > 1` |
| `src/graspo/flow/backend/native/lora/lora_helpers.py:20-123` | `native_qwen_lora_available_targets()` 枚举**整模型**可命中模块（正则匹配域，与 ms-swift/PEFT 逐字相同） |
| `src/graspo/flow/backend/native/models/common/base.py:179-197` | `enabled_lora_hf_module_paths()` 只返回**本 stage 真建出来**的 `LoRALinear` |
| `models/qwen35_36/model_adapter.py:120-125`（HEAD） | 差集只减 `exempted_visual_targets`（判据 = `include_embeddings`），**语言侧层归属没有任何豁免** ⇒ stage>0 上「别的 stage 的层」恒被判成"本机未实现" |

两个集合的**范围不同域**（`resolved` = 整模型 vs `enabled` = 本 stage）⇒ `pp>1` 必然报
`Resolved LoRA target(s) are not implemented by this model yet: model.language_model.layers.0…`。

### 3.3 修法：按 stage 归属豁免（判据不放宽）

新增纯函数 `src/graspo/flow/backend/native/lora/lora_helpers.py`：

```python
def lora_target_layer_index(hf_module_path) -> int | None      # `.layers.<N>.` 取层号；无层号 → None
def stage_exempt_lora_targets(targets, *, local_layer_indices, is_pipeline) -> frozenset[str]
```

**边界（三条，缺一即会静默放过真错的配置）**：

1. 只剔除**有层号且该层号 ∉ 本 stage `local_layer_indices`** 的目标；
2. 无层号的目标（embeddings / lm_head / `model.visual.*`）**一律不剔除**——视觉塔仍由既有
   `exempted_visual_targets`（`include_embeddings` 判据）管，两者各自同域、互不替代；
3. `is_pipeline=False` ⇒ **恒空集**，`pp=1` 什么都不豁免。

覆盖性论证：每条 target 恰好归属某一层，因而恰好由某一个 stage 负责建成 ⇒ 豁免只把"谁该建"摊销到正确的
stage，不会让空集/漏配静默通过。`qwen3`（dense，仅 `manual layer_ranges` 可达 `pp>1`）同族同缺陷一并修，判据同一份。

## 4. 单测：修复前 = failed（原文），修复后 = 绿

`tests/flow/backend/native/lora/test_native_adapter_lora_targets.py`（9 例，无 GPU / 无权重 / 无真 checkpoint；
期望集合由本文件**独立枚举**，不复用被测函数）。

修复前（临时把 3 个 src 文件退回 HEAD 版，跑完原样恢复，`sha256sum -c` 通过）：

```
FAILED …::test_full_builds_model_without_lora_targets
FAILED …::test_full_tolerates_missing_target_modules_key
FAILED …::test_full_does_not_even_compile_the_lora_regex
PASSED …::test_lora_path_resolves_the_same_targets_as_before
PASSED …::test_pp1_does_not_exempt_anything
FAILED …::test_pp2_exempts_layers_this_stage_does_not_hold
FAILED …::test_pp2_still_fails_when_an_owned_layer_is_missing
FAILED …::test_qwen3_full_builds_model_without_lora_targets
FAILED …::test_qwen3_pp2_manual_ranges_exempt_layers_outside_the_stage
========================= 7 failed, 2 passed in 1.69s ==========================
```

修复前失败的两句原文（与 228 证据同源）：

```
E  ValueError: lora.target_modules is required (a single real HF module-name regex; …)
E  ValueError: lora.target_modules='^model\.visual\..*$' matched none of the model's LoRA-able modules
E  ValueError: Resolved LoRA target(s) are not implemented by this model yet: model.language_model.layers.0.mlp.down_proj, …
```

修复后：`9 passed in 2.39s`。其中 **`test_lora_path_resolves_the_same_targets_as_before`（golden）** 与
**`test_pp1_does_not_exempt_anything`** 在修复前后**都通过**——这正是"LoRA 路径行为不变"与"pp=1 不豁免"的锁。

用例 ↔ 验收点对照：

| 验收要求 | 用例 |
|---|---|
| N1: full ⇒ 不解析 LoRA 且能建模型 | `test_full_builds_model_without_lora_targets` |
| N1: full 缺 `lora.target_modules` 字段 ⇒ 不报错 | `test_full_tolerates_missing_target_modules_key` |
| N1: full 连"零命中正则"都不解析 | `test_full_does_not_even_compile_the_lora_regex` |
| N1: lora ⇒ 行为不变 golden | `test_lora_path_resolves_the_same_targets_as_before` |
| N2: pp=1 ⇒ 不豁免 | `test_pp1_does_not_exempt_anything` |
| N2: pp>1 ⇒ 本 stage 未持有的层被豁免 | `test_pp2_exempts_layers_this_stage_does_not_hold` |
| N2: pp>1 但真该命中的层缺失 ⇒ 仍报错（且只点名该层） | `test_pp2_still_fails_when_an_owned_layer_is_missing` |
| 同族同缺陷（qwen3 dense） | `test_qwen3_full_builds_model_without_lora_targets` / `test_qwen3_pp2_manual_ranges_exempt_layers_outside_the_stage` |

## 5. 回归对照

为避免同仓**另一执行者对 `scripts/**`、`tools/matrix/**` 的在途改动**污染读数，对照跑在
`git archive HEAD` 的**隔离导出**里（`.local/worktrees/{head,postfix}`；postfix = HEAD ＋ 本包 4 件）。

| 套件 | 修复前 | 修复后 | 判定 |
|---|---|---|---|
| `pytest tests/tools tests/scripts` | **340 passed / 2 failed** | **340 passed / 2 failed** | 红名单逐条相同（`test_t023_final_formats_e2e` × 2，既有红）⇒ 零新增红 |
| `pytest tests/flow/backend/native tests/core/test_lora.py` | **575 passed / 4 skipped** | **584 passed / 4 skipped** | ＋9 = 本包新增；红 0、skip 数不变 |
| `pytest tests/test_ast_boundary.py tests/compliance tests/core` | 468 passed / 13 failed | 468 passed / **13 failed（同一份红名单，`diff` 为空）** | 零新增红 |
| `ruff check`（本包新增/改动行） | — | **All checks passed** | 两处 adapter 的 `I001`×2 / `E501`×1 经与 HEAD 版逐条比对为**既有**，未新增 |

⚠️ **工作包写的 339 passed / 3 failed 与实际环境的差异（如实登记）**：主工作树当时是
**338 passed / 4 failed**，多出的两条是
`tests/scripts/test_entry_smoke.py::test_entry_scripts_smoke_all_covered_and_broken_zero` 与
`tests/scripts/test_scripts_no_threshold_literals.py::test_scripts_have_no_threshold_literal_comparison`，
成因是**另一执行者未提交的 `scripts/mk_readings.py`、`scripts/step_readings.py`＋两个改动件**（报错直接点名
`filename = '…/scripts/mk_readings.py'`）。HEAD 干净导出里这两条**不红** ⇒ 与本包改动无关，
且本包**未触碰 `scripts/**`、`tools/matrix/**`、`docs/**` 任何文件**。未出现任何 skip/xfail 新增，未改任何期望值。

## 6. 真机最小冒烟（228，非全量、无评测、未填矩阵）

前置：新树 `/data/remote/graspo-baseline-r1-fix-85dcab8/`；卡池**先到先得**，起跑前逐卡跑
`scripts/gpu_idle_assert.py --visible <卡> --position-mib 49152`（只读闸门，rc=0 才起）；⛔ 未停/未动任何他人容器与进程。

### 6.1 N1：9B / SFT / **全参** / native / 单卡（M011 形状）

| 项 | 值 |
|---|---|
| 跑次目录 | `/data/remote/graspo-baseline-r1-fix-85dcab8/smoke/runs/M011/` |
| 容器 | `graspo-smoke-M011-1793102529`，`NVIDIA_VISIBLE_DEVICES=2`，`nproc=1` |
| 优化步 | **35 步**（`sft_step` 事件 35 条；末事件 `"step": 35`，`stdout.log:428`；首事件 `step 1` 在 `:20`） |
| 崩溃是否消失 | `grep -c 'lora.target_modules is required' stdout.log` = **0** |
| checkpoint | **3 个**：`M011-smoke/step_10`、`step_20`、`step_30`，各 `rank_00000_tp_00_pp_00.pt` = 56,459,833,729 B ＋ `manifest.json` |
| 耗时 / 退出码 | **ELAPSED_SEC=803（13.4 min）**、**RC=137** ＝ 达标后**我方具名 `docker stop`**（TERM 10 s 未退 ⇒ KILL），非训练失败；单步 `batch_sec≈12.7 s`、显存 `allocated≈35.1 GB`、宿主侧 GPU2 峰值 42,267 MiB |
| 全参语义自证 | `step_30/manifest.json`：`"format": "native-full-param"`、`"tuner_type": "full"`、`"lora_target_signature": {"resolved": [], "parameter_count": 0}` ⇒ 全参档确实以**空 LoRA 目标集**建模 |

### 6.2 N2：9B / LoRA / **pp=2** / dp=1（最便宜 pp>1 档 = M004）

| 项 | 值 |
|---|---|
| 跑次目录 | `/data/remote/graspo-baseline-r1-fix-85dcab8/smoke/runs/M004/` |
| 容器 | `graspo-smoke-M004-1793103376`，`NVIDIA_VISIBLE_DEVICES=2,3`，`nproc=2` |
| 步数 | **100/100 步自然跑完**（`SFT complete: steps=100 elapsed=428.6s`），门槛只要 ≥5 |
| 崩溃是否消失 | `grep -c 'Resolved LoRA target' stdout.log` = **0**；PP 真在通信：`[pp-debug rank=0/1] … pp_size=2` 收发帧 |
| checkpoint | 每 5 步一个：`step_5 … step_95`（每个 553 MB = LoRA 形态，与 pp2 预期一致） |
| 耗时 / 退出码 | **ELAPSED_SEC=439**、**RC=0 / `exit_code` = 0**（自然退出，容器 `--rm` 自动清） |

### 6.3 冒烟配置口径（未污染冻结档）

冒烟配置由**树内冻结档**派生（`smoke/make_smoke_configs.py`，逐键断言"只有下列 5 行不同"）：

| 档 | 差异 |
|---|---|
| M011-smoke | `training.max_epochs 4→1`、`training.save_steps 401→10`、`output_dir /out/M011→/out/M011-smoke`、`run_name`、`gpu_monitor.tag` |
| M004-smoke | `training.max_epochs 4→1`、`training.save_steps 201→5`、`output_dir /out/M004→/out/M004-smoke`、`run_name`、`gpu_monitor.tag` |

理由：`save_steps` 若维持 401/201，30 min 上限内**永远等不到 checkpoint**（验收要求"落一个 checkpoint"）；
`max_epochs` 收成 1 让冒烟**自然收敛退出**而不是被超时砍。
**树内冻结档未被改动**：`samples/configs/matrix68/M011.yaml` = `7438d760d05a9c3a…`、
`M004.yaml` = `061018ad830fd634…`（远端与本地逐字相同）。训练子集亦未改：
`subsets-v2/M011.jsonl` = `5ad2d7cc588bab7b…`（100 行）。

## 7. 228 投放与清理

| 项 | 值 |
|---|---|
| 新树 | `/data/remote/graspo-baseline-r1-fix-85dcab8/`（**1096 文件 / 50 M**；`git archive 85dcab80054cd2c8d02a131b94c9c5ee4395f5a4`） |
| tar | `graspo-baseline-r1-fix-85dcab8.tar`，12,749,567 B（gz），远端 sha256 = `53f93c2390…` 与本地同值 |
| 侧车 | `.env.sh`（`GRASPO_DEPLOY_COMMIT="85dcab8…"`、`GRASPO_REPO_ROOT`、`GRASPO_IMAGE=graspo:v0.28.11-cu130fix-v2-msgspec-20261012-run`）／`.hostdeps`（hardlink 复用）／`.eval-env.sh`(600) |
| 4 件逐文件核对 | 树内 ≡ 本地：`lora_helpers.py` `7d8ab242…`、`qwen3/model_adapter.py` `38fabf5d…`、`qwen35_36/model_adapter.py` `ad49cb94…`、`test_native_adapter_lora_targets.py` `28ad292e…` **4/4 相同** |
| 在役树/历史树 | ⛔ 零触碰（`graspo-baseline-r1-6189bd8` 等在盘不动，仅**只读**hardlink 复用其 `.hostdeps`） |
| 容器 | 训练容器 2 个，均已退出（M011 具名 stop、M004 `--rm` 自然退出）；`docker ps` 我方 = **0** |
| 八卡回落 | 起跑前 = `0:13108 / 1:10033 / 2:0 / 3:0 / 4:5555 / 5:11639 / 6:72149 / 7:77201` MiB；收尾后**逐卡完全相同** |
| 临时目录清理 | 4 个 `.pt` 证据后删除（3×53 G 全参 ＋ 19×553 M LoRA），`df` 3.9T→4.0T 可用；只留 `manifest.json`/`config.yaml`（抽到 `<run>/ckpt-evidence/`）；本地 `/tmp` 中转件已删 |
| 跑次证据路径（远端） | `…/smoke/runs/M011/{stdout.log,readings.txt,smoke_meta.txt,entry.sh,gpu/,ckpt-evidence/}`、`…/M004/{stdout.log,readings.txt,exit_code,smoke_meta.txt,entry.sh,gpu/,ckpt-evidence/}` |
| 证据回传（本机） | `.local/hb-workspace/20261022-baseline-r1/task-fix-native-adapter/evidence/228/{M011,M004}/` |

## 8. 未做 / 超出范围的发现（单列）

**未做（有意）**：
1. ⛔ 未跑全量矩阵、未做评测、未填矩阵数值、未跑 `sync_results`/`fill_one`；
2. ⛔ 未做 N3c 分片合并（2~4 天功能，未获批）；
3. ⛔ 未动 `docs/GRASPO训练框架功能测试报告.html`（改前/改后均 77,079 B、mode 444、sha `b8ed080d…`）；
4. ⛔ 未动 `scripts/**`、`tools/matrix/**`；本包提交 `--name-only` 不含这两目录任何文件；
5. ⛔ 未 push（分支无 upstream，`git status` 无上游配置）；
6. 冒烟只覆盖 2 档（N1 各 1 档 + N2 1 档），其余 8+22 命中档的**真实性仍由矩阵重跑判定**。

**超出范围的发现（只登记，未动）**：
1. `models/qwen3/model_adapter.py:6` 与 `models/qwen35_36/model_adapter.py:9` 的 **ruff `I001` 导入排序**、
   以及后者 `:16` 的 **`E501`（102 > 100）**，均为 **HEAD 既有**（已用 HEAD 版 ruff 输出逐条比对），未在本包顺手修（避免无关改动面）；
2. `core/lora.py` 的报错文案把 `lora.target_modules is required` 与"`target_preset` 已废弃"混在一句里，
   全参档误踩时容易误导游改配置——**文案层面**可改成按 `tuner_type` 分派，属独立小改，未纳入本包；
3. 主工作树 `tests/tools tests/scripts` 当时多出的 2 条红（见 §5 ⚠️）来自**另一执行者的在途 `scripts/**` 改动**，
   需在其提交后由其自证清零；
4. `models/qwen3/model_adapter.py` 走 `pp_size>1` 只能用 `native.layer_ranges`（`placement_plan.py:105-109`
   对 qwen3 族拒绝除 `manual` 外的 pipeline 策略）；本包的 stage 豁免对 manual 区间已实测覆盖，但该族
   `pp>1` 的端到端能力**未在真机验证**（不在本包范围）。

## 9. 复现与证据索引

```bash
# 单测（红→绿）
.venv/bin/python -m pytest tests/flow/backend/native/lora/test_native_adapter_lora_targets.py -v
# 回归对照（隔离导出，避开并行改动）
mkdir -p .local/worktrees/head && git archive HEAD | tar -x -C .local/worktrees/head
cd .local/worktrees/head && /abs/.venv/bin/python -m pytest tests/tools tests/scripts -q
# 228 冒烟（新树内，先到先得闸门在 run_smoke.sh 里）
cd /data/remote/graspo-baseline-r1-fix-85dcab8/smoke
bash run_smoke.sh M011 2 1 1800     # N1：全参单卡，≥30 步 ＋ ckpt
bash run_smoke.sh M004 2,3 2 1800   # N2：LoRA pp=2，≥5 步
```

本机证据：
`.local/hb-workspace/20261022-baseline-r1/task-fix-native-adapter/evidence/`
（`01_pre_fix_red_full9.txt` 修复前红原文／`04_post_fix_green.txt` 修复后绿／`02b`,`05` tools+scripts 对照／
`03_`,`06_` native 套件对照／`02c_`,`07_` core+compliance+ast 对照／`fixed-src/` 修复版 sha 清单／`228/` 真机跑次回传件）。

提交：`85dcab8` = 代码＋单测（4 件，+531/−12）；台账提交 = 本文件 `docs/native-lora-targets-fix.md`。
