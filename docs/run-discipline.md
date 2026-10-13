# 运行纪律（GPU 跑次）

本文件登记**跑次期间**必须遵守的判定与操作纪律。每条都对应一次实测事故，违反即产生**假结论**。

## 1. 在途跑次不得按"跑后产物"判失败（★ 容器 `Status=running` 优先）

跑次目录里这几件是**跑后**才写的产物，**不代表本次跑次的结局**：

```
<T>/exit_code              # 宿主在 docker run 返回后写
<T>/ckpt_retention.state   # entry.sh 收口段写
<T>/torch_probe.json       # entry.sh 收口段写
<T>/stdout.log             # 容器训练输出（本次可能仍在写）
<T>/attempt-<ts>/…         # 历史尝试的归档（见 §3）
```

**判定顺序（硬性）**：

1. 先看容器：`docker inspect <name> --format '{{.State.Status}} {{.State.ExitCode}}'`
   —— **`running` ⇒ 本次跑次在途，⛔ 不得据此判失败**；
2. 容器已退：只有 **`mtime >= 本次尝试的锚点`** 的产物才算本次的；
   **早于锚点的，一律视为陈旧，⛔ 不得据此判败**；
3. ★★ **锚点必须取"本次尝试真正开始的时刻"，首选 `docker inspect … {{.State.StartedAt}}`**，
   次选 runner 打印的 `[attempt-start] tier=<T###> launched_at=<UTC> run_dir=<path>`。
   ⛔ **不得只用 launcher 的"末次 `nohup 起跑` 时刻"**——**两次尝试重叠**时它会误判：
   实测 T033（fix8ep，2026-10-12）attempt1 的 `exit_code` mtime = 15:16:16，
   而 attempt2 的 `nohup 起跑` = **15:16:05**（更早！）⇒ "≥ 末次起跑"会把它判成"本次"，
   实际它属于 attempt1。用 **attempt2 容器 `StartedAt` = 15:16:30** 作锚点则正确判定为陈旧。
   ⇒ **判定优先级：容器 `StartedAt` > `[attempt-start]` 行 > launcher `nohup` 时刻**。

### 实测事故（2026-10-12，T033）

`runs/T033/exit_code` 内容 `1`、mtime **15:16:16**，而第二次尝试的容器
`loving_nobel` **Started 15:16:30 且仍在 running**（训练 ~84/320、指标健康）。
四件"失败证据"（`exit_code` 15:16:16 / `ckpt_retention.state` 15:16:13 /
`torch_probe.json` 15:16:14 / `status-T033/train.T033.stdout.log` 15:16:56）
**全部早于第二次起跑** ⇒ 它们是第一次尝试（`nostalgic_lederberg`，39 s 退出、
`exitCode=1`）留下的**陈旧产物** ⇒ **假失败**。
详见 `.local/hb-workspace/20261008-105451/task-t033-diagnosis/report.md`。

## 2. 重跑清场（B11）

同一 tier **重跑**时，`run_matrix54.sh` 在 `docker run` **之前**把上一轮的上述
**具名**跑后产物**移入** `<RUN_DIR>/attempt-<UTC 时间戳>/`：

- ⛔ **不用通配符**、⛔ 不删任何其他文件、⛔ 不触碰训练输出目录 `<RUN_DIR>/<T###>/`；
- 首次起跑（无旧产物）**不建**空的 `attempt-*` 目录；
- 归档是**证据**，不是垃圾：复核历史跑次时可从 `attempt-<ts>/` 取回上一轮的结局。

## 3. 与邻居共存（先到先得）

- 判断某张卡能不能用：**剩余显存 ≥ 本档头寸** ∧ **util == 0**（连续 2–3 帧采样全满足）；
- 运行期退让：本卡出现**非我方 PID** 的计算进程，或剩余显存**跌破安全余量** ⇒ 退让/停跑**上报**；
- ⛔ **不 kill 他人进程**、⛔ 不碰他人容器；自己的容器只**具名**停；
- 具名停请用**精确 PID/容器名**，并在日志里留一行说明（停的是哪一次尝试、为什么）。

## 4. 真因优先

- 判失败前先**取证**：`stdout.log`（entry.sh 已把 stderr 合并进来）、`logging.jsonl`、
  GPU 采样、`docker events`、`journalctl -k`（OOM）；
- ⛔ **不臆断**：没有显式 `Error/Traceback/OOM` 行 + 容器被信号杀死 ⇒ 结论是"被外部终止"，
  不是"训练逻辑失败"；
- 任一档的判定被推翻时，**同批**记录到本文件与交付页（下一位不必重踩）。

## 5. 起跑后多久才能判"未起/已退"

- ★ **判"未起/已退"必须等容器 `entry.sh` 落地与 stdout 起写，或据 `docker events` 的
  create/start 时序**；⛔ **不得在起跑后 ~20 s 内据"无容器/无 stdout"下"起后即退"结论**。
- 为什么：228 实测（2026-10-12 T038 冒烟）起跑后 ~20 s 观察 ⇒ "无容器、无 stdout" ⇒ 误判
  "起后即退"；实际容器 **19:46:41** 才落 `entry.sh`、**19:47** 起写 stdout，随后正常训练
  （DeepSpeed 关全过）。**观测过早 ≠ 失败**，误判会触发一次多余的"真因提取"与一次误报。
- 可操作判据（任一条满足即可下结论）：
  · `docker ps` 有目标容器 **且** `runs/<T>/entry.sh` 已落地（文件、非目录）；
  · `runs/<T>/stdout.log` 已在增长（隔 ~30–60 s 两次核对 size 变化）；
  · `docker events --filter event=die` 出现该容器名（这才是"真的退了"）。
- 同理：`exit_code` / `ckpt_retention.state` / `torch_probe.json` 都是**跑后**产物，
  在途跑次读到它们一律视为上一轮遗留（B11 runner 已把上一轮具名产物移入
  `runs/<T>/attempt-<ts>/` 以消歧）。

## 6. 评测服务档：容器命名与服务余量

两条都是 228 实测踩出来的运行编排缺陷（2026-10-13）。

### 6.1 服务容器名必须唯一，且**一个档同时只允许一个实例**

- ★ **容器名带唯一后缀**（如 `-$(date -u +%s)`，或按 `GPU`/批次区分）——
  ⛔ **不得复用固定名**（如 `t039-vllm-after`）。
- 为什么：同一条评测脚本被启动两次（第一次撞邻居后未彻底终止，第二次换卡重起）时，
  **两次共用同一个容器名** ⇒ **第一次实例的收尾 `docker stop <名>` 命中了第二次实例的容器**
  ⇒ 服务在评测中途被杀（`exit=137`、`oom=no value`），评测端只看到
  `http.client.RemoteDisconnected`，**真因离现场很远**（228 实测：T039，停机 01:37:48
  早于断连 01:37:59，而 `vllm` 日志里**没有任何 OOM/CUDA 错误**、结尾是正常停机序列）。
- ★ **收尾只停自己这一批的唯一名**；停机前先确认该容器就是本次实例（名字或 `docker inspect` 创建时间）。

### 6.2 服务档余量判据 = `util × 总显存`，⛔ 不是"头寸"

- ★ vLLM 这类**服务档**起跑前，按**该卡剩余 ≥ `util × 总显存`**（默认 `util=0.85` ⇒ 81920 MiB × 0.85 ≈ **69,632 MiB**）
  且 **`util == 0` 连续 3 帧**判定；必要时把 `util` 下调到 0.60–0.70 再判。
- ⛔ **不得沿用训练档的"头寸 48 GiB"判据**：训练档要的是"本档头寸"，而服务档要的是
  `util × 总显存`（KV cache 预占按整卡比例算）。
- 为什么：228 实测（T004 评测）—— GPU2 有第三方常驻 13,335 MiB ⇒ 仅剩 **67,818 MiB**，
  而 `util=0.85` 需要 ≈68 GB ⇒ vLLM **`RuntimeError: Engine core initialization failed`**，
  起不来。旧判据（剩余 ≥ 48 GiB ∧ util 0）**会放行**这种卡。
- 同一口径也适用于"卡上有邻居常驻"的一切服务类作业（评测 vLLM、探针服务等）。

## 7. 评测配置的 `checkpoint_path` 必须是**评测容器内可达**路径

- 规则：凡"**vLLM 直指全参目录、零 merge**"的档，写入 `eval-<TIER>-<role>.yaml` 的
  `checkpoint_path` **必须是评测容器内可达的路径** —— 即 `$TREE/runs` 被挂到 `/runs`
  （`run-eval.sh` 的 `-v "$TREE/runs":/runs:ro`）之后的 **`/runs/<TIER>/<TIER>/v0-*/checkpoint-*`**；
  ⛔ **不得写 vLLM 容器内的 `/model`**（评测容器没有该挂载）。
- **起跑前置断言（fail-closed）**：解析结果必须 **存在 ∧ 含 `config.json`**，否则拒绝起跑；
  解析一律用 **`v0-*` 通配**，⛔ 不用 `ls … | head -1`（会取到 `msswift/` 之类的目录名）。
- 为什么：228 实测**三次同类事故**（T039 / T004 / T005）——`checkpoint_path: /model` 让
  `after` 侧 `graspo eval run` 直接抛
  `graspo.eval.error.OrchestrationError: checkpoint directory not found: /model`，
  `after` 报告从未产出，随后 `delta` 又以 `DeltaError: eval report not found` **次生失败**；
  症状（"delta 挂了"）离真因（"after 从没跑成"）很远。
- 判读提示：**`after` 缺报告 + `delta` 报 `report not found`** ⇒ 先看 `after` 的 `eval run` 输出，
  ⛔ 不要只在 `delta` 上排查。

## 8. 产物新鲜度断言：⛔ 不得以"存在产物文件"判成功

- 规则：**合并/导出目录必须为空、或由本批新建**；⛔ **不得以"目录里存在产物文件"作为成功判据**。
- 为什么：228 实测（2026-10-13 T024 评测）——`/evalout/merged-hf-t024` 是**上一轮评测留下的
  root 属主半成品**（它确实含 `.safetensors`）。启动脚本里的守卫写成"`ls $MD/*.safetensors`
  有结果就算 OK"，于是：① 新 export 因 `FileExistsError: Output directory is not empty`
  **失败**；② 守卫**误判通过**；③ 脚本继续起 vLLM，**服务的其实是上一次跑次的合并权重**
  （权重与本跑次不同）⇒ 若跑完，就是一个**假读数**。已当场具名停容器，未产出读数。
- 三条修法（同批落地）：① **清掉本档陈旧 merged 目录**（容器内 root `rm -rf`，仅本档 merged，
  ⛔ 不碰训练输出）；② **export 配置加 `training.overwrite_output_dir: true`**；
  ③ **守卫改为"目录为空 或 本批新建"**。
- 与既有纪律同源：**"日志不可作判据""跑后产物不得当本次结果"** ⇒ 现在再加一条
  **"陈旧产物不得当本次成功"**。

## 9. 收尾进程：⛔ 禁用 `pkill -f`，一律精确 PID

- 规则：停止自己启动的进程**一律用精确 PID**（`kill <pid>`）；⛔ **禁止 `pkill -f <模式>`**。
- 为什么：228 **两次**实测（2026-10-12 与 2026-10-13）——`pkill -f` 的**模式命中了承载命令的
  ssh 外壳进程本身**，导致**当前会话以 255 退出**、后续清理动作被截断。第一次是"教训"，
  第二次仍复现 ⇒ 这**必须是硬纪律**，⛔ 不再靠"重申"。
- 配套：容器**只具名停**（且只停自己这一批的**唯一名**，见 §6.1）；需要连带停止派生进程时，
  先 `ps -ef` 取 PID，再逐个 `kill <pid>`。

## 10. 27B（Mamba）服务档：`--max-num-seqs` 必须 ≤ 可用 Mamba cache blocks

- 规则：用 vLLM 服务 **27B（Mamba 混合架构）**档时，**必须显式设 `--max-num-seqs`
  且其值 ≤ 该卡可用 Mamba cache blocks**（或改为**提高 `--gpu-memory-utilization`** 以换取更多
  blocks）。**base 与 after 必须同配**，并在跑次 json 的 `anomalies` 里**如实留痕**（服务参数变更、
  可能影响吞吐、**不改评测语义**）。
- **起跑前探针（fail-closed）**：读 vLLM 日志/配置给出的可用 blocks 数，断言
  `max_num_seqs ≤ blocks`，不满足即拒绝起跑。
- 为什么：228 实测（2026-10-13，T024 27B 评测）——vLLM 缺省 `max_num_seqs = 1024`，而单卡
  `--gpu-memory-utilization 0.85` 下**可用 Mamba cache blocks 仅 232** ⇒
  `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (232). Each decode
  sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.` ⇒
  `RuntimeError: Engine core initialization failed` ⇒ 引擎**从未就绪**（`docker events` =
  `exit=1`、**非 OOM**），评测容器空转 16 分钟无日志推进。
- 判读提示：**"vLLM 起不来 + 日志无 OOM + GPU 显存已释放"** ⇒ 先怀疑**引擎初始化参数**类
  （`max_num_seqs` / cache blocks / `max-model-len`），而不是显存不够。

### 7.1 §7 的谓词**必须按形态分支**（2026-10-13 补）

- 谓词不能只写 `config.json`：
  · **全参/HF** ⇒ `config.json` ∨ `model-*.safetensors`；
  · **PEFT adapter** ⇒ `adapter_config.json` ∨ `adapter_model.safetensors`（★ adapter 目录里
    **通常没有** `config.json` —— 只认 `config.json` 会把 adapter 档**误拒**，T008 实测）；
  · **native rank 分片** ⇒ `rank_*.pt` ∨ `manifest.json`。
- 落地件：`scripts/run_guards.py`（`form` / `fresh` / `mamba` 三条子命令）＋
  `tests/scripts/test_run_guards.py`（每族正例 + 负例，共 13 例）。

## 11. 仓内件必须**同批同步**部署树；调用方先探测存在性

- 规则：**仓内新增的护栏/脚本，落仓后必须同批 `scp` 同步到部署树，并在真机上跑一次冒烟**
  （如 `run_guards.py form` / `mamba` 各一条）。**调用方必须先探测依赖脚本存在性，缺失即
  fail-closed**（不要等它 `No such file` 才发现）。
- 为什么：228 实测（2026-10-13）——护栏脚本 `scripts/run_guards.py` **已提交进仓**，但**没有同步到
  部署树** ⇒ 评测脚本调用 `python3 $TREE/scripts/run_guards.py` 报 `No such file or directory` ⇒
  §7 自检 fail-closed 拒跑（零副作用，但白跑一轮）。
- 结构性配套：**启动流水线**（生成 → 护栏 → 起跑 → 监控 → 评测 → 落 json）不再每轮临时写，
  而是固化为 `scripts/launch_pipeline.py` ＋ `tests/scripts/test_launch_pipeline.py`。
  该链在 228 上连踩 5 次坑（§8 断言位置 → §7 谓词过窄 → 护栏件未同步 → 生成顺序错 →
  生成链变量为空），根因都是"启动脚本每轮现写"。

## 12. 起跑卡集必须**从 manifest 读 `cards`**，⛔ 禁止写死

- 规则：起跑前**从该档 manifest 读 `cards`**，据此在**实测空闲**卡里取同样多的卡；
  并**断言 `len(卡集) == cards`**（无重复、`cards > 0`）。不匹配即 **fail-closed**；
  实测空闲不足时**顺延**（⛔ 不硬上）。
- 为什么：228 实测（2026-10-13 T009）—— 起跑脚本**写死 `SEL=7`（1 卡）**，而该档
  `nproc_per_node=4` ⇒ torchrun 起 4 个 rank，`rank: 3 (local_rank: 3)` **`exitcode: 1`**，
  起跑即失败；而 launcher 的四条前置断言**全部通过**（树指纹/逐卡复测/镜像/dry-run），
  dry-run 行 `[dry-run] T009 gpus=7 nproc=4` 已经把矛盾写在脸上。
- 这是"没按本档自身实体取值"族错误的又一例（同族：基座取模板默认、`$MP` 当宿主机挂载源、
  `checkpoint_path` 写成 vLLM 容器路径）⇒ 落地件：`launch_pipeline.check_card_count()` /
  `plan_cards()`（含回归用例）。

### 11.1 §11 的"一条命令"（2026-10-13 补）

- 仓内件落仓后，**同轮跑一次**：`scripts/sync_deploy_tree.sh --tree <部署树>`。
  它**幂等**（可重复跑）、把 `scripts/*.py` 同步到部署树，并**逐件冒烟**：
  ① sha256 与仓内一致；② `python3 -c "import <mod>"` 可导入；③ 带 CLI 的件 `--help` rc=0。
  **任一失败即非零退出**（fail-closed）。
- 为什么需要它：228 上**两次**都是"起跑时才发现护栏不在部署树"（`ModuleNotFoundError`）——
  第一次是 `run_guards.py`，第二次是 `launch_pipeline.py` / `gen_tier_json.py`。

## 13. 每卡余量按 workload 定，⛔ 不用单一"头寸"量所有档

- 规则：择卡时**按该档 workload 定每卡余量**，并**默认拒绝"卡上有第三方常驻"**的卡
  （`memory.used > 0` ⇒ 不选，除非显式放行）；大模型**训练档**要"**整卡基本空闲**"。
  落地件：`launch_pipeline.plan_full_cards()`（`used_mib <= allow_used_mib` 且 `free_mib >= need_mib`）。
- 为什么：228 实测（2026-10-13 T009，27B 全参 4 卡）—— 旧判据"`free ≥ 头寸 48 GiB` ∧ `util == 0`"
  **放行了带第三方 13.01 GiB 常驻的 GPU2**（它当时 67.8 GB free ∧ util 0），而该档实际需要接近整卡 ⇒
  rank1（容器内 "GPU 1" = 宿主 GPU2）直接
  `torch.OutOfMemoryError: Tried to allocate 1.81 GiB. GPU 1 … of which 154.25 MiB is free.
   Process 3912182 has 13.01 GiB memory in use …`。
- 与 §6.2 同族：**服务档**要 `util × 总显存`，**训练档**要整卡 —— ⛔ 一把尺子量不了所有档。
- 判读提示：**"起跑成功但 rank N 秒退 + OOM"** ⇒ 先查 **卡集里是否混进了带常驻的卡**
  （容器内 GPU 序号 ≠ 宿主 GPU 序号，务必按 `CUDA_VISIBLE_DEVICES` 映射回宿主核对）。
