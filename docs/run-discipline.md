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

- 判断某张卡能不能用（**起跑空闲判据 = B10 唯一口径**）：**剩余显存 ≥ 本档头寸**
  ∧ **util == 0**（连续 2–3 帧采样全满足）。落地件：`src/graspo/cli/gpu_monitor.py` 的
  B10 实测空闲断言。⛔ 已作废的旧判据"`memory.used ≤ 64 MiB` ∧ `util ≤ 5%`"不再使用；
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

- 规则（**2026-10-17 对齐冻结口径**）：择卡判据 = 该卡**剩余显存 ≥ 本档头寸** ∧ **util == 0**；
  "本档头寸"按该档 workload 定（训练档默认 48 GiB；服务档见 §6.2 = `util × 总显存`），
  ⛔ 不用单一"头寸"量所有档。落地件：`src/graspo/cli/gpu_monitor.py` 的 B10 实测空闲断言
  与 `launch_pipeline.plan_full_cards()`（`used_mib <= allow_used_mib` 且 `free_mib >= need_mib`）。
- ⛔ **【更正·原规则作废】** 原句"**默认拒绝"卡上有第三方常驻"的卡（`memory.used > 0` ⇒ 不选，
  除非显式放行）；大模型**训练档**要"**整卡基本空闲**""**已被取代**：现行判据只看
  **剩余 ≥ 本档头寸 ∧ util == 0**，卡上有他人常驻但**余量足够时可用**；
  同时**不抢占、不 kill 他人进程**（见 §3）。
- **历史依据**（保留作留痕，**不再作为择卡规则**）：上机实测（2026-10-13 T009，27B 全参 4 卡）——
  旧判据"`free ≥ 头寸 48 GiB` ∧ `util == 0`"
  **放行了带第三方 13.01 GiB 常驻的 GPU2**（它当时 67.8 GB free ∧ util 0），而该档实际需要接近整卡 ⇒
  rank1（容器内 "GPU 1" = 宿主 GPU2）直接
  `torch.OutOfMemoryError: Tried to allocate 1.81 GiB. GPU 1 … of which 154.25 MiB is free.
   Process 3912182 has 13.01 GiB memory in use …`。
- 与 §6.2 同族：**服务档**要 `util × 总显存`，**训练档**要 **`剩余 ≥ 本档头寸`（默认 48 GiB）**
  —— ⛔ 一把尺子量不了所有档，但也**不再要求"整卡基本空闲"**。
- 判读提示：**"起跑成功但 rank N 秒退 + OOM"** ⇒ 先查 **卡集里是否混进了带常驻的卡**
  （容器内 GPU 序号 ≠ 宿主 GPU 序号，务必按 `CUDA_VISIBLE_DEVICES` 映射回宿主核对）。

## 14. 起跑后再改脚本对"已在运行的实例"无效

- 规则：**脚本必须在起跑前一次性改对**。若起跑后才发现脚本有错，**必须"停旧实例（精确 PID）＋
  用新文件重起"**，⛔ **不是只改文件** —— 改完文件不等于在跑的实例会跟着变。
- 为什么：228 实测（2026-10-13 T007）—— 我用 `sed -i` 把评测脚本里 base 侧挂载源改成宿主路径，
  但**该实例已在 05:05 起跑**：`sed -i` 换掉了 inode，**运行中的 bash 仍持有旧内容** ⇒ 其 base 阶段
  （05:21:56）照样用容器内 `/models/...` 起 vLLM ⇒ `/model` 空目录 ⇒
  `ValidationError … ensure the presence of a 'params.json'` ⇒ 脚本在 `wait_ready` **空转 43 分钟**。
- 也就是**同一族第三次**（T024、T008、T007 都是 base 挂载源）；区别在于前两次是"写错"，
  这次是"改对了但对在跑实例无效"。
- 判读提示：**"改了脚本但现象没变"** ⇒ 先确认**在跑的是哪个 inode/实例**（`ps -o lstart= -p <pid>`、
  `ls -l /proc/<pid>/fd` 或直接看进程启动时刻 vs 文件 mtime）。

## 15. 护栏必须作用于"**被执行的那份产物**"

- 规则：护栏要么**直接嵌入产物**（生成脚本时就把已解析的宿主路径**字面量**写进去），
  要么在起跑前**对"即将执行的脚本实际文本"做断言**（fail-closed）。
  ⛔ **只校验辅助函数的返回值，不等于校验了产物**。
- 为什么：228 实测（2026-10-13 T009）—— 我**确实调用并核对了**
  `render_vllm_base_mount()` 的返回值（正确的宿主路径），**但生成脚本时仍写出 `-v $MPB:/model:ro`**
  （`MPB=/models/…` 容器内路径）⇒ `/model` 空目录 ⇒ `ValidationError … params.json`。
  这是 base 挂载源**同族第四次**：前三次是"写错"，第四次是"**验证了错的层**"。
- 落地件：`launch_pipeline.check_script_mount_sources()`（对脚本文本断言：`:/model` 的源必须是
  宿主模型根下的路径或脚本内解析到该处的变量；⛔ 拒 `/model`、`/models/…`、`/evalout/…`、
  `/workspace/…`、`/runs/…`）＋ `render_eval_script()` 把**运行期取值**（环境派生/入参的宿主路径，
  见 §18）解析后写进模板。
  ★ **2026-10-16 订正**：仓内已无宿主路径字面量 ⇒ 原先"直接把宿主路径**字面量**嵌入模板"的写法
  在去标识后**不再成立**（现为运行期取值 / 环境派生，缺省值由 `__file__` 与 `$HOME` 推导）。

## 16. 评测路径由**形态探测**决定，⛔ 不靠惯性（全参 ⇒ 零 merge）

- 规则：起评测前**先跑 `run_guards form`**，再按形态分支：
  · **`full`（全参/HF）** ⇒ ★ **零 merge**，vLLM **直指 checkpoint 目录**；
  · **`adapter` / `native_rank`** ⇒ ★ 先 `graspo eval export` **合并为 merged-hf** 再服务；
  · **unknown** ⇒ ★ **fail-closed**，⛔ 不得凭惯性选一条路。
  落地件：`launch_pipeline.decide_eval_mode()`。
- 为什么：228 实测（2026-10-13 T019）—— 我按"LoRA ⇒ 合并"的惯性写了 export 步骤，
  而该档实际是**全参/HF**（`form` 已打印 `[full]`）⇒ export 回执
  **`checkpoint is already a servable base model — nothing to merge`** ⇒ **合并目录根本不会生成**
  ⇒ `§8(post)` **正确 fail-closed**（拦下"没有产物的成功"）。护栏救了场，但**惯性本身该改**。

## 17. 评测预约卡与训练择卡互斥

- 规则：**择卡不得仅凭 `nvidia-smi` 快照判空闲** —— 必须**同时看 `docker ps` 里"我方容器占用的卡"**
  （`--gpus '"device=N"'` 的参数），把已预约的卡排除。★ **评测 vLLM 的加载期（约 4 min）里卡看起来是空的**
  ⇒ 必须视为**"已预约"**。★ 更稳的做法 = **评测与训练分卡池**。
- 为什么：228 实测（2026-10-13 T038 评测 / T022 训练）—— 08:11:48 起 T022 训练到 **GPU3**（当时
  T038 的 vLLM 尚未加载、显存 0 MiB ⇒ §13 整卡判据放行），08:12:30 起 T038 评测到**同一张 GPU3**；
  4 分钟后 vLLM 要显存却只剩 14.48 GiB ⇒
  `ValueError: Free memory on device cuda:0 (14.48/79.25 GiB) on startup is less than desired GPU
   memory utilization (0.85, 67.36 GiB)` ⇒ `RuntimeError: Engine core initialization failed`。
- 判读提示：**"卡上没显存"≠"卡没人用"** —— 容器刚起、模型还在加载时就是这个形态。

### 17.1 占卡台账：从"人记"改成"代码强制"（2026-10-13 补）

- 落地件：`run_guards.busy_cards_from_docker()`（解析 `docker inspect` 的 `HostConfig.DeviceRequests`
  ⇒ **我方容器占用的卡**，★ **同时覆盖 `running` 与 `created`/`restarting` 等"已预约未加载"态**）
  ＋ `include_reserved_cards()`（择卡**硬排除**，⛔ 不是建议）。
- 规则：★ **择卡前必须查询占卡台账（含 loading 态）**，并把结果作为硬排除集。
- 为什么（**历史事故记录**，2026-10-13，非现行卡池政策）：T002 评测起在同一张卡上，而 T044 正占该卡 ⇒ vLLM 只剩 `3.81/79.25 GiB` ⇒
  `ValueError: Free memory ... less than desired GPU memory utilization (0.85, 67.36 GiB)`。
  ★ 根因 = "哪些卡被我方占着"**靠人记/人看** ⇒ 当天已复发多次 ⇒ 故升级为代码强制。

### 17.2 择卡前跑 `cards-busy`（逐字命令与判读，2026-10-13 补）

```bash
docker ps --format '{{.Names}}' > /tmp/ps.txt
docker inspect $(docker ps -q) > /tmp/ins.txt 2>/dev/null
python3 $TREE/scripts/run_guards.py cards-busy --ps-file /tmp/ps.txt --inspect-file /tmp/ins.txt
# ⇒ BUSY_CARDS=0,1,3,4,5,6,7
```
- 判读：★ **禁用集非空 ⇒ 这些卡一律排除**（含 `created`/`restarting` 的"已预约未加载"态）；
  择卡把该集合作为 `plan_full_cards(..., reserved=…)` 的**硬排除**参数。
- ★ 采集口径：★ 卡可能来自 **`HostConfig.DeviceRequests`** 或 **`Config.Env` 的 `NVIDIA_VISIBLE_DEVICES`**
  （228 实测为后者）⇒ 台账**两者都读**。

### 17.3 禁用集不得复用旧快照（连排必逐档重取 + 起跑后回验）

- ⓐ **连排多档时，每档起跑前必须重新采样 `cards-busy`** —— ⛔ 不复用上一次的快照。
- ⓑ **每档起跑后必须回验"该档容器已登记 / 该卡已入禁用集"** —— **未登记 ⇒ 不得视为已占用**。
- ⓒ **`plan_full_cards` / 择卡只能吃"刚取到的"禁用集**。
- 机器护栏：`launch_pipeline.require_card_registered(card, busy_after_start)`（未登记 ⇒ fail-closed）。
- 实证三条（2026-10-13，历史实证）：① **T022 撞卡** —— 连排 T032 评测 + T029 后立刻起 T022，快照缺 GPU6 ⇒
  `plan_full_cards` 返回 `['6']` ⇒ 与评测同卡；② **T034 起后 8 s 仍未登记**；③ **T043 评测起后亦未登记**。
  ⇒ 前一次同类被门禁拦下、后两次放行 ⇒ **光靠台账不够，必须"起跑后回验"**。

## 18. 评测生成脚本的运行期取值（最短 runbook，2026-10-16 补）

- 生成脚本里的宿主路径与远端执行目标一律**运行期取值**（§15.1：源码里不留内网字面量）。
  本批引入的 6 个变量与必需等级：
  - `GRASPO_EVAL_SSH_USER`、`GRASPO_EVAL_SSH_HOST` —— ★ **必需**（两者任一缺失 ⇒
    `scripts/gen_eval.py` **fail-closed 退出**；⛔ 不回落成假地址把命令静默发到错机器）。
  - `GRASPO_EVAL_TREE`、`GRASPO_EVAL_DIR`、`GRASPO_MODEL_HOST_ROOT`、`GRASPO_ELAM_V5_DIR`
    —— 可选，有按脚本自身位置 / `$HOME` 推导的默认值。
- ★ **这 4 个默认值只在"在部署机上、以部署账号、从部署树运行"时才成立，且该前提未在部署机实测。**
  在别的机器上直接跑生成脚本**不报错**，而是**静默产出指向本机路径的脚本**（`$T`/`$EV`/`-v …:/models`/
  `-v …:/work/elam-v5` 全落在本机）⇒ 起跑前必须逐条核对生成脚本里的宿主路径。
- `scripts/launch_pipeline.py` 侧同族变量为 `GRASPO_MODEL_HOST_ROOT` / `GRASPO_ELAM_V5_DIR`。

## 19. 判据③④ 的判定与证据（谁产出、放哪、缺了会怎样）

**依据（唯一真相源，⛔ 本文件不另立判据）**：判据③④ 原文见
`docs/GRASPO训练框架功能测试报告.html` **第 2 章**；GRASPO 的"逐条判断"清单 = 
`docs/graspo-algorithm.html` **第 9 节 不变式清单**（共 8 条）。判据② 的评测读数仍看第 2 章与
`docs/capability-matrix.data.json` 的 `meta.criteria_caliber`。

### 19.0 判据①「跑完」= 跑满 Epoch（2026-10-23 对齐第 2 章原文）

**判据① = 跑满 Epoch；步数仅作台账。** 通过条件 =
`run_result.exit_code == 0` ∧ `run_result.epochs_completed ≥ run_result.epochs_planned`
（`epochs_planned` = 清单该档 `max_epochs`）∧ `run_result.final_ckpt_loadable == true`
∧ `run_result.nan_or_inf_detected == false`。`steps_actual` 仍必填并回填进「实验数据·实跑步数」列，
但**不参与**通过判定。epoch 读数来源 = `<run-dir>` 下的 `trainer_state.json`（native / ms-swift 同形）
或 `events.jsonl` 的 `epoch_summary`（native）；读不到 ⇒ 结果 json 写 `null` ⇒ 回填器判 `unknown`
（fail-closed，⛔ 不许回落成"实跑步数 == 计划步数"比较）。

**通道（只有这一条，⛔ 没有第二个入口）**：

```
评测跑完（scripts/gen_eval.py）
  → scripts/record_checks.py --tier M0nn --run-dir <run-dir> --eval-dir <EVAL_DIR>
        ⇒ 写 <run-dir>/readings.json 的 checks{algo_checks_passed, inference_parseable}
          （附 checks_evidence：判定条件、实测数、证据原件路径）
  → scripts/make_result_json.py …（只消费上面那两个布尔，⛔ 不自己判、⛔ 不补 true）
        ⇒ 结果 json 的 checks{} ⇒ 回填器 translate 成判据③④ 的 pass/fail/unknown
```

- `--tier-json`（`scripts/gen_tier_json.py` 的产物）**只提供判据② 的 after/base 读数**，
  ⛔ 不提供③④；`readings.json` 里没有 `checks` 就是"没有证据"。
- 回填器 `tools/matrix/backfill_matrix.py` 对缺键一律判 `unknown` ⇒ **fail-closed**：
  本批不写任何文件、该档保持「待测试」。⛔ **不许**为让流程变绿给回填器开"缺③④也算通过"的口子。

### 19.1 判据④「能推理」= 自动判定（数据源：评测报告）

数据源：`<EVAL_DIR>/.local/reports/after/eval_report.json`（`graspo-eval-report-1` 契约，
见 `src/graspo/eval/schema.py`）。判据④ 的六个条件**同时**成立才算通过：

| # | 条件 | 字段（证据） |
|---|---|---|
| 1 | 报告可读 ∧ 有逐样本明细 | `samples[]` 非空（报告即"最小充分信息"，缺它无从判可解析） |
| 2 | 评测温度为 0 | `decoding.temperature == 0.0`（算法说明 §9 第 4 条：评测温度恒 0、入口不提供覆盖参数） |
| 3 | 在训练数据集上完整推理一遍 | `summary.sample_count_total ==` 清单该档 `data.subset_size`（本矩阵 100）∧ `dataset.sample_count_requested` 同值 ∧ `total == len(samples)` |
| 4 | 无请求级失败 ∧ 输出可解析 | `summary.sample_count_error == 0` ∧ 逐样本「无 `error` ∧ 无 `parse_error` ∧ `pred_name` 非空」的条数 == 期望条数 |
| 5 | LoRA 档必须是**合并后**的产物 | `model.export_format ∈ {peft-adapter, graspo-native, native-rank}` ∧ `model.path ≠ model.checkpoint_path`，或目录名命中合并落点 `merged-hf-*`；全参档不适用（记通过并注明）。缺证据 ⇒ **不写该键** |
| 6 | 必须**用 vLLM 启动** | 报告 `engine`（含 vllm）或 `model.served_name` 非空（= vLLM `--served-model-name`）。缺两者 ⇒ **不写该键**；`engine` 明确非 vLLM ⇒ 写 `false` |

- `parse_error` = 有输出但 `arguments` 串解析不出（`src/graspo/eval/criteria.py::extract_prediction`）；
  `pred_name` 空 = 没产出结构化工具调用 —— 两者都算"不可解析"。
- 条件 1–2 缺 ⇒ **不写该键**（没有证据，不是"不通过"）；条件齐但 3–4 不成立 ⇒ 写 `false`
  （**有据的不通过**，备注列会写可解析条数）。
- 本判定与 `gen_tier_json.py` 的效果口径**互不替代**：④ 只判"能否产出可解析结果"，
  "效果是否变好"是判据②（`eval.after_acc/base_acc`）。

### 19.2 判据③「算法正确性检查通过」= 分级（两套签署件，⛔ 不混）

判据③ 原文（第 2 章）＝「SFT / CPT / OPD 按各自公开的标准实现逻辑判断；GRASPO 按《算法说明》
逐条判断。**SFT / CPT / OPD 另按公开标准核对损失函数与标签掩码的处理规则（OPD 另含教师信号
的处理规则）**」。后半句是一个**动作**，所以三类公开算法同样要**签署件**——只是清单不同：

| 算法 | 签署件（唯一位置） | 清单 | 与档号的绑定 |
|---|---|---|---|
| SFT / CPT / OPD | `samples/configs/matrix68-checks/public-algos.json` | 损失规则 + 标签掩码规则（OPD 另加教师信号规则） | **不绑定**（一套规则覆盖该算法全部档） |
| GRASPO | `<algo-checks-dir>/<tier>.json` | 《算法说明》第 9 节 8 条不变式 | **绑定**（逐档签） |

#### 19.2.1 SFT / CPT / OPD：公开算法签署件（`public-algos.json`）

```json
{
  "$schema_version": "graspo-public-algos-signature/1",
  "signer": "executor-r4", "checked_at": "2026-10-22T12:15:00+00:00",
  "methods": [
    {
      "method": "sft",                      // sft | cpt | opd（小写，逐类一条）
      "verdict": "pass",                    // ⛔ 只有真核过才写 pass；fail ⇒ 该算法全部档保持「待测试」
      "checker": "executor-r4",             // 核对人（必需）
      "checked_at": "2026-10-22T12:15:00+00:00",   // 核对时间（必需）
      "loss_rule": "…实现位置 src/…/training_sft.py:170-180",          // 必需，须带 file:line
      "label_mask_rule": "…实现位置 src/graspo/ripple/data.py:397-410", // 必需，须带 file:line
      "teacher_signal_rule": "…（仅 opd 必需，须带 file:line）",
      "evidence": [{"item": "…", "path": "…:行", "conclusion": "…"}]   // 必需，逐条核对项
    }
  ]
}
```

校验规则（`scripts/record_checks.py::judge_public_algo`，全 fail-closed）：缺件 / 无该 `method`
条目 / `verdict != "pass"` / `checker` 或 `checked_at` 为空 / 缺 `loss_rule` 或 `label_mask_rule`
（OPD 另缺 `teacher_signal_rule`）/ 规则里**没有 `file:line` 代码坐标** / `evidence` 为空
⇒ **不写 `algo_checks_passed` 键** ⇒ 回填器判 `unknown` ⇒ 该档保持「待测试」。
`basis_id` 落 `public_algorithm_signed_public_algos_json`（人读件的备注列据此写签署人与时间）。

- **为什么放仓内而不是跑次目录**：损失/掩码规则是**算法级**的，与档号无关；放进 `<run-dir>`
  会在 68 档里产生 68 份必须同步的副本（单一真相源 §1.4）。`--public-algos` / `$GRASPO_PUBLIC_ALGOS`
  只是**指向**另一份（演练/回归用），不是第二份口径。
- ⛔ **不得手填 `pass`**：`verdict` 只能来自真读过被引代码后的结论；相关代码一改，必须重核并
  更新 `checked_at`/`checker`（本件引用的行号对的是当次 HEAD）。
- ⛔ 缺件即不写键：**没有签署件 = 没核过 = 不许说通过**（不是"缺数据"，是"没人做"）。

#### 19.2.2 GRASPO：逐档签署件（第 9 节 8 条不变式）

**GRASPO（⛔ 不许自动置位，必须逐档签署）**：证据文件 `<algo-checks-dir>/<tier>.json`
（runner 缺省 `<run-root>/algo-checks/`；也可用 `GRASPO_ALGO_CHECKS_DIR` 或
`--algo-checks-dir` 指向别处，例如 `.local/hb-workspace/<会话时间戳>/algo-checks/`）：

```json
{
  "tier": "M018",
  "algorithm": "GRASPO",
  "checker": "<检查人（姓名或角色）>",
  "checked_at": "2026-10-22T20:00:00+00:00",
  "items": [
    {"id": "group_integrity",           "pass": true, "evidence": "<怎么核的：命令/读数/结论>"},
    {"id": "reward_sample_alignment",   "pass": true, "evidence": "…"},
    {"id": "algo_logic_unchanged",      "pass": true, "evidence": "…"},
    {"id": "temperature_not_overridden","pass": true, "evidence": "…"},
    {"id": "label_length_invariant",    "pass": true, "evidence": "…"},
    {"id": "sampling_train_same_policy","pass": true, "evidence": "…"},
    {"id": "ratio_interval_constant",   "pass": true, "evidence": "…"},
    {"id": "value_error_not_truncated", "pass": true, "evidence": "…"}
  ]
}
```

`items[].id` 的 8 个取值**逐条对应**《算法说明》第 9 节（顺序同上，⛔ 不许自造 id、不许少条）：
组完整性 / 奖励与样本一一对齐 / 算法逻辑严禁改动 / 温度互不覆盖 / 标注长度不变式 /
采样与训练同源 / 裁剪区间恒定 / 值错误不截断。

校验规则（`scripts/record_checks.py::judge_graspo_algo`，全 fail-closed）：`algorithm=="GRASPO"`
∧ `tier` 相符 ∧ `checker`/`checked_at` 非空 ∧ 8 个 id **恰好全覆盖**（无缺、无重、无未知）∧
每条 `pass` 是布尔且 `evidence` 非空。不合契约 ⇒ **不写该键**；8 条里有 `pass=false` ⇒ 写 `false`。
人工签署只写"逐条结论 + 依据"，⛔ **不得**改成"看着像过了就写 true"。

### 19.3 缺证据的后果（⛔ 不许绕）

| 情形 | `checks` | 回填器 | 该档状态 |
|---|---|---|---|
| ③④ 都有据且真 | 两键都在 | 判 pass | 可「通过」（还要 ①② 同时过）|
| ③ 或 ④ 缺证据 | **缺对应键** | 判 `unknown` ⇒ fail-closed，本批不写任何文件 | 保持「待测试」 |
| ③ 或 ④ 有据但假 | 写 `false` | 判 fail | 「不通过」并落盘（备注列点名未过判据）|

`make_result_json.py` 在缺 ③④ 时判 **rc=4**（结构合法但证据不全）：结果 json 照落盘供审计、
也进 `collect/`，但**不会**被判成通过。补齐证据后复跑 `record_checks.py` → `make_result_json.py`
即可（幂等；`record_checks.py` 每次先清旧结论再按本次证据重写，⛔ 不留陈旧通过）。

#### 19.3.1 档号护栏：`--tier` 必须与 `readings.json.tier` 一致（⛔ 不自动纠正）

`scripts/record_checks.py` 与 `scripts/make_result_json.py` **各自**先核
`readings.json["tier"]`：**缺该字段**或 **`!= --tier`** ⇒ **rc=2 并打印两个档号**，拒跑。

为什么必须有（实测事故，审查第 3 轮 🔴-2）：68 档是**人工分档补跑**的场景，`--tier M001` 与
`--run-dir …/run/M018` 这类手滑极易发生。此前两个脚本都不核，于是
`--tier M001 --run-dir <M018 的跑次>` 会把 SFT 的 `algo_checks_passed=true` 写进 GRASPO 档
（`readings.json` 自带的 `tier:"M018"` 被无视）⇒ 适配器按 M018 出件 ⇒ 回填 **rc=0、状态「通过」**，
备注列还印出「按《GRASPO 算法说明》第 9 节逐条算法检查，全部不变式通过」——**一份签署件都没有**。

⛔ 护栏只**拒绝**，不做任何"纠正"：不按目录名改写 `--tier`、不改写 `readings.json.tier`、
不"取两者的公共前缀"。配错就重跑，读数不动。

#### 19.3.2 备注列文案只能由证据生成（⛔ 不得只凭布尔）

人读件的备注列进发布件，因此判据③ 的措辞**只能从 `checks_evidence.algo.detail` 里的
`basis_id` + `checker` + `checked_at` 现读**：

| 证据声明 | 备注列措辞 |
|---|---|
| `graspo_algorithm_section9_signed` ∧ 有签署人/时间 | 「按《GRASPO 算法说明》第 9 节逐条算法检查，全部不变式通过（签署人 X，时间）」 |
| `public_algorithm_signed_public_algos_json` ∧ 有签署人/时间 | 「按 public-algos 签署件核对损失函数与标签掩码处理规则（OPD 另含教师信号），结论通过（签署人 X，时间）」 |
| 有布尔但**无**上述签署留痕 | 「算法正确性检查证据缺签署人/时间，本节不作结论陈述」 |

⛔ 任何情形都不得在**缺签署**时输出"已通过检查/逐条通过"一类字样（这就是第 3 轮的
"发布件不实陈述"）。

### 19.4 ⛔ 红线

- ⛔ **不得为凑通过手填 `true`**：`readings.json.checks` 只能来自 `record_checks.py` 的实测/签署证据；
  发现手填即视为伪造读数，同批记录并回退。
- ⛔ 不得放宽 19.1 的条件、不得减少 19.2 的 8 条、不得把"缺证据"改成"通过"。
- ⛔ **不得手填公开算法签署件的 `verdict: pass`**：缺件/`fail`/缺规则/规则无 `file:line`
  ⇒ 一律不写 `algo_checks_passed` 键（§19.2.1）。"公开算法不作逐条清单要求"**不是**置位理由。
- ⛔ 不得绕过档号护栏（§19.3.1）：`--tier` 与 `readings.json.tier` 不一致就修命令，不许改读数。
- ⛔ 判据③④ 的结论**只落 `readings.json`**（及其派生结果 json）；人读件只由回填链写入，
  ⛔ 不得绕过回填直接改数据件/人读件（见本文件 §19 与 `tools/matrix/backfill_dated.sh` 的落点闸门）。

### 19.5 CPT 的判据② = 「训练 loss 正常下降」∧「after ≥ base」

loss 读数由 `scripts/make_result_json.py` 从 `trainer_state.json` 的 `log_history[*].loss`
（ms-swift/HF）或 `events.jsonl` 的 `train_step.optimize.loss_mean`（native）抽成
`eval.loss_curve{n, first, last, min, series, …}`；回填器用
`src/graspo/core/result_judge.judge_cpt_loss` 的**同一口径**判定（窗口 =
`min(max(3, round(n × 0.10)), n)`，判定式 = **终窗均值 < 起窗均值**，严格小于）。
缺 `eval.loss_curve.series` ⇒ `unknown`（fail-closed）；loss 未下降 ⇒ `fail`。
⛔ 不得另立阈值、不得把"缺 loss 读数"当成通过。

## 20. OPD 教师侧评测（判据② 的教师读数：谁跑、跑什么、放哪、缺了会怎样）

**依据（唯一真相源，⛔ 本文件不另立判据与阈值）**：
- 判据② 原文（`docs/GRASPO训练框架功能测试报告.html` **第 2 章**）：
  「OPD（蒸馏）：教师在训练数据集上的完全正确率，与学生未训练模型相比 ≥ +10.0 个百分点
  （教师 − 学生未训练模型 ≥ +10.0 pp）；学生训练后，训练数据集完全正确率提升至少 5.0 个百分点」。
- 教师身份（同一文档 **第 3 章「模型说明 · OPD 教师」**）：教师 = **27B 在 SFT / LoRA / dp=1
  这一档训练完成后的产物**（⛔ 不是基座权重本身）。数据件
  `meta.criteria_caliber.threshold.by_method.opd` 的 `teacher` / `epoch_note` 是同口径的第二处逐字。
- 阈值数值（教师线 +10.0 pp、学生线 +5.0 pp）**只在数据件里**：回填器
  `tools/matrix/backfill_matrix.py::teacher_line_pp()` / `judge_effect()` 现读现算，⛔ 本文件与代码都不抄数值。

**口径三条（与学生侧**完全相同**，⛔ 不另立第二份）**：

| # | 口径 | 落点（唯一来源） |
|---|---|---|
| 1 | 数据集 = **该 OPD 档自身训练用的同一份 100 条**（`dataset_split: train`） | `scripts/gen_eval.py::compute_caliber()`（与学生侧 after/base 同一个函数、同一份清单读数） |
| 2 | 解码温度 **恒 0**（贪心），入口不提供覆盖参数 | `src/graspo/eval/schema.py::EvalDecoding`（非 0 直接拒绝）+ vLLM 调用侧 |
| 3 | 判分函数 = 逐字段**完全一致**比对（与学生同一条码） | `src/graspo/eval/criteria.py`（与算法、rank_method、`judge_method` 无关） |

⇒ 教师读数与 `eval.after_acc` / `eval.base_acc` **必然同口径**；`record_checks.py` 还会用
`dataset.sha256 ==` 清单该档 `data.subset_sha256`（并等于学生 after 报告的数据 sha）当场证一遍。

**谁跑（顺序约束 ⛔ 不可调）**：

```
① 教师档训练完成（27B / SFT / LoRA / dp=1；本矩阵按清单现算 = M041）
     ⇐ 教师档必须先于全部 OPD 档完成：教师产物是 OPD 档判据② 的前置证据
② 导出可推理形态：LoRA ⇒ 必须 merge 成 merged-hf 才能被 vLLM 加载
     （由生成的教师侧脚本内 `graspo eval export` 完成，与学生 LoRA 档**同一条**路径）
③ 教师侧评测（一次即可，8 个 OPD 档共用；它们共用同一份 100 条，sha256 相同）
     python3 scripts/gen_eval.py --tier M068 --role teacher --gpu <评测卡> --port <端口> [--blocks <实测>]
        ⇒ 产出 <tmp-dir>/run-m068-teacher-eval.sh + eval-m068-teacher.yaml（`role: other`）
        ⇒ 跑完落 <EVAL_DIR>/.local/reports/teacher/eval_report.json
④ 折算成读数：
     python3 scripts/record_checks.py --tier M0nn --run-dir <run-dir> --eval-dir <EVAL_DIR> \
         [--teacher-report <path>] [--algo-checks-dir <DIR>]
        ⇒ 写 <run-dir>/readings.json 的 `teacher_after_acc`（比例）+ `checks_evidence.teacher_after_acc`
⑤ 透传进结果 json（字段名以回填器实际要求为准 = `eval.teacher_after_acc`）：
     python3 scripts/make_result_json.py … --eval-dir <EVAL_DIR> …
⑥ 回填：判据② 由 `judge_effect()` 现算 ——
     学生线：`(after_acc − base_acc) × 100 ≥ 5.0 pp`；
     OPD 另加教师线：`(teacher_after_acc − base_acc) × 100 ≥ 10.0 pp`（⛔ 是 **base_acc**，不是 after）。
```

- `--role teacher` 只接受 **OPD 档**的 `--tier`（判据② 只有 OPD 一条带教师要求）；教师档本身按
  文档口径从清单**现算**（SFT ∧ 27B ∧ LoRA ∧ TP=PP=DP=1），命中数 ≠ 1 即 fail-closed，⛔ 不写死档号。
  ⚠ 文档散文「dp=1」单独不唯一（本矩阵有 6 档 dp=1）⇒ 判别键含 `nproc_per_node=1`（单卡）。
- 教师侧评测**只跑一侧**（没有 base/after/delta 三件套）：教师的 Δ 由回填器用**学生**的 base 现算。
- 教师侧 `max-num-seqs` 按**教师档**形态算（27B ⇒ 需 §10 限流），与学生档是不是 9B 无关。

**产物与字段（唯一通道，⛔ 没有第二个入口）**：

| 环节 | 产物 | 字段 |
|---|---|---|
| 教师侧评测 | `<EVAL_DIR>/.local/reports/teacher/eval_report.json` | `model.role == "other"`、`decoding.temperature == 0.0`、`summary.sample_count_total == 100`、`summary.sample_count_error == 0`、`dataset.sha256`、`summary.accuracy`、`model.checkpoint_path`、`model.base_model_path` |
| 折算 | `<run-dir>/readings.json` | 顶层 `teacher_after_acc`（0–1 比例）+ `checks_evidence.teacher_after_acc`（条件、实测数、报告路径） |
| 适配 | `<run-dir>/<档号>.json` → `collect/<档号>.json` | `eval.teacher_after_acc`（与 `after_acc`/`base_acc`/`delta_pp` 同级） |
| 回填 | 数据件该档「是否通过」+ 备注列 | 回填器现算；备注列写「教师在训练数据集上的完全正确率 X%；教师 − 学生未训练模型 = +Y pp」 |

**缺证据的后果（⛔ 不许绕）**：

| 情形 | readings.json | 适配器 | 回填器 | 该档状态 |
|---|---|---|---|---|
| 教师报告齐且合规 | 写 `teacher_after_acc` | `eval.teacher_after_acc` 落盘 | 判据② 现算（可能 pass / fail） | 可「通过」（还要 ①③④ 过）或「不通过」 |
| 缺教师报告 / 不合口径（温度≠0、条数≠100、数据 sha 不同、role≠other、准确率读不到） | **不写该键** | 不写 `teacher_after_acc` + `gaps` 点名 | `judge_effect` 判 `unknown` ⇒ fail-closed，本批不写任何文件 | 保持「待测试」 |
| 教师读数在但教师线没过 | `teacher_after_acc` 已写 | 已透传 | 判据② `fail` | 「不通过」（⚠ 这是**有据的不通过**，与学生线是否过无关） |

- `record_checks.py` 对 OPD 档把教师读数计入**必需证据**：缺 ⇒ rc=4（非 OPD 档不受影响，仍 rc=0）。
- ★ **教师身份是真的核过的**（不只是 `role=other`）：`judge_teacher` 另外核
  ① `model.checkpoint_path` 必须落在**清单现算的教师档**（本矩阵 `M041`）产物目录下
  （按路径分段比对，不是子串匹配）；② `model.base_model_path` 必须等于该档的 `model_path`
  （本矩阵 `/models/Qwen3.8-27B`）。两条都从清单现读，⛔ 不写死第二份路径；不符 ⇒ **不写键**。
  ⇒ `gen_eval --role teacher` 的现算 + 本核对一起构成"评的确实是那个教师"的证据链，
  报告的 `role` 字段本身**不**证明模型身份。
- `make_result_json.py` 对 OPD 档缺 `teacher_after_acc` 时进 `gaps` ⇒ rc=4（结构合法、证据不全）。
- ⛔ **不得**给回填器或适配器开"缺教师读数也算通过"的口子；⛔ 不得填默认值
  （0 / base 读数 / 学生 after 读数冒充教师读数）；⛔ 不得改 `tools/matrix/backfill_matrix.py`
  的判定或数据件口径来让流程变绿。

**8 个 OPD 档共用一份教师报告**：报告落点默认挂在 `<EVAL_DIR>` 下，而 runner/gen_eval 的
`EVAL_DIR` 是**逐档**的（`eval-<档号>-recon`）⇒ 两种做法二选一，都合法：
① 让 8 个 OPD 档共用同一个 `EVAL_DIR`；② 教师报告放一处，各档折算时用
`--teacher-report <同一份路径>`（或 `$GRASPO_TEACHER_REPORT`）指过去。⛔ 不要为每档重跑一遍教师评测
（教师模型与数据都没变，重跑只会引入新的读数分叉；报告本身已带 `dataset.sha256` 与温度可核）。

**复跑与防污染**：教师报告是**幂等**输入（同一份报告可被 8 个 OPD 档反复折算）；
教师档重训 ⇒ 教师报告必须重跑（`run_guards fresh` 已内建在教师侧脚本的 export 前后），
⛔ 不得拿旧教师报告配新教师产物。

## 21. OPD 档教师走**外部 vLLM 服务**（教师占卡**不计入**矩阵卡数，2026-10-24 主席裁定）

**依据（主席原话，逐字）**：

> 「OPD×27B 三档（M066–M068），教师用 vLLM 部署，单独弄别的卡，不算在测试表的卡数里面，
> 测试表那个卡数是针对被训练的模型的。」

- **受影响档**：数据件 `rows[].tier ∈ {M066, M067, M068}`（27B × OPD × LoRA）。接入方式是**行级**字段
  `distill_teacher_server`（= 外部教师服务端点），重生成的 yaml 里 `distill.teacher_model_server` 生效、
  `teacher_model_path` **不写**（= 未提供）、`offload_teacher_model` **不再出现**（本地教师不加载 ⇒ 换出无意义；
  二者与外部服务**互斥**，见 `src/graspo/core/schema.py::DistillConfig`）。
  ⛔ 数据件是唯一可编辑源头，yaml 一律由 `tools/matrix/gen_configs.py` 重生成（见 yaml 文件头）。
- **9B×OPD 五档（M036–M040）不变**：仍走本地冻结教师（`teacher_model_path` + `offload_teacher_model: true`）。
  教师**来源**是逐档的接入方式，⛔ 不是全矩阵一刀切。
- ★ **卡数口径**：矩阵第 4 章 `cards` 只算**被训练的模型**（学生侧）；教师服务独占的卡**不计入** `cards`，
  也不进 `check_68_readiness` / 运行清单的卡数断言。⇒ 教师卡是**额外**的一组卡，起服与训练**分别取卡**。
- **起服前必须取卡**（顺序固定，⛔ 不可跳）：

  ```bash
  docker ps --format '{{.Names}}' > /tmp/ps.txt
  docker inspect $(docker ps -q) > /tmp/ins.txt 2>/dev/null
  python3 scripts/run_guards.py cards-busy --ps-file /tmp/ps.txt --inspect-file /tmp/ins.txt   # §17.2
  python3 scripts/gpu_idle_assert.py --visible <教师卡列表> --position-mib <头寸>              # 逐卡实测空闲
  ```

  `cards-busy` 命中 / `gpu_idle_assert` 判不空闲 ⇒ **换卡或等**；⛔ 不 kill 他人进程、⛔ 不停他人容器（§3）。
- **起服（默认 dry-run；`--apply` 才真起）**：

  ```bash
  bash tools/matrix/serve_teacher.sh --gpus 3 --model <教师 merged-hf 目录> --port 18901          # 只打印命令
  bash tools/matrix/serve_teacher.sh --gpus 3 --model <同上>            --port 18901 --apply        # 真起容器
  ```

  脚本**不自动选卡**、**不判卡是否空闲**（那是设施层判据的单一真相源）、不写死宿主内网地址；
  无 `--apply` 时**不做任何外部调用**（不起容器、不探网、不查卡）。
- **就绪判据**（⛔ 「容器起来了」不算就绪）：`curl -s http://127.0.0.1:<port>/v1/models` 返回该 `served-model-name`。
- ⚠ **可达性**：yaml 里的端点值是**默认占位**（`http://127.0.0.1:18901`）。训练跑在容器内 ⇒ 容器内的
  `127.0.0.1` 是**它自己**：起跑时按**实际可达地址**（宿主网关 / 服务名）注入/对齐，⛔ 不要假定占位值直接可用。
- **失败处置**：服务不可达 ⇒ **该档不得起跑**（fail-closed；⛔ 不许让训练进程"等教师"，⛔ 不许静默回退成本地教师）。
  容器异常 ⇒ `docker ps --filter publish=<port>` 找到本脚本起的容器，用 `docker logs` / `docker stop` 精确处理
  （§9：⛔ 禁 `pkill -f`）。
- **`gkd_logits_topk` 是外部服务的硬前置**（⛔ 不是可选优化）：ms-swift 走 `--teacher_model_server` 时要求
  `--gkd_logits_topk` 非空（服务端只回 top-k logprobs，`swift/arguments/rlhf_args.py:762-765`）；
  配置层已 fail-closed（`src/graspo/core/schema.py::DistillConfig._validate_external_teacher_requires_topk`）。
  ⇒ 三档 yaml 的 `distill.gkd_logits_topk` = **64**（取值来自数据件行字段 `distill_gkd_logits_topk`，
  ⛔ 生成器不写死超参），起服脚本相应传 `--max-logprobs 65`（= topk + 1，服务侧低于它请求会被拒）。
  ⚠ 该 top-k 取值是**本包取的默认**（上游契约要求非空，仓内既有示例/测试同为 64）；若要改，改数据件行字段后
  重生成 + 重起服务，⛔ 不要只改 yaml。
