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
