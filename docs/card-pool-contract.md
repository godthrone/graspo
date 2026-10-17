# ★ 卡池契约（2026-10-17 重立；规则 1–2 承 2026-10-14 实例驱动版）

规则来自实打实的事故，不是推演。本件为**契约唯一真相源**。

## 规则 0：主席裁定（唯一标尺，逐字）

> 「**228 先到先得，随便用。**」
> 另：「**单个作业不限制卡数。**」

B10（2026-10-12「允许卡 = 本机全卡 0–7、先到先得；生产卡/预留卡语义已退役」）方向一致，
本次推进到终点：卡号判据**只剩存在性**，⛔ 不再有任何"某张存在的卡不能用"的静态策略。
⛔ 与本件冲突的旧表述一律作废（清单见文末「附」）。

## 规则 1：同一张卡不得同时承载训练与推理服务
**实例**：T022 的评测 vLLM 与 T030 的训练共用 GPU4 ⇒ 起服务失败，逐字
`ValueError: Free memory on device cuda:0 (30.82/79.25 GiB) on startup is less than
desired GPU memory utilization (0.85, 67.36 GiB)` ⇒ 评测白跑一轮。移至整卡空闲的 GPU7 后一次通过。
★ 本规则与规则 0 **不冲突**：它管的是**我方训练与推理之间**的互斥（同机同卡争显存），
⛔ 不是"某张卡被别人占着就静态禁用它"。

## 规则 2：评测择卡一律要求整卡空闲
训练可占整卡组；**推理与训练不得交叉**。评测择卡判据 = 整卡余量（⛔ 不是"卡上没显存就算空闲"
——评测 vLLM 加载期卡看似空闲，实为已预约）。
★ 该判据与规则 3 同源：**剩余 ≥ 本档头寸 ∧ util == 0**，⛔ 不是"占用 ≤64 MiB"这类绝对阈值。

## 规则 3：★ 不抢占他人（唯一保留的硬要求，只增不减）
静态上"卡号合法" ≠ 此刻能起跑。**起跑前必须逐卡实测**：

> **可用 ⟺ 该卡剩余显存 ≥ 本档头寸（默认 48 GiB；`GRASPO_GPU_POSITION_MIB` 可覆盖）
> ∧ `utilization.gpu == 0`**（默认连续 3 帧、间隔 12 s，**帧帧满足**）。

- 判据实现（单一真相源）：`graspo.flow.guard.idle`；入口 `scripts/gpu_idle_assert.py`
  （`graspo.cli.gpu_monitor:assert_target_gpus_idle` 用同一判据）。
- 被占则 **换卡或等**；⛔ **绝不 kill 他人进程、⛔ 绝不停他人容器、⛔ 绝不 `--gpu-reset`**。
- ⛔ 旧口径「占用 ≤64 MiB / util ≤5%」**已作废**：它把"卡上有他人常驻但不在算（util=0）"
  误判为不可用，与"先到先得"直接冲突。

## 规则 4：全卡可用，无预留、无禁用
- 可用卡号 = **本机全卡 0–7**；判据 = **存在性**（≥8 的卡号**不存在** ⇒ 拒绝）。
- ⛔ **不预留**、⛔ **不禁用**、⛔ **无"生产锁卡"**、⛔ **无"邻居卡"** —— 四类静态名单已全废。
- ★ 落地形态（单一真相源）：
  - 运行期：`src/graspo/core/gpu_guard.py::ALLOWED_INDICES = (0..7)`；
  - 清单/生成期：`tests/e2e/gpu_assignment.py::GPU_ALLOWED = (0..7)`（stdlib-only 派生）
    ＋ `generate_matrix_manifest_source.card_range_violation()`；
  - `GRASPO_ALLOWED_GPU_INDICES` 是**运维自限**开关（默认不设 = 全卡），⛔ 不是"禁用卡名单"。

## 规则 5：单个作业不限制卡数
- ⛔ 无"每次最多 4 卡"这类平台级上限：清单侧 `MAX_CARDS_PER_JOB` 与运行期
  `MAX_CARDS = 4` 的**拦截均已删除**（运行期 `max_cards=None` = 默认不判卡数；
  仅当调用方**显式**传入正整数时才按该值自限，且⛔ 不得被描述成平台口径）。
- 卡数的唯一约束来自**卡计划自身**：TP × PP × DP = 卡数（`parse_gpu_plan` 的一致性判据），
  以及清单侧"该档 `cards`"（覆盖只改"用哪几张"，⛔ 不改卡数）。

## 规则 6：每档清单里的 `gpus` = "建议首选卡位"
- 清单（`tests/e2e/matrix54_manifest_source.json` 等）与卡池映射（`GPU_POOL_BY_CARDS` /
  `TIER_GPU_OVERRIDE`）里的 `gpus` **降级为调度建议**：⛔ **不是**合法性判据。改跑别的卡，
  只要 ∈ 0–7 且卡数不变，**即合法**。
- 它仍有一个不可让渡的作用：**A4「同档两跑必须同卡位」的确定性锚** —— 同一档的两次运行
  必须引用**同一份清单**（`GRASPO_RUNNER_MANIFEST` 指同一路径）。

## 规则 7：§17/§17.3 为机器护栏（非人记）
- `run_guards cards-busy --ps-file … --inspect-file …` ⇒ `BUSY_CARDS=<csv>`
  （★ 须同读 `HostConfig.DeviceRequests` 与 `Config.Env` 的 `NVIDIA_VISIBLE_DEVICES`）
- 连排多档：**逐档重取** `cards-busy`（⛔ 不复用旧快照）
- 起跑前**逐卡实测空闲**（规则 3）；起跑后**回验**容器已登记
  （`launch_pipeline.require_card_registered`，未登记 ⇒ fail-closed）

## 规则 8：他人资源只读不占
第三方占用（任何卡上的他人进程/容器）**只观测、不排队、不干预**：`cards-busy`
不把它计入我方，守卫**不 kill 它、不停它**。
★ 本件为对外正式件 ⇒ 第三方**具名账号一律以角色化表述**（§15 去标识）；具名归属另存内部台账。

## 落地要求
1. 本文件为契约唯一真相源；
2. 机器护栏若缺"训练/推理互斥"或"不抢占他人"断言 ⇒ **补断言 ＋ 用例**；
3. 违反契约的档 ⇒ 起跑前 fail-closed，⛔ 不"先跑再看"；
4. ★ 任何人想"为了稳妥"再加一张静态禁用卡或再加一个卡数上限 ⇒ **必须先拿到主席新裁定**：
   规则 0 是唯一标尺，⛔ 不得由工程侧自行加严。

---

## 附：本件作废的旧表述（供审查者对照，⛔ 均已失效）

| 旧表述 | 出处（改前） | 处置 |
|---|---|---|
| `GPU_ALLOWED = (0,1,2,3,4,5)` | `tests/e2e/gpu_assignment.py:69` | 改为 `(0..7)`（本机全卡） |
| `GPU_FORBIDDEN_PRODUCTION = (6,7)` | `tests/e2e/gpu_assignment.py:71` | **删除**（无生产锁卡） |
| `MAX_CARDS_PER_JOB = 4` | `tests/e2e/gpu_assignment.py:73` | **删除**（不限制卡数） |
| `NEIGHBOR_CARDS = (2,)` / `LOCKED_CARDS = (2,6,7)` | `generate_matrix_manifest_source.py:464/470` | **删除**（无邻居卡） |
| `CHAIRMAN_BANNED_CARDS = (0,2)` | `generate_matrix_manifest_source.py` | **删除**（无禁用卡） |
| `legal_card_set() = (1,3,4,5)` | `generate_matrix_manifest_source.py` | **删除**（改判"卡号 ∈ 0–7"） |
| `locked_card_violation()` / `non_legal_card_violation()` | 同上 | 由 `card_range_violation()` 取代 |
| `MAX_CARDS = 4`（"用户拍板每次最多 4 卡"） | `src/graspo/core/gpu_guard.py:45` | 默认**不判卡数**（`max_cards=None`） |
| 空闲判据「占用 ≤64 MiB / util ≤5%」 | `cli/gpu_monitor.py:44,703`、`run_matrix54.sh:375` | 改为「剩余 ≥ 头寸 ∧ util == 0」 |
| 旧规则 4 的"邻居卡 GPU2"具名表述 | 本文件旧版第 20–22 行 | 保留"只读不占"原则（规则 8），删除具名卡 |
