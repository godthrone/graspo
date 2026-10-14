# ★ B10 卡池契约（2026-10-14 立，实例驱动）

规则来自今日实打实的事故，不是推演。

## 规则 1：同一张卡不得同时承载训练与推理服务
**实例**：T022 的评测 vLLM 与 T030 的训练共用 GPU4 ⇒ 起服务失败，逐字
`ValueError: Free memory on device cuda:0 (30.82/79.25 GiB) on startup is less than
desired GPU memory utilization (0.85, 67.36 GiB)` ⇒ 评测白跑一轮。移至整卡空闲的 GPU7 后一次通过。

## 规则 2：评测择卡一律要求整卡空闲
训练可占整卡组；**推理与训练不得交叉**。评测择卡判据 = 整卡余量（非"卡上没显存就算空闲"——
评测 vLLM 加载期卡看似空闲，实为已预约）。

## 规则 3：§17/§17.3 为机器护栏（非人记）
- `run_guards cards-busy --ps-file … --inspect-file …` ⇒ `BUSY_CARDS=<csv>`
  （★ 须同读 `HostConfig.DeviceRequests` 与 `Config.Env` 的 `NVIDIA_VISIBLE_DEVICES`）
- 连排多档：**逐档重取** `cards-busy`（⛔ 不复用旧快照）
- 起跑后**回验**容器已登记（`launch_pipeline.require_card_registered`，未登记 ⇒ fail-closed）

## 规则 4：邻居资源只读不占
第三方占用（如 GPU2 的 `remote+`）**只观测、不排队、不干预**；`cards-busy` 不把它计入我方。

## 落地要求
1. 本文件为契约唯一真相源；
2. 机器护栏若缺"训练/推理互斥"断言 ⇒ **补断言 ＋ 用例**；
3. 违反契约的档 ⇒ 起跑前 fail-closed，⛔ 不"先跑再看"。
