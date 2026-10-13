# `tools/matrix/` — 功能矩阵文档工具链

本目录是功能矩阵文档（`docs/capability-matrix.html`）与其机读形式
（`docs/capability-matrix.data.json`）的**唯一产出与校验通道**。

> ★ 功能矩阵是 GRASPO **所有训练实验的唯一真相源**（见 `docs/capability-matrix.html` 元信息块）。
> 训练配置（`samples/configs/**/*.yaml` 等）以该文档为准派生。

## 产出模型

```
tools/matrix/template.html   （骨架 / 版式 / 定义文字）
docs/capability-matrix.data.json（机读数据：列定义 + 统一口径 + 逐档参数与结果）
        └── render_matrix.py ──► docs/capability-matrix.html（人读正式件）
        └── verify_matrix.py ──► 重新渲染 + 逐字节比对 + 结构/内容/脱敏断言
```

**HTML 不得手工编辑。** 任何修改都必须改「模板」或「数据」，再经 `render_matrix.py` 重新产出；
`verify_matrix.py` 会重新渲染并与在盘 HTML 逐字节比对，任一字节不同即失败，并打印首个差异的
字节偏移 / 行号 / 列号。

> 本文件（`README.md`）承载**作业说明**；正式文档 `docs/capability-matrix.html` 只保留
> **定义与表格**，不含作业过程与决策记录。

## 命令

```bash
# 渲染（唯一允许的 HTML 产出通道）
python3 tools/matrix/render_matrix.py \
    --template tools/matrix/template.html \
    --data docs/capability-matrix.data.json \
    --out docs/capability-matrix.html

# 校验（逐字节比对 + 结构 / 内容 / 脱敏断言）
python3 tools/matrix/verify_matrix.py \
    --template tools/matrix/template.html \
    --data docs/capability-matrix.data.json \
    --html docs/capability-matrix.html

# 三组自证（空数据 PASS / 假数据 PASS / 手改 1 字符 FAIL + 报偏移）
python3 tools/matrix/run_self_proof.py
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `template.html` | 文档骨架、版式与**定义文字**（判据定义 / 参数定义 / 状态定义） |
| `render_matrix.py` | **唯一渲染通道**：模板 + 数据 ⇒ HTML（确定性，不读时钟 / 环境 / 网络） |
| `verify_matrix.py` | 执法：重新渲染 + 逐字节比对 + 手改定位 + 结构 / 内容 / **脱敏**断言 |
| `build_official.py` | 由内部数据生成**脱敏**的正式机读件 `docs/capability-matrix.data.json` |
| `build_data.py` | 由抽取结果构建内部数据（列定义 + 统一口径 + 逐档参数） |
| `extract_params.py` / `extract_spec.py` | 从配置来源抽取逐档参数与规格（内部用） |
| `backfill_matrix.py` | 跑次结果回填：跑次记录 ⇒ 数据 ⇒ 重渲染 ⇒ 校验（增量、失败即停） |
| `run_self_proof.py` | 三组自证 |

## 脱敏（强制）

正式件（HTML 与 data.json）**不得出现**宿主路径、内网地址、机器名、人名。
`verify_matrix.py` 的 ④G 断言在每次校验时扫描这些模式，**命中即校验失败**；
`build_official.py` 在生成机读件时先行替换为符号路径，并对跑次只保留**不透明跑次 ID**。


## 设计规则：表格之外一切写死不变，变量全在数据

★ **全文确保只有矩阵随测试与 debug 变化，其他部分写死不变**（主席裁决）。

| 层 | 内容 | 是否随跑次变化 |
|---|---|---|
| `template.html` | 版式、定义（判据定义 / 参数定义 / 状态定义）、表头、边界声明、统一口径块、生成标记 | **写死不变**——不含任何随跑次变化的数值或文案 |
| `capability-matrix.data.json` → `rows` | 档位清单、逐档参数、逐档结论与读数 | **只有这里随跑次变化** |
| `capability-matrix.data.json` → `meta` | 固定文案（边界声明、判定口径、权威性声明、状态词表） | 写死不变；与表格数据**物理分开** |
| `capability-matrix.html` | 由上面两者确定性生成 | 结果 = **只有表格区随数据变化** |

**推论：**

- ⛔ **任何"计数 / 汇总 / 结论性散文"不得出现在表格之外**（它们会随跑次变化，一旦写死就得人工维护）。
- **表格区是唯一随数据变化的部分**；表格之外在数据变化时**逐字节不变**。

**校验方式**：`run_self_proof.py` 的「④ 表格之外逐字节不变」——只改 `rows` 里的数值重新渲染，
断言矩阵表之外的前缀与后缀**逐字节相同**（表格内则确实变化）。

## ★ 与宪法 §17.7 的一处已知冲突（已登记）

§17.7 明文要求「未达覆盖度门槛时**必须**在文档首屏写明显式警示，并列出『尚未验证』的档位清单与数量」。
**主席裁决（明确豁免）：首屏警示与统计全部去掉**——理由是面对用户与测试的功能矩阵要不断更新，
「只有表格是变化的，其他都不变」，写死警示与统计会带来人工维护负担；且当前半成品状态不会推送。

**处置**：本工具链已按主席裁决**移除** `{{COVERAGE_WARNING}}` / `{{UNVERIFIED_LIST}}` 及其派生产出；
**冲突逐字登记**于交付说明与工作日志；**§17.7 修订请求由指挥部提交宪法团队**。
除该条外，§17.7 的其余强制项全部保留（生成标记 / 四态文字标签 / 口径完整 / 同源生成 / 逐字节重建 / 脱敏）。


## 通用规则：结论必须与同一行的参数一一对应

★ **若一行的参数已被更正，而结论来自旧参数下的跑次 ⇒ 该结论作废，该档标「尚未验证」；
仅在参数冻定后重跑，才可给出「已验证可用 / 已验证不可用」判定。**

- 落地：跑次结论携带「结论有效性注记」（数据字段 `conclusion_annotation`，`conclusion_valid: false` + 作废理由 + 参考记录）；
- 生成器处理：`build_official.py` 见 `conclusion_valid: false` ⇒ **状态回落为「尚未验证」**，并把旧跑次事实写入**备注**，
  显式标注"不对应当前行参数 ⇒ 判定作废、不作为该档可用/不可用判定"（⛔ 事实不丢，但**不得用作判定**）；
- 用词：`尚未验证` 等同 **未测试**；宪法 §17.7 的固定取值为「尚未验证」，用词以它为准。

## 流程纪律（强制）

★ **长文本写入后、`git add` 前必须先跑语法检查**：
```bash
python3 -m py_compile tools/matrix/<改动文件>.py     # 或等价的 AST 解析
```
理由：本战役曾两次因长文本里的**裸双引号/未闭合字符串**引入语法错（都当场发现、未入库，但成本白付）
⇒ **提交前语法检查**已固化为本工具链的固定步骤（与 `verify`/自证同批执行）。

★ **`git add` 一律逐条点名文件**（⛔ 不对目录 add）：曾因对 `tools/matrix` 目录 add，把他人未提交的单行改动一并带入提交。

## 常设检查项（系统性要求）

★ **凡新增「入口 / 产物 / 字段 / 依赖」，必须"同批"改其校验器·登记表·测试** —— ⛔ 不得只加入口不登记。
本战役已为此痛过多次：`PARAM_KEYS` 滞后、Dockerfile pin 引号空格、入口 POLICY 未登记（`tests/e2e/launch_matrix54.sh` 致闸门 4 变红）等。
⇒ 提交前对照本清单自查：**新增入口 ⇒ 登入 `tests/scripts/test_entry_smoke.py` 的 POLICY/EXCLUSIONS（含冒烟）**；新增产物 ⇒ 同批加 `verify_matrix.py` 断言；新增字段 ⇒ 同批改 `PARAM_*`/渲染/文档。

## 状态字符串纪律（强制）

★ **状态/结论字符串一律走规范化匹配，⛔ 不用精确相等**（`== "通过"` 这类写法**禁止**）。
理由：跑次会写**带括注**的形态（如 `通过（已验证可用）`、`失败（效果）`、`失败（效果）` 等），精确相等会把"通过"判成非通过 ⇒ 本战役已真实发生一次（T013 一度被误判为「已验证不可用」，幸在提交前自检拦住）。
★ 统一入口：`backfill_matrix.conclusion_of(raw)` ⇒ `pass` / `fail` / `other`（**前缀匹配**）。
★ **负例必须被区分**：`未通过` / `非通过` / `不通过` **不以「通过」开头** ⇒ 归 `other`，⛔ 不得判过（已在 `--self-test` 用例 ⑫ 钉住）。

## 身份字符串纪律（与状态字符串同源）

★ **身份 / 状态字符串一律走规范化匹配，⛔ 不用精确相等**（`== "全参"`、`== "通过"` 这类写法**禁止**）。
★ 统一入口：`backfill_matrix.norm_ident()` —— **只去**：**括注（全/半角）**、**markdown 强调符**、**首尾空白**、**大小写**。
★★ **⛔ 不得做任何可能"合并真实不同值"的归一**（如把 `全参` 与 `LoRA`、`native` 与 `ms-swift`、`dp_size=2` 与 `dp_size=1` 混同）——否则"防错位"防线会被归一化**悄悄打薄**。
★ 本战役同类根因已两次被拦下（均 fail-closed 零写入）：① `通过（已验证可用）`；② `全参（full-parameter）` vs `全参`。
★ **负例已钉在 `--self-test`**：`全参（full-parameter）`≡`全参` ✓／`仅视觉`≠`全参`／`LoRA`≠`全参`／`native`≠`ms-swift`／`dp_size=2`≠`dp_size=1`。

## 脱敏纪律（强制 · 对抗性验证）

★ **脱敏规则必须"顺序无关"，并用对抗性输入验证** —— 因为规则顺序一旦让某条**先行规则把路径打碎**，后面的规则就再也匹配不到。
★ 真实事故（2026-10-13）：`/models/` 排在 `/home/` **之前** ⇒ `/home/<u>/models/Qwen3.8-27B` 被打碎成 `/home/<u>Qwen3.8-27B` ⇒ **任何带斜杠的规则都匹配不到** ⇒ 泄漏进 `data.json` 2 处。
★ 因此 `sanitize_str()` **末尾必须有一道与顺序无关的兜底**：`/home/<seg>[/<seg>…]` 残形整体抹掉。
★ **回归自检**：`python3 tools/matrix/build_official.py --self-test` —— 含三例：① 真实案例宿主绝对路径 ⇒ 0 命中；② 无尾斜杠同形 ⇒ 被抹掉；③ ★ **反转 `PATH_MAP` 后仍须 0 命中**（顺序无关）。
★ 交付前另跑 **`grep -c <用户名> docs/capability-matrix.{data.json,html}` ⇒ 必须 0**。

## 审计件纪律（强制 · 断言拦截）

★ **审计件（台账等）写失败必须由断言拦截，⛔ 不能靠人看输出。**
★ 理由：**落盘与登记之间一旦异常，数据与台账会不一致，而输出看起来仍然成功**。
★ 真实事故（2026-10-13，`4862a73` 修复）：触发方式标记引用了未定义变量 ⇒ `NameError`，且**发生在数据三件 `copyfile` 落盘之后** ⇒ **数据写成、台账行丢**；traceback 又落在当时 `grep '^OK|VERIFY'` 的可视范围之外 ⇒ 表现为"既报 +1 行、又不见行"。
★ 因此：**落盘之前**必须断言 **"本次处理的每一档都已生成台账行"**（`ledger_covers`），缺行即 **FAIL-CLOSED、拒绝落盘**（用例 `--self-test` ⑲ 钉住）。
★ 另：审计件路径**一律锚定数据目录**（⛔ 不用相对 CWD 的默认值，避免散落 stray 文件）。

## 数值口径纪律（强制 · 精确推导 + 容差）

★ **一切阈值判定相关的数值，必须由精确分数推导（⛔ 不用已四舍五入的显示值反推），比较时给 `1e-3` 量级容差。**
★ 真实事故（2026-10-13，`2f33e8b`）：报告侧按 4 位小数显示 `0.1424`，而精确值是 `0.1425` ⇒ **在阈值边缘会翻转判定**；编排侧**生成器自检拦下**该错，已改为"精确分数推导 + 1e-3 容差"并加回归用例（`tests/scripts/` 91 passed）。
★ 适用范围：Δ（pp）、准确率、阈值比较、`steps_actual` 与推导步数对账等**一切进入判定的数值**；★ 显示值可以四舍五入，**判定不得基于显示值**。

## 卡号纪律（§12 一族 · 强制）

★ **物理卡号不进配置**（`launch` 块只留 `cards: N` ＋ `nproc_per_node: N`，两者**同源**）；★ **卡集由起跑侧按实测整卡空闲 ＋ `docker ps` 我方绑定动态择取**（§12/§13/§17）。
★ 真实事故（2026-10-13）：派生档硬写 `gpus: [1, 7]` ⇒ 编排侧传的 `"4,6"` **被配置覆盖** ⇒ 撞上被占用卡 —— ★ **逐卡门禁正确拒绝、零副作用**，但问题在**产物里 pin 了卡号**。
★ 断言：生成器**拒绝产出**任何含 `gpus:` 的配置（用例 ⑮/⑯）。

## 可用性轴 vs 效果轴（强制 · ⛔ 不得混用）

★ **卡数/依赖"不可行" ⇒ 记 `availability: "blocked"` ＋ `blocked_reason`（可用性轴）**；
★ **⛔ 不得写成四态里的「已验证不可用」** —— 那是**效果层否定**（方法/输出退化）。★ 混用会**污染结论语义**（后人会把"硬件跑不动"读成"该方法失败"）。
★ 真实裁定（2026-10-13，T044）：该档 canonical `cards=2` **实测 OOM**（`Tried to allocate 1.89 GiB / 1.23 GiB free / 78.01 GiB in use`、`exitcode=1`），该族 `min_cards=4`（T045 4 卡实测可行）⇒ ★ **保持四态「尚未验证」**（未获得效果读数）＋ `availability: blocked` ＋ 理由"**这不是效果结论，是硬件不可行**"。
★ 断言：**同一档不得同时标 `blocked` 与「已验证不可用」**（用例 ⑯b/⑯c，冲突即 fail-closed）。

## 对账式纪律（强制 · 按档型选公式，⛔ 不得跨档套用）

★ **对账式必须按"该档自身的训练口径"选公式**：
| 档型 | 公式 |
|---|---|
| **GRASPO**（含 27B GRASPO LoRA 族） | `8×20×8 / cards`（★ T038 2 卡 = **640**；T045 4 卡 = **320**） |
| **ms-swift SFT/CPT** | `epochs×subset / nproc`（★ 如 `2×100 / nproc`） |
| **native** | ★ **按实测**（`measured_authority`） |
| **OPD** | ★ `epochs×(subset×G)/(micro×grad_accum×nproc)`（含 `nproc`） |
★ **⛔ 不得跨档套用公式**（与"没按本档自身实体取值""实测权威"同族）。
★ 真实事故（2026-10-13）：对 **GRASPO 档 T045** 误用 ms-swift SFT 口径 `2×100/4 = 50` ⇒ ★ 正确为 `8×20×8/4 = **320**`；★ 该档 `steps_actual` 按实测取 **320**（与 T038 的 640 同族）✓。

## 族级结论的作用域纪律（强制 · ⛔ 不得跨后端外推）

★ **族级结论必须限定作用域（后端/形态/卡数），⛔ 不得跨后端外推；★ 且"标注"与"活证据"必须双向相容。**
★ 真实事故（2026-10-13）：把 native 的 `T040`（1 卡 OOM）与 ms-swift 的 `T044`（2 卡 OOM）概括成"该族 `min_cards=4`"⇒ ★ **误标了 ms-swift 的 `T043`（1 卡），而它当时已在训练 768/1280** ⇒ ★ **标注与活证据直接冲突**（★ 与今日 T016 native 9B 全参单卡 OOM vs T019/T020 ms-swift 成功 **同类现象**：**后端不同、显存行为不同**）。
★ 处置：① **撤销 T043/T041 的 `blocked`**；② 族级条目改为 **`by_backend`**（native / ms_swift **各列各自实测**）；③ **双向断言**：`family_under_min_unblocked`（防"该标没标"）＋ `blocked_contradicted_by_live`（防"**不该标乱标**"：标了 blocked 却有活证据 ⇒ fail-closed）。

## 可行性结论的登记口径（强制 · 配置五元组 + ⛔ 不得假设单调）

★ **可行性结论必须按「配置五元组」登记**：**model × algo × mode × backend × cards** ⇒ ★ **⛔ 不得跨卡数 / 跨后端外推**。
★ ★★ **且不得假设单调**："卡数越多越可行" **不成立**：★ 真实实证（2026-10-13）—— ★ **9B GRASPO 全量族实测非单调**（★ 1 卡可跑、**2 卡 OOM**）；★ 可能的机理：**2 卡配置改了 per-device batch / rollout 并行度 ⇒ 单卡占用反而升高**（⛔ 未下结论）。
★ 断言（新增）：`combo_infeasible_still_runnable` —— ★ **同族同后端内已被实测证伪的 `(cards)` 组合**，⛔ **不得再有标为可起的同组合档**（用例 ⑰g/⑰h/⑰i，含"**不同后端不套用**"负例）。
★ 与既有两条断言构成**三重防线**：`family_under_min_unblocked`（防该标没标）＋ `blocked_contradicted_by_live`（防不该标乱标）＋ `combo_infeasible_still_runnable`（防**同配漏标**）。
★ **归属纪律**：★ **tier 表的 `backend`/`cards` 与跑次侧报告不一致时，⛔ 不得单侧采信** —— ★ 如实标出差异并**请上级裁定**（★ 2026-10-13 实例：矩阵表记 T035 = native、跑次侧报 ms-swift ⇒ 已在 manifest `graspo_9b_full_cards.open_question` 如实登记）。

## 归属纪律（强制 · 以配置与运行产物为准）

★ **档的归属（backend / cards / algo）⛔ 不得以叙述或记忆为准** —— ★ **一律以"配置产物 + 运行产物"钉死**：
★ 配置产物 = `samples/configs/matrix54-*/<T>.yaml`（`backend` / `launch.cards` / `nproc_per_node` / `native.*`）；★ 运行产物 = `runs/<T>/…` 的 `entry.sh` 与 stdout 头部（实际 `--backend` / `nproc` / 卡集）。
★ **实证（2026-10-13）**：跑次侧叙述称「T038 = 1 卡」⇒ ★ 但 **`T038.yaml` = `backend: msswift` + `cards: 2`**、且 **`T038.json` 的对账式 `8×20×8/2 = 640`（分母 = cards = 2）** ⇒ ★ **矩阵表正确、叙述有误**；★ 据此**撤回了"同族非单调"的说法**（真实情况是 **native 2 卡 OOM / ms-swift 2 卡可行 —— 后端不同**）。
★ **引用纪律**：★ 结论若**引用了其他档的实测（外推）** ⇒ ★ **必须同时引用被引档的配置字段**（`backend` / `cards` 等），否则 **fail-closed**（用例 ⑰j/⑰k）。

## 清单与结论一致性（强制）

★ **"已判定"的档 ⛔ 不得在待跑清单里标 `ready` 而不加注明**（★ 否则会被**重跑**，浪费算力：2026-10-13 T028 实测发生过）。
★ 处置：★ manifest 的逐档条目增 **`matrix_conclusion`**（四态）＋ **`list_stale: true`**（标明"清单滞后"）；★ 若该档是**同档再测** ⇒ 另加 **`retest`**（如 `flush 修正重测`）。
★ 断言/作业：★ **每次回填后须扫一遍**"四态 ∈ {已验证可用, 已验证不可用} 而 manifest `status == ready`"的档，逐一列出并标注。

## `status` 单一真相源（强制）

★ **`status` 必须由矩阵推导，⛔ 不得手工维护**：`derive_tier_status(matrix_status, availability, running)`
| 矩阵状态 | `status` |
|---|---|
| 四态 ∈ {已验证可用, 已验证不可用} | **`judged`**（在跑 ⇒ `running`） |
| `availability == blocked` | **`blocked`**（优先于四态判定） |
| 四态 = 未纳入验收 | `not_applicable` |
| 其余 | `ready` |
★ 事故（2026-10-13）：`status` 手工维护 ⇒ **21 档"已判定却标 `ready`"** ⇒ **T028 被误重跑，白耗一张卡数小时**。
★ 断言：**四态已判定 ⇒ `status` 不得为 `ready`**（用例 ⑱②–⑱⑤，违规即 fail-closed）。

## 产物形态纪律（强制 · 逐档实测，⛔ 不得按族/后端/mode 推断）

★ **"评测通道有无"取决于该档实际产出的终态格式（`adapter` / `native_rank` / HF）⇒ ★ 必须逐档实测（§16），⛔ 不得按 `mode`（LoRA/全参）、`backend` 或 `model` 推断。**
★ 实证（2026-10-13）：★ 旧推断"native LoRA 走 adapter、不撞缺口"**被 T022（27B SFT LoRA native）推翻** —— 它同样产出 `native_rank`；★ 而同为 27B native LoRA 的 T024 **有通道**（已评测）⇒ ★ **配置字段无法预判**。
★ 保护：★ `FORM_CHANNELS`（单一真源）＋ `assert_form_has_channel` ⇒ **起跑前探测到无通道即拒跑**（★ 本次实测证明该保护必要）。
★ 影响面口径：★ **"已确认（实测）＋ 疑似（待测）"分段**，⛔ 不用"族 × mode"聚合。
★ **补充（2026-10-13）**：★ "通道有无"的**语义开关**已定位 = **`export.final_formats`**（`src/graspo/flow/backend/native/trainer/checkpoint.py:38`：`if path.name == "final" and self.config.export.final_formats and self.runtime.is_primary()`）⇒ ★ **但矩阵各档均未设该项** ⇒ ★ 差异来自**运行期条件**；★ **建议跑次侧/生成器为每档显式设 `export.final_formats`** ⇒ **使形态可预判、从源头消除缺口**（★ 在未设之前仍须逐档探测）。

## 资源度量纪律（强制 · 在"被使用的路径"所在挂载上量）

★ **资源余量必须在"该资源被使用的路径"所在挂载上量，⛔ 不得以 `/` 或任意挂载代替**（与"别假设环境存在""卡上没显存≠卡没人用"同族）。
★ 实证（2026-10-13）：★ 我曾量 `/` ⇒ **1.5 T 可用**；★ 而**产物实际落在 `/data/remote`** ⇒ **5.5 T 可用**（★ 误差近 4 倍、且结论的"余量"判断依据错误）。
★ 同类：★ GPU 余量看目标卡上；★ 磁盘看产物落点挂载；★ 网络看目标链路。

## 校验对象 = 实际执行对象（强制 · 本日反复撞到的同一族）

★ **产物级断言的对象必须是"实际被执行的 config"**（如 `runs/<T>/<T>/config.yaml`），⛔ **不是"生成器写出的 yaml"**。
★ 实证（2026-10-13，T023）：★ 部署树 `T023.yaml:23` 有 `final_formats: ["merged-hf"]`，而**运行 config 的 `export.final_formats = []`** ⇒ ★ `checkpoint.py:38` 的门不成立、导出未触发、`ec=1`、磁盘零增量。
★ 工具：`tools/matrix/check_run_config.py --run-config <runs/…/config.yaml> --deploy-yaml <部署树 yaml>` ⇒ **不一致即 FAIL-CLOSED**（用例见 `tests/scripts/test_check_run_config.py`）。
★ 注意 `schema.py:981` 的默认值 = `[]` ⇒ ★ 若装配链**未携带 yaml 的 `export` 段**，默认空值会静默生效 ⇒ **须逐档比对运行 config 与部署树**。
★ **今日同族三例**：① 断言过了但产物没对（T023）② 改动未提交 ⇒ 交付物与本地不一致 ③ 段落未透传 ⇒ 共同根因 = **"校验对象 ≠ 实际执行对象"**。
★ **归属对账（补充）**：★ **manifest / 矩阵 / 部署树 yaml 三处的 `backend`/`cards` 应逐档对账**（2026-10-13 全表对账：**不一致 0**；★ T032 三处一致 = `ms-swift`，此前"记错"的怀疑**不成立**）。

## 计数纪律（强制 · 先判"重测"还是"新判定"）

★ **预测四态演进前，必须先判定该档是「新判定」还是「已判定档的重测」**：
| 情形 | 四态总数变化 |
|---|---|
| **已判定档的重测/更新** | ★ **0**（只更新档内结论；★ 旧测量留痕、新测量生效，按 `(档号, json sha)` 并存） |
| **新判定**（该档原不在四态内） | ★ **对应一格 +1 / 另一格 −1** |
★ 断言：`verdict_transition(already_judged, before, after)` ⇒ ★ **已判定 ⇒ 期望变化 0**；★ 未判定且本次出结论 ⇒ **+1**（用例 ⑲a/⑲b/⑲c）。
★ 事故（2026-10-13/14）：★ **人工算术两次出偏差**（把"已判定档的重测"当成"新增判定"⇒ 预期 10/21/3/20，实际 10/22/3/19）⇒ ★ **计数漂移现由机器拦截**。

## launch 段键集合（强制 · ⛔ 只写镜像认识的键）

★ **镜像 `LaunchConfig` 允许键（逐字来自镜像报错）**：`determinism / master_addr / master_port / nnodes / node_rank / nproc_per_node / python`；★ **`extra="forbid"` ⇒ 越界直接 `SystemExit`**。
★ 事故（2026-10-13，P0）：★ 生成器曾写 `launch.cards` ⇒ ★ **整个 `matrix54-generated` 代际（51/51）在当前镜像上都起不来**（★ run 侧至今能跑全靠**旧代际**文件）。
★ 现规定：★ **只写 `nproc_per_node`**（★ 值仍由 `cards` 推导 ⇒ §12"同源"落在它上面；★ 卡数如需留档 ⇒ **写注释**，⛔ 不作为字段）；★ 断言 `launch_keys_invalid()` ⇒ 越界即 FAIL-CLOSED（用例 ⑳①–⑳⑤）。

## ★★ "两棵树"（⛔ 陈述时**必须注明扫的是哪一棵**）

| 树 | 位置 | `launch.cards` 实测 |
|---|---|---|
| ★ **仓内（矩阵侧产物）** | `samples/configs/matrix54-generated/` | ★ 修复前 **51/51 含**（★ 即**新代际整体不可跑**）；★ **修复后 = 0** |
| ★ **部署树（编排侧装配/同步）** | 宿主部署树 | ★ 只有 **T023** 含（★ 即**旧代际仍在用、仅 T023 被新代际覆盖**） |
★ **两者不矛盾**：★ **仓 = 新代际全不可跑**；★ **部署树 = 旧代际可跑、仅 T023 被覆盖** ⇒ ★ **grep 类陈述一律注明树名**。
★ **分工**：★ **矩阵侧**只保证"**仓内产物内容正确**"（键合规 ＋ 目标档含 `export` ＋ 回归通过）；★ **编排侧**负责"**选 active 代际 ＋ 同步 ＋ 打印实际读取 config 的路径与 sha256**"。
