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
