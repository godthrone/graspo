# `scripts/matrix/` — 功能矩阵文档工具链

本目录是功能矩阵文档（`docs/capability-matrix.html`）与其机读形式
（`docs/capability-matrix.data.json`）的**唯一产出与校验通道**。

> ★ 功能矩阵是 GRASPO **所有训练实验的唯一真相源**（见 `docs/capability-matrix.html` 元信息块）。
> 训练配置（`samples/configs/**/*.yaml` 等）以该文档为准派生。

## 产出模型

```
scripts/matrix/template.html   （骨架 / 版式 / 定义文字）
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
python3 scripts/matrix/render_matrix.py \
    --template scripts/matrix/template.html \
    --data docs/capability-matrix.data.json \
    --out docs/capability-matrix.html

# 校验（逐字节比对 + 结构 / 内容 / 脱敏断言）
python3 scripts/matrix/verify_matrix.py \
    --template scripts/matrix/template.html \
    --data docs/capability-matrix.data.json \
    --html docs/capability-matrix.html

# 三组自证（空数据 PASS / 假数据 PASS / 手改 1 字符 FAIL + 报偏移）
python3 scripts/matrix/run_self_proof.py
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
