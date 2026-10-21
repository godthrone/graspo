#!/usr/bin/env python3
"""render_matrix.py — 由「模板 + 数据」渲染出功能矩阵 HTML（唯一允许的 HTML 产出通道）。

用法：
  python3 render_matrix.py --template template.html --data matrix-data.json \
                           --out capability-matrix-new.html
  python3 render_matrix.py --template template.html --data matrix-data.json --stdout

设计要点（对应主席硬要求 ★★）：
- HTML 的一切内容 = template.html（骨架 / 版式 / 第 1/2/3 章静态文字 / 第 4 章三态图例列表）+ 数据件
  （逐档参数 11 列 + 结果 6 列的**行数据**）。
- ★★ 使用者口径（2026-10-19 裁定）：「JSON 里面只有测试结果，脚本只负责把测试结果填写到最后一章
  测试表里面，其它所有的部分都不能动」。落地为三条：
    ① 表格结构（列 key / 表头 / 列组 / 顺序）= **本文件模块级常量**（`MATRIX_COLUMNS` / `MATRIX_GROUPS`），
       ⛔ 不再随数据件走（数据件已无 `columns` / `groups`）；
    ② 第 1/2/3 章 + 目录 + 页脚 + 首屏注释 = **模板静态文字**（模板里不再有这些章节的占位符）；
    ③ 渲染输出里**只有**最后一章表格的单元格随行数据变（自证见 `run_self_proof.py` ⑲）。
- 结果列禁止手工回填：本脚本是唯一写入 HTML 的通道；verify_matrix.py 会重新渲染并逐字节比对。
- 渲染确定性：不读时钟、不读环境、不读网络；同一 (template, data) ⇒ 同一字节串。
- 支持两类列：
    * 普通列：一个数据字段一格（当前 19 列**全部**是普通列）；
    * 合并列（列定义里带 "merge"）：把若干字段按 "sub_labels" 组装成「小节 + 值」的一格。
      ★ 2026-10-21「第 4 章重构」轮后**已无列使用**它（原 `data_scale` / `steps` 两列按主席令删除）
        —— 实现保留、供后续需要时复用，⛔ 不是死码（`merge_cell_inner` / `cell_display_text` 同源引用）。
"""
import argparse, html, json, re, sys
from pathlib import Path

#: ★★ 模板占位符白名单（**只列模板里真实存在的**）：`render()` 逐个替换，替换后若仍见 `{{`
#:   ⇒ 直接 SystemExit（防"占位符漏配 ⇒ 静默把 `{{X}}` 发出去"）。
#:   ★ 2026-10-19（「脚本只填最后一章表格」）：第 1/2/3 章、目录、页脚、首屏注释已改为模板静态
#:     文字 ⇒ 原 `{{CRITERIA}}` / `{{EVAL_INFO}}` / `{{VENUE}}` / `{{CODE_VERSION}}` /
#:     `{{UNIFORM_PARAMS}}` / `{{VERDICTS}}` / `{{COVERAGE_WARNING}}` / `{{BOUNDARY}}` /
#:     `{{GEN_BANNER}}` 等占位符一律下架（⛔ 不留"静态区仍由数据现算"的通道）。
PLACEHOLDER_TOKENS = ("{{DOC_TITLE}}", "{{GENERATOR}}", "{{DATA_PATH}}", "{{MATRIX_TABLE}}")

#: ★★ 19 列的功能矩阵表结构（**单一真相源**，宪法 §1.4）：
#:   档号 → 是否通过 → 主要参数 11 列 → 实验数据 5 列 → 备注 1 列；两级表头 + 5 个列组。
#:   ★ 2026-10-21「第 4 章重构」轮（主席口头指令）：渲染面由 21 列收敛为 **19 列** ——
#:     删 `lora_target` / `data_scale` / `steps` 三列；新增渲染列 `model_tbl`（模型，值只 `9B`/`27B`）；
#:     `lr` / `max_epochs` 两列**按主席给的删除条件实测不成立而保留**（第 3 章无 lr/epoch）。
#:   ★ 2026-10-19 对齐轮曾由旧版 `65ac5e7` 的 21 列逐字恢复；**本轮（分区阈值 ≤2/≥3 轮）**再把
#:     分区阈值由「< 5」改为「**≤ 2**」⇒ 表格列从 13 列扩回 **21 列**（13 参数 + 6 结果 + 档号/状态）。
#:     （列 key / 表头 / 列组归属 / 出现顺序 / sticky / group_start）从数据件 `columns` 上移到这里
#:     —— 数据件从此**只有行数据**（逐档参数 + 结果），结构不再随数据走。
#:   ★★ 2026-10-19 条件/结果**分区轮**（主席原话：两张表的并集须覆盖全部参数、交集为空）；
#:     2026-10-19 **阈值 ≤2/≥3 轮**（主席原话：先看"我有多少卡、多少显存、多大模型 ⇒ 能选哪些
#:     并行方案（如 dp2tp2 / pp4tp1dp1）、后端、速度与显存是多少"这张**查表**用途）：
#:     「取值枚举数量 **≤ 2**」的参数**上提到第 3 章「测试条件」列出**
#:     ⇒ 由本常量里带 `into="conditions"` 的列标记；`cond_label` **逐字**给出该参数在第 3 章
#:     那一条 `<li>` 里的前缀（`<b>cond_label：</b>`）—— 它是"渲染器常量 ⇄ 模板静态文字"的**唯一
#:     同源锚点**（校验端 ④Z-e 按它逐项查第 3 章，⛔ 不另抄一份模板清单）。
#:     「取值枚举数量 ≥ 3」（不同档位之间会变）的参数**留在表格列**（本常量里 group="param"
#:     且**不带** `into` 的列，见 `TABLE_PARAM_KEYS`）—— ★ 尤其 **TP 必须留在表里**：
#:     并行方案（TP×PP×DP）是查表的第一等公民，缺了 TP 就无法回答"dp2tp2 / pp4tp1dp1 哪个更快"。
#:     两条清单由**同一个常量**现算
#:     —— 校验端（`verify_matrix` ④Z）与自证端（`run_self_proof`）都从这里取，⛔ 不各留一份。
#:   ★ 消费方（⛔ 不得各留一份）：
#:       `build_official.COLUMNS` / `build_official.GROUPS`（生成端）＝ 直接引用本常量；
#:       `verify_matrix.EXPECTED_COLUMNS` / `EXPECTED_GROUPS`（校验端）＝ 由本常量现算。
#:   ★ 合并列（`merge`）：把若干字段按 `sub_labels` 组装成一格（机制保留，本轮后无列使用）。
MATRIX_COLUMNS = (
    {"key": "tier", "label": "档号", "group": "base", "sticky": True},
    {"key": "status", "label": "是否通过", "group": "status", "group_start": True},
    # ── 以下 3 条（`into="conditions"`）是**第 3 章「测试条件」的登记项**：54 档取值枚举数量 ≤ 2 ──
    #    ★ 2026-10-21「第 4 章重构」轮（主席口头指令）：模型 / 模式 / 后端 **同时**在表格里逐档给出
    #      （`model_tbl` / `mode_tbl` / `backend_tbl`，均带 `source` 指回同一行字段）。
    #      主席原话「模型写 9B 和 27B 就行了，因为前面已经写的很清楚了」⇒ 表格侧只给短值
    #      （`9B` / `27B`），第 3 章保留全名（`Qwen3.5-9B` / `Qwen3.8-27B`）。
    #      ⚠ 代价如实登记：这三项因此成为"同一参数两处发布"，与 ④Z「交集为空 / 条件参数在表格
    #      ≥ 3 个不同取值」两条字面判据相抵 —— 取舍以主席原话为先（`verify_matrix` 已于
    #      2026-10-20 退役、不再作为交付门禁，见其文件头退役说明）。
    {"key": "model", "label": "模型", "group": "param", "into": "conditions",
     "cond_label": "基础模型"},
    {"key": "mode", "label": "模式", "group": "param", "into": "conditions",
     "cond_label": "可训练参数（模式）", "source": "mode"},
    {"key": "backend", "label": "后端", "group": "param", "into": "conditions",
     "cond_label": "训练后端", "source": "backend"},
    # ── 以下 **11 列**留在表格「主要参数」组：逐档差异项（前 4 项顺序 = 主席 2026-10-21 指定的 ③–⑪）──
    #    ★ 并行方案三列（TP / PP / DP）必须相邻给出：读者按"卡数 × 显存 × 模型"查表时先看这一组。
    #    ★ 单元格取值规范（2026-10-21 轮）：TP / PP / DP / 总卡数 **只写数字或 `—`**；
    #      数据件里 `—（框架内部决定）` / `N（派生）` / `nproc_per_node×nnodes = N 卡` 这类
    #      说明性文字已在外科清理中归一（查不到真值的档位记 `—`，逐档清单见工位 report.md）。
    #    ★ 表格列 key 必须唯一，而 `model` / `mode` / `backend` 已被上面的条件登记占用
    #      ⇒ 表格侧用 `model_tbl` / `mode_tbl` / `backend_tbl` 作内部别名，`source` 指回行字段。
    {"key": "model_tbl", "label": "模型", "group": "param", "group_start": True,
     "source": "model"},
    {"key": "algo", "label": "算法", "group": "param"},
    {"key": "mode_tbl", "label": "模式", "group": "param", "source": "mode"},
    {"key": "backend_tbl", "label": "后端", "group": "param", "source": "backend"},
    {"key": "tp", "label": "TP", "group": "param"},
    {"key": "pp", "label": "PP", "group": "param"},
    {"key": "dp", "label": "DP", "group": "param"},
    {"key": "cards_total", "label": "总卡数", "group": "param"},
    # ★ `lr` / `训练轮次` **保留**：主席给的删除条件是"第 3 章「统一超参」已逐算法写明 lr 与 epoch"。
    #   实测第 3 章那 11 条统一超参里**既无 lr 也无 epoch**（template.html:173-189）⇒ 按主席条件分支
    #   保留该两列；取值已在数据件里为纯数字（lr: 5.0e-06 / 2.0e-05 / —；轮次: 2 / 4 / 8）。
    {"key": "lr", "label": "lr", "group": "param"},
    {"key": "max_epochs", "label": "训练轮次", "group": "param"},
    {"key": "switches", "label": "关键开关", "group": "param"},
    # ── 实验数据 5 列 + 备注：2026-10-21 轮**原样不动** ──
    {"key": "peak_mem", "label": "显存峰值(GiB)", "group": "data", "group_start": True},
    {"key": "steps_actual", "label": "实跑步数", "group": "data"},
    {"key": "elapsed", "label": "耗时（每步/总时长）", "group": "data"},
    {"key": "throughput", "label": "吞吐", "group": "data"},
    {"key": "effect", "label": "效果（评测 Δ / loss·reward 窗口）", "group": "data"},
    {"key": "note", "label": "备注（条件 / 限制 / 失败归因）", "group": "note", "group_start": True},
)
#: ★★ 列组的**逐字**定义（第一行表头 colspan 由它给出）：列组顺序 == 列出现顺序。
MATRIX_GROUPS = {"base": "基础", "status": "状态", "param": "主要参数",
                 "data": "实验数据", "note": "备注"}

#: ★★ 条件/结果分区（2026-10-19「条件/结果分区轮」）的**分区常量** —— 全部由 `MATRIX_COLUMNS`
#:   **现算**（§1.4 单一真相源；⛔ 不许在下游各留一份清单，⛔ 不许写死档号或取值）：
#:   ① `CONDITION_PARAMS`：**上提第 3 章「测试条件」**列出的参数（列标记 `into="conditions"`）；
#:   ② `TABLE_PARAM_KEYS`：**留在表格**「主要参数」组的列 key（含合并列的**全部**子键）；
#:   ③ 两者**并集 ⊇ 该档全部训练参数**、**交集 = ∅** —— 由 `verify_matrix` ④Z 逐档机核。
#:   ★ 归一化键：表格列 **⇄ 行字段**的对应关系由列的 `source` 给出（缺省 = 列 key 同名）；
#:     需要显式 `source` 的三列是 `model_tbl` / `mode_tbl` / `backend_tbl`（见上方注释：同一分类参数
#:     既要第 3 章给全名/全义、又要表格里逐档可比 ⇒ 表格侧用内部别名 key，⛔ 行字段名不变）；
#:     "轮次"列的 key 直接用数据行的权威字段 `max_epochs`
#:     （⛔ 不新造 `epoch` 别名：行里那个 `epoch` 字段是历史遗留、54/54 为 `None`）。
#:   ★ **`步数` 列是单字段列（⛔ 不是合并列）**：主席要求"保留完整推导文本"，而数据行 `steps`
#:     字段本身就是 `2 epoch × 100 步/epoch = 推导 200 步`（含轮次 + 每 epoch 步数 + 总步数）
#:     ⇒ 再叠一层"推导 200 步（…）"就是**重复发布同一信息**（且取值基数会被拆成两列）。
#:     故 `steps` 直接呈现行字段原文，`derived_steps` 不再进分区清单（⛔ 它本是同一信息的第三份）。
#:   ★ `into` 列**不出现在渲染结果里**（`build_matrix_table` 只渲染不带 `into` 的列）
#:     ⇒ 第 3 章的清单在页面上是**模板静态文字**，它的"列了哪几项"由本常量 → 模板一致性机核守住。
CONDITION_PARAMS = tuple((c["key"], c.get("cond_label", c["label"])) for c in MATRIX_COLUMNS
                         if c.get("into") == "conditions")
#: ★ `TABLE_PARAM_KEYS` = 表格「主要参数」组各列**真正读取的行字段名**（含合并列的全部子键）
#:   —— 它是"条件 ∪ 表格 ⊇ 逐档训练参数"这条并集判据的**表格侧清单**（④Z-a）。
#:   ⛔ 用 `source`（而非列 key）才是对的：列 key 是面向读者的**表头身份**，行字段是**数据身份**，
#:   两者在 `mode_tbl` / `backend_tbl` 上刻意不同（见上）；并集判据比的是**数据身份**。
TABLE_PARAM_KEYS = tuple(src for c in MATRIX_COLUMNS
                         if c["group"] == "param" and not c.get("into")
                         for src in (c.get("merge") or (c.get("source", c["key"]),)))
TABLE_PARAM_COLUMNS = tuple(c["key"] for c in MATRIX_COLUMNS
                            if c["group"] == "param" and not c.get("into"))
RENDERED_COLUMNS = tuple(c for c in MATRIX_COLUMNS if not c.get("into"))
#: 表格列 key → 它**读取的行字段**（`source`，缺省 = 同名列 key；合并列取全部子键）。
COLUMN_SOURCES = {c["key"]: tuple(c.get("merge") or (c.get("source", c["key"]),))
                  for c in MATRIX_COLUMNS}
#: 一个参数在同一页上**只允许有一列承载它的取值**（§1.4 单一真相源、④Z-b1）：
#:   行字段 → 承载它的列 key 列表（正常长度恒为 1）。
SOURCE_OWNERS = {}
for _c in MATRIX_COLUMNS:
    for _f in COLUMN_SOURCES[_c["key"]]:
        SOURCE_OWNERS.setdefault(_f, []).append(_c["key"])

#: ★★ **第 3 章「测试条件」逐项登记表**（`CHAPTER3_CONDITION_ITEMS`）—— 模板第 3 章里每一条
#:   `<li><b>条目名：</b>…</li>` 都必须在下面**登记**，并声明该条是"哪个参数的取值发布处"：
#:     - `row_field`：该条**承载其取值**的行字段（⇒ 这条在第 3 章发布该参数的值）；
#:     - `row_field=None`：**说明式条目**（caliber 条目）—— 只解释某参数的取值是**怎么来的**，
#:       ⛔ 不在本页发布任何取值（例：`TP 口径` 说明 TP 由谁决定，TP 的取值在表格 "TP" 列）。
#:   ★ 这张表把"第 3 章列了哪几项"从**注释/人眼**变成**可机核清单**（④Z 逐条比对），
#:     并让"两处发布/漏项/越界基数"的判据有**参数身份（行字段）**可比，⛔ 不靠列名字面串。
#:     `verify_matrix` ④Z-c1/c2 与自证 ㉕ 都从这里取（§1.4：⛔ 不各留一份）。
CHAPTER3_CONDITION_ITEMS = (
    # 值登记（这些参数在 54 档里的取值枚举数量 ≤ 2 ⇒ 归第 3 章，表格不列）
    {"label": "基础模型", "row_field": "model", "kind": "cond"},
    {"label": "可训练参数（模式）", "row_field": "mode", "kind": "cond"},
    {"label": "训练后端", "row_field": "backend", "kind": "cond"},
    # 说明式条目（caliber）：主席本轮点名要保留的"TP 口径"——
    #   TP 的**取值**回到表格列（查表第一等公民，`tp` 枚举 2 ⇒ 本可以是条件项，
    #   但它是并列方案的组成项、必须与 PP/DP 同屏可比），此处只说明"这个值是怎么来的"
    #   ⇒ 两处不是同一件事。★ `row_field="tp"` 表示"该参数已由本登记在案"
    #   （基数判据由此**不重复计数**；⛔ 不写死任何档号/取值）。
    {"label": "TP 口径", "row_field": "tp", "kind": "caliber"},
)
#: 第 3 章条目**名**（供机核逐条比对；⛔ 顺序即页面顺序）。
CHAPTER3_ITEM_LABELS = tuple(i["label"] for i in CHAPTER3_CONDITION_ITEMS)
#: 第 3 章**已登记在案**的行字段集合（含说明式条目：如 `TP 口径` → `tp`）——
#:   基数判据（④Z-c1）据此判断"这条参数是否已由第 3 章登记过"，⛔ 不写死清单、⛔ 不看列名字面串。
CHAPTER3_FIELDS = frozenset(i["row_field"] for i in CHAPTER3_CONDITION_ITEMS if i.get("row_field"))


def condition_field(label, items=None):
    """第 3 章某条目**承载取值的行字段**；说明式条目（caliber）返回 `None`。

    `items` 缺省 = 本模块的 `CHAPTER3_CONDITION_ITEMS`（§1.4 单一真相源）；
    ⛔ 未登记的条目名返回 `None` 并**不**等于"放行"——`verify_matrix` ④Z-c3 会把
    "条件列标签未登记"直接判红（本函数只负责查表）。
    """
    for _i in (items if items is not None else CHAPTER3_CONDITION_ITEMS):
        if _i["label"] == label:
            return _i.get("row_field")
    return None


def merge_cell_inner(col, row):
    """★ 合并列的**唯一拼装实现**（渲染 HTML ⇄ 校验端期望文本**同一份代码**，§1.4 单一真相源）。

    - 槽位顺序 = `col["merge"]`；分隔符 = `col["merge_sep"]`（含 `<br>` 时逐字插入，供换行）；
    - `col["derived"][sub]` = 展示层模板（如 `推导 {v} 步`）：⛔ 只改**呈现**、不改数值；
    - `col["sub_labels"][sub]` = 小节名（如 `训练数据版本：`）；标点型标签（`（`）由
      `merge_close` 收尾（如 `）`）。
    """
    labels, derived = col.get("sub_labels", {}), col.get("derived", {})
    close = col.get("merge_close") or ""
    sep = col.get("merge_sep", " ／ ")
    sep_html = sep if "<" in sep else html.escape(sep)
    parts = []
    for sub in col["merge"]:
        v = row.get(sub)
        v = "" if v is None else str(v)
        if sub in derived:                       # 展示层模板（如 `推导 {v} 步`）
            v = derived[sub].format(v=v)
        lbl = labels.get(sub)
        if lbl:
            cls = "hp" if sub in ("hparams",) else "f"
            parts.append(f'<span class="{cls}" data-f="{html.escape(sub)}">{html.escape(lbl)}</span>'
                         f'<span data-v="{html.escape(sub)}">{html.escape(v)}'
                         f'{html.escape(close) if close else ""}</span>')
        else:
            parts.append(f'<span data-v="{html.escape(sub)}">{html.escape(v)}</span>')
    return sep_html.join(parts)


def merge_cell_text(col, row):
    """与 `merge_cell_inner` **逐字同源**的纯文本形态（去标签，供校验端比对期望值）。"""
    return html.unescape(re.sub(r"<[^>]+>", "", merge_cell_inner(col, row)))


def cell_display_text(row, key):
    """★ **展示层**的单元格文字（供校验端算"该格应该是什么"；⛔ 不读渲染结果反推）。

    - 普通列：行里同名字段的原文（★ 列带 `source` 时读 `row[source]`，见 `mode_tbl` / `backend_tbl`；
      `effect` 列另经 `human_view_effect` 归一，见调用方）；
    - 合并列：与渲染**同一份**拼装实现（`merge_cell_text`）⇒ 校验端不会出现第二套规则。
    """
    col = next((c for c in RENDERED_COLUMNS if c["key"] == key), None)
    if not col:
        return "" if row.get(key) is None else str(row.get(key))
    if col.get("merge"):
        return merge_cell_text(col, row)
    src = col.get("source", key)
    return "" if row.get(src) is None else str(row.get(src))

# 结果列着色映射（此处只用于生成带 class 的单元格；绿/红/黄与第 4 章三态图例同源）
#: ★ 2026-10-21「第 4 章重构」轮：人读件的「是否通过」列只允许**三值** `通过` / `未通过` /
#:   `尚未验证`（见模板第 4 章图例）⇒ 前三条即本轮三态；后三条是 2026-10-20 之前的旧四态，
#:   仅为历史行数据（里程碑页等）仍能着色而**保留**，⛔ 当前 54 档无一命中。
STATUS_CLASS = {
    "通过": "st-pass", "未通过": "st-fail", "尚未验证": "st-pending",
    "已验证可用": "st-pass", "已验证不可用": "st-fail",
    "未纳入验收": "st-illegal",
}


def cell_html(col, row, by_method=None):
    """渲染一个数据单元格的 HTML 片段。

    ★ `by_method` 只用于「效果」列的**展示层归一**（见 `human_view_effect`）：
      `rows[].effect` 是**冻结行**里的历史读数原文，⛔ 一字不动；旧口径措辞在渲染时改写。
    """
    key, group = col["key"], col["group"]
    classes = [f"g-{group}"]
    if col.get("sticky"):
        classes.append("col-tier")
    if col.get("group_start"):
        classes.append("col-start")

    if col.get("merge"):
        return (f'<td class="{" ".join(classes + ["param"])}" data-col="{key}">'
                f'{merge_cell_inner(col, row)}</td>')

    val = "" if row.get(col.get("source", key)) is None else str(row[col.get("source", key)])
    if key == "effect":
        val = human_view_effect(val, row.get("train_method"), by_method)
    if key == "status":
        cls = STATUS_CLASS.get(val.strip(), "st-dash")
        classes.append(cls)
    return f'<td class="{" ".join(classes)}" data-col="{key}">{html.escape(val)}</td>'


#: ★★ 展示层归一（2026-10-17 修正轮 1，全局审查 B2）：
#:   `rows[].effect` 是**冻结行**里的历史读数原文，其中 T037/T038/T039 三格带已作废的
#:   旧口径措辞（「冻结阈值 +5.0 pp」/「目标 +10.0 pp 仅展示」），与同一页 §2「通过标准」
#:   自相矛盾。★ 修法**只能在展示层**（⛔ `rows[]` 一字不动），⇒ 与「rows sha 不变」不冲突。
#:   ★ 数值与方向**一律取自** `meta.criteria_caliber.threshold.by_method[<算法>]`
#:     （⛔ 本文件不写死任何阈值）；旧措辞的*模式*匹配不改变任何读数/结论。
_RE_STALE_CALIBER_PAREN = re.compile(r"（\s*冻结阈值\s*[^）]*?仅展示\s*）")
_RE_STALE_FROZEN_CLAUSE = re.compile(r"（\s*[^）]*?冻结阈值[^）]*?）")
_RE_STALE_TARGET_SENTENCE = re.compile(r"[；;]?\s*目标\s*[^。]*?仅展示[^。]*。?")
#: 兜底（防新变体漏网 ⇒ 机核 ④H 必红）：数字无关的旧措辞清除。
_RE_STALE_FROZEN_ANY = re.compile(r"冻结阈值\s*\+?\d+(?:\.\d+)?\s*pp")
_RE_STALE_TARGET_ANY = re.compile(r"目标\s*\+?\d+(?:\.\d+)?\s*pp\s*仅展示[^。]*。?")


def human_view_effect(text, method, by_method) -> str:
    """把「效果」列里的**旧口径措辞**归一为现行口径（展示层专用）。

    - 现行通过线文本逐字取自 `by_method[<算法>].pass_line_text`（唯一真相源）；
    - 数值/读数/结论⛔ 一字不改，只改**已作废的说法**；
    - `method` / `by_method` 缺失时仍然清除旧措辞（fail-safe，⛔ 不因读不到就把旧口径放行）。
    """
    line = str(((by_method or {}).get(str(method or "").lower()) or {}).get("pass_line_text") or "")
    name = str(method or "").upper() or "本算法"
    cur = f"（判据② {name} 通过线：{line}）" if line else f"（判据② {name} 通过线，见第 2 节）"
    cur_missed = (f"（未达判据② {name} 通过线：{line}）" if line
                  else f"（未达判据② {name} 通过线）")
    t = str(text)
    t = _RE_STALE_CALIBER_PAREN.sub(cur, t)
    t = _RE_STALE_FROZEN_CLAUSE.sub(cur_missed, t)
    t = _RE_STALE_TARGET_SENTENCE.sub("。", t)
    t = _RE_STALE_FROZEN_ANY.sub("判据② 通过线", t)
    t = _RE_STALE_TARGET_ANY.sub("", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def build_matrix_table(data):
    # ★★ 表结构来自**本模块常量**（⛔ 不再读 `data["columns"]` / `data["groups"]`）：
    #   使用者的口径是「JSON 里面只有测试结果，脚本只负责把测试结果填进最后一章测试表」
    #   ⇒ 列 key / 表头 / 列组 / 顺序属于**渲染器的静态结构**，不随数据走（§1.4 单一真相源）。
    #   ★★ 条件/结果分区轮：带 `into="conditions"` 的列**不上表**（它们在第 3 章静态文字里列出）；
    #      ★ 阈值 ≤2/≥3 轮：这样的条件列收敛到 **3 列**（`model` / `mode` / `backend`）
    #      ⇒ 渲染面 = `RENDERED_COLUMNS` = **21 列**（档号 1 + 状态 1 + 参数 13 + 实验数据 5 + 备注 1）；
    #      `MATRIX_COLUMNS` 仍是**唯一真相源**
    #      （第 3 章清单 ↔ 表格列清单由 ④Z 从它现算比对，⛔ 不靠第二份手抄清单）。
    cols = [dict(c) for c in RENDERED_COLUMNS]
    groups = dict(MATRIX_GROUPS)
    rows = data["rows"]
    # ★ 展示层归一用的现行通过线（唯一真相源；⛔ 渲染层不写死任何阈值数值）。
    by_method = (((data.get("meta") or {}).get("criteria_caliber") or {})
                 .get("threshold") or {}).get("by_method") or {}

    # 列组按出现顺序聚合（仅用于第一行表头的 colspan）
    order, spans = [], {}
    for c in cols:
        g = c["group"]
        if g not in spans:
            order.append(g)
            spans[g] = 0
        spans[g] += 1
    sticky_key = next((c["key"] for c in cols if c.get("sticky")), None)
    first = cols[0]
    assert first.get("sticky") is True and first["key"] == sticky_key, "首列必须是 sticky 档号列"

    out = []
    out.append('<table class="matrix">')
    out.append("<colgroup>")
    for c in cols:
        out.append(f'<col data-col="{c["key"]}" class="g-{c["group"]}">')
    out.append("</colgroup>")
    out.append("<thead>")
    head1 = [f'<th class="g-{first["group"]} col-tier col-start" data-colgroup="{groups[first["group"]]}" '
             f'rowspan="2">{html.escape(first["label"])}</th>']
    for g in order:
        span = spans[g] - 1 if g == first["group"] else spans[g]
        if span <= 0:
            continue
        head1.append(f'<th class="g-{g}" colspan="{span}">{html.escape(groups[g])}</th>')
    out.append("<tr>" + "".join(head1) + "</tr>")
    head2 = "".join(
        f'<th class="g-{c["group"]}{" col-start" if c.get("group_start") else ""}" '
        f'data-col="{c["key"]}">{html.escape(c["label"])}</th>'
        for c in cols if c["key"] != sticky_key)
    out.append("<tr>" + head2 + "</tr>")
    out.append("</thead>")
    out.append("<tbody>")
    for r in rows:
        out.append("<tr>" + "".join(cell_html(c, r, by_method) for c in cols) + "</tr>")
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)


# ★★ 2026-10-19（使用者口径「脚本只负责把测试结果填进最后一章测试表」）：原先在这里的
#   `build_criteria` / `build_eval_info` / `build_uniform_block` / `build_venue` /
#   `build_code_version` / `build_coverage_warning` / `build_boundary` / `build_gen_banner`
#   一律下架 —— 第 1/2/3 章（判据①–④ 与逐算法通过线、评测口径、实验条件统一口径块）现在是
#   **模板静态文字**（`template.html`），⛔ 不再由数据现算。
#   ⛔ 判据数值与语义一字未改：静态文字逐字取自本轮在盘渲染结果，只换了承载位置。
#   仍保留在渲染层的：`build_matrix_table`（最后一章表格，随行数据变）与
#   `human_view_effect`（「效果」列展示层归一，`verify_matrix` 同源引用）。


# ★★ 2026-10-19：`_lora_backends_html` / `_target_regex_html` 一并下架 —— 它们是**从未被
#   模板引用**的死函数（模板占位符里没有对应 token，渲染结果里也不出现其文本）。
#   本轮把「渲染器只负责填最后一章表格」写成可机核约束 ⇒ 顺带清掉这两个不再有出口的死码。
#   ⛔ 它们依赖的 `meta.lora_target_backends` 数据一字未动（仍在数据件里，另有消费方）。


def render(template_text, data):
    meta = data["meta"]
    table = build_matrix_table(data)
    # ★★ 只保留模板里**真实存在**的 4 个占位符（见 `PLACEHOLDER_TOKENS`）：
    #   第 1/2/3 章、目录、页脚、首屏注释已是模板静态文字 ⇒ ⛔ 不再从数据现算
    #   （使用者口径：脚本只负责把测试结果填进最后一章测试表）。
    repl = {
        "{{DOC_TITLE}}": meta.get("title", ""),
        "{{GENERATOR}}": (meta.get("authority") or {}).get("generator", ""),
        "{{DATA_PATH}}": (meta.get("authority") or {}).get("data_path", ""),
        "{{MATRIX_TABLE}}": table,
    }
    out = template_text
    for token in PLACEHOLDER_TOKENS:
        out = out.replace(token, repl[token])
    leftover = []
    i = 0
    while True:
        j = out.find("{{", i)
        if j < 0:
            break
        k = out.find("}}", j)
        leftover.append(out[j:k + 2] if k > 0 else out[j:j + 40])
        i = j + 2
    if leftover:
        raise SystemExit(f"ERROR: 模板中存在未替换的占位符: {leftover}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()

    template_text = Path(a.template).read_text(encoding="utf-8")
    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    out = render(template_text, data)

    if a.stdout:
        sys.stdout.write(out)
        return
    if not a.out:
        raise SystemExit("ERROR: 需要 --out 或 --stdout")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(out, encoding="utf-8")
    print(f"rendered {len(out.encode('utf-8'))} bytes -> {p}")


if __name__ == "__main__":
    main()
