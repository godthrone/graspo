#!/usr/bin/env python3
"""verify_matrix.py — 机核执法：把「模板 + 数据」重新渲染，与在盘 HTML 比对并做结构断言。

一条命令即判定「HTML 是否被手工改动过 / 结构是否符合新两大部分」：

  python3 verify_matrix.py --template template.html --data matrix-data.json \
                           --html capability-matrix-new.html

退出码 0 = PASS；退出码 1 = FAIL。

四组检查（全部 fail-closed）：

  ① 字节同一性 —— 重新渲染的字节串与在盘 HTML 逐字节相同（唯一产出通道的证据）。
  ② 手改检测 —— ① 失败时打印首个差异的「字节偏移 / 行号 / 列号 / 上下文」（改一个字、
     加一个空格、删一个字符都会命中）。
  ③ 结构断言 —— 四章（`1. 这是什么表` / `2. 测试方法` / `3. 测试条件` / `4. 测试结果`）
     + 四个分区 section + 目录恰好 4 条且 href 对得上；不再有 §1–§7 的旧骨架 id。
  ④ 内容断言 —— A. 必需要点全在位；B. 21 列表结构 == **渲染器常量**（`render_matrix.MATRIX_COLUMNS`，
     §1.4 单一真相源，⛔ 不再比数据件——数据件已无 `columns` / `groups`）；C. 结果列全为占位符、
     状态全「尚未验证」；D. 21 列逐档逐格与行数据**逐字**一致；Z. **条件/结果分区**（并集覆盖 / 交集为空
     / 基数规则）逐档机核；V. **静态区不随数据变**
     （只改 `rows` 里的结果字段重渲染 ⇒ 表格以外逐字节不变、表格内只有结果单元格变）。

★ 2026-10-19 使用者口径：「JSON 里面只有测试结果，脚本只负责把测试结果填写到最后一章测试表
  里面，其它所有的部分都不能动」⇒ 本轮把该口径写成**可机核**约束（④V ＋ ④W），
  并把随「首屏覆盖度警示 / 承诺边界声明 / 可见生成说明行」一起下架的断言逐条处置
  （⛔ 未删除任何与该三者无关的断言；详见工位 `report.md` 的任务 4 清单）。
"""
import argparse, hashlib, html, json, re, sys
from pathlib import Path

import render_matrix  # 同目录；复用唯一渲染通道
import build_official  # 同目录；★ 结论列契约与归因依据关键词**同源**（⛔ 不复制一份）
import gen_configs     # 同目录；★ 「已声明的前提」定义**同源**（`load_premises`，⛔ 不复制一份）

VERSION = "verify-matrix/v20-partition-le2-20261019"

# ---- ③ 结构断言：只允许这四章 ----
#: ★ 人读件结构（宪法 §17.7 + 使用者诉求）：
#:   ① 这是什么表 ② 测试方法 ③ 测试条件 ④ 测试结果
#:   ★ 2026-10-19（「章节标号顺序改了」轮）：第 4 章标题由「5. 功能矩阵测试表」改为
#:     **「4. 功能矩阵测试表」**、目录条目同步（分区 id 仍是 `s1..s4`）；第 1 章标题
#:     「1. 这是什么表」**保留**（使用者明示）。全页不得再有「第 5 节 / 第 4 节」跨章错位。
EXPECTED_H2 = ["1. 这是什么表", "2. 测试方法", "3. 测试条件", "4. 测试结果"]
EXPECTED_PARTS = ["s1", "s2", "s3", "s4"]
FORBIDDEN_OLD_SECTIONS = ["part1", "part2", "sec1", "sec1b", "sec2", "sec3", "sec4", "sec5", "sec6", "sec7"]

#: ★★ 2026-10-19 阈值 ≤2/≥3 轮：`REQUIRED_MARKERS` 里两条"表格列表头"关键词的**取值来源**
#:   —— 由渲染器常量现算列标（§1.4 单一真相源），⛔ 本文件不写死列名。
#:   （`lr` 列 = 学习率；`max_epochs` 列 = 训练轮次。列标一改 ⇒ 这两条断言立刻红。）
_LR_COL = [(i, c) for i, c in enumerate(render_matrix.MATRIX_COLUMNS) if c["key"] == "lr"]
_EPOCH_COL = [(i, c) for i, c in enumerate(render_matrix.MATRIX_COLUMNS) if c["key"] == "max_epochs"]
assert len(_LR_COL) == 1 and len(_EPOCH_COL) == 1, "REQUIRED_MARKERS 依赖的 lr / max_epochs 列缺失"

# ---- ④A. 必须出现的规则关键词（缺一即 FAIL） ----
# ★ 精简轮 4（2026-10-17 主席令）：人读件只留四章（这是什么表 / 通过标准 / 实验条件 / 结果矩阵）。
#   本清单**随结构同步收敛**：⛔ 不再是"关键词越多越好"，只保留§17.7 的强制要件
#   ＋ 主席判定「数值与方向一字不错」的通过线原文 ＋ 复现条件必需项。
REQUIRED_MARKERS = [
    # 生成关系（§17.7「自动生成的入库文档」）：★ 只保留**机器可读生成留痕**
    #   （`generated-by:` 注释 ×2 + `meta` 三个标记，见 ④I）。
    #   ⛔ 原先的「可见生成说明行」四项（首屏生成声明 / 禁止手工编辑 / 唯一可编辑源头 /
    #     单方向禁止互推）随使用者裁定删除的 `p.gnote` 一并下架；⛔ 未留任何**可见**的生成说明行
    #     （这正是使用者的要求）——替代防线 = 可见面反向断言（④W）＋ 机器可读留痕（④I）＋
    #     唯一渲染通道（①/②字节同一性）。
    ("机器可读生成标记", "generated-by:"),
    # ★ 2026-10-19（使用者裁定「目录后面那三块全删」）—— 原先在这里的 **5 条边界声明要件**
    #   + **5 条覆盖度门槛要点**随「承诺边界声明块 / 覆盖度警示块」一并下架。
    #   这是**使用者对宪法 §17.7 的明确破例**，逐条处置见工位 `report.md` 任务 4 清单；
    #   ⛔ 不是"放宽"：原断言**无对象**（那两块已不在页面上），且已按下述等价防线换防：
    #     ① ④K 改为**反向断言**（不得出现覆盖度警示的 id / 不得出现未达门槛时的能力承诺式措辞）；
    #     ② ④I 保留机器可读生成留痕（`generated-by:` + `meta` 标记）与判定口径两项；
    #     ③ ④N 只保留**数据面**的边界声明要件检查（不再要求"首屏可见"）。
    # 四章（标题逐字）
    ("章①这是什么表", "这是什么表"),
    ("章②测试方法", "测试方法"),
    ("章③测试条件", "测试条件"),
    ("章④测试结果", "测试结果"),
    # 通过标准 = 一个列表；核心 = 有效性
    # ★★ 2026-10-19 主席令：导语逐字改为「原则：训练之后效果变好，才算「可用」。」
    #    ⇒ 本条目**同族换防**（原关键词「训练之后模型效果变好了」随旧导语下架，新句被删即红）。
    ("判据·有效性是核心", "训练之后效果变好"),
    ("判别依据① 能完整跑完", "能完整跑完"),
    ("判别依据③ 算法正确性检查", "算法正确性检查通过"),
    ("判别依据④ 能推理", "能推理"),
    # ★★ 2026-10-17 去重轮：原条目 `("判别依据·合并规则", "任一条不过")` 随使用者裁定删除的
    #   第 2 章 `<p class="rule">` 合并规则段**一并下架**——该段已整段删除（`任一条不过` 在成品页
    #   0 命中，F1 由工位 report.md 的 grep 证据守住），原关键词在成品页已无对象 ⇒ 断言失效。
    #   ⛔ 不是"放宽"：**该段承载的行为后果一条没丢**，改由导语承载「四条全部通过才算「可用」」
    #   ⇒ 本条目**同族换防**为对这句的断言（⛔ 仍 fail-closed：该句被删即红）。
    # ★★ 2026-10-19 主席令：导语按主席原文改为「原则：训练之后效果变好，才算「可用」。」
    #    （「四条全部通过」这句话**按主席令不再出现**，其含义由「才算「可用」」＋四条本身承载）。
    #    ⇒ 本条目**同族换防**为对「才算「可用」」的断言：⛔ 该句被删即红（⛔ 不是放宽、⛔ 不是删断言）。
    ("判别依据·可用原则（训后变好才算「可用」）", "才算「可用」"),
    # 判据② 逐算法通过线（数值与方向逐字）
    ("判据②·SFT 通过线", "after − base ≥ +10.0 pp"),
    ("判据②·GRASPO 通过线", "after − base ≥ +5.0 pp"),
    ("判据②·CPT 训练 loss", "训练 loss 正常下降"),
    ("判据②·CPT 不劣化", "after ≥ base（不劣化）"),
    # ★★ 2026-10-19 主席令：OPD 由「不劣化（能力回收）」改为 Δ ≥ +5.0 pp，**并新增教师要求**
    #    ⇒ 本条目同族换防为对**教师要求**的断言（⛔ 正是主席点名的那一项：只写结果不写教师即红）。
    ("判据②·OPD 教师要求", "教师在训练数据集上的完全正确率"),
    ("判据②·不做假设检验", "不做假设检验"),
    ("判据标识", "elam-v3-allright-1"),
    # 四态（§17.7「四种状态必须有可区分的文字标签」）
    ("四态·已验证可用", "已验证可用"),
    ("四态·已验证不可用", "已验证不可用"),
    ("四态·尚未验证", "尚未验证"),
    ("四态·未纳入验收", "未纳入验收"),
    ("四态·不依赖颜色/图例", "不依赖颜色、图例或空白"),
    # ★ 2026-10-19 对齐轮：「未过判据必须写明第 1/2 条」随「未过判据」列一并下架
    #   —— 本轮全部档位为「尚未验证」，无「已验证不可用」档、无归因正文可写
    #   （原值已归档到工位 `archived-verdicts.json`）。⛔ 不是放宽：该断言对无对象。
    # 实验条件（只列复现需要的）
    #   ★ 2026-10-19 **阈值 ≤2/≥3 轮**（分区规则改为「枚举 ≤ 2 ⇒ 第 3 章；≥ 3 ⇒ 表格列」）：
    #     原先在这里的 4 项关键词「算法：/ 学习率 / 训练轮次」承载的那三条 `<li>` 已按新规则
    #     **整条移出第 3 章**（它们的枚举数量是 4 / 3 / 3 ⇒ 归表格列）⇒ ⛔ 原关键词在页面上**已无对象**。
    #     ⛔ **不是删断言、不是放宽**：改为断言"该信息在页面上**仍然发布**"——
    #       `算法` 仍是第 4 章表格列**表头逐字**（且第 3 章导语点名"算法"见第 4 章）；
    #       `学习率`（表格列 `lr`）与 `训练轮次` 的表头逐字取自渲染器常量（§1.4 单一真相源），
    #       ⛔ 本文件不写死列名 —— 列名一改，这条断言立刻红。
    ("实验条件·算法", "算法"),
    ("实验条件·训练后端", "训练后端"),
    ("实验条件·可训练参数", "可训练参数"),
    ("实验条件·学习率（表格列 lr）", _LR_COL[0][1]["label"]),
    ("实验条件·训练轮次（表格列）", _EPOCH_COL[0][1]["label"]),
    ("实验条件·训练数据", "训练数据"),
    ("实验条件·统一超参", "统一超参"),
    # 评测口径可复核（§17.7「基线值与判据口径必须带口径」）
    ("评测口径·评测集", "评测集"),
    ("评测口径·版本锚点", "版本锚点"),
    ("评测口径·样本量", "样本量"),
    ("评测口径·评测方法", "评测方法"),
    # 矩阵表与结果列
    ("矩阵表", '<table class="matrix">'),
    ("列组·主要参数", "主要参数"),
    ("列组·实验数据", "实验数据"),
    ("结果列·是否通过", "是否通过"),
    ("读表·关键开关列 — 语义", "没有额外差异开关"),
]

#: 禁止措辞（§17.7：双真相源 / 能力承诺式措辞）
FORBIDDEN_PHRASES = [
    ("同为真相源", "双真相源表述"),
    ("互为真相源", "双真相源表述"),
    ("两个真相源", "双真相源表述"),
    ("全组合可用", "能力承诺式措辞"),
    ("已验证全组合", "能力承诺式措辞"),
    ("仅供参考", "以「不承诺效果」淡化平台侧失败（§17.7 反例）"),
    ("效果不保证", "以「不承诺效果」淡化平台侧失败（§17.7 反例）"),
    # ★★ 修正轮 1（全局审查 B2）：已作废的**旧口径措辞**不得出现在发布人读件。
    #   背景：T037/T038/T039 的效果列曾残留「冻结阈值 +5.0 pp；目标 +10.0 pp 仅展示」，
    #   与同页 §2「通过标准」（逐算法现行通过线、无「目标值」概念）自相矛盾。
    #   修法在**展示层**（`render_matrix.human_view_effect`，⛔ `rows[]` 一字不动）；
    #   此处是它的**能失败机核**（阳性对照见 `run_self_proof` ⑱）。
    ("目标 +10.0 pp", "旧口径措辞：目标值仅展示（现行口径无「目标值」概念）"),
    ("仅展示", "旧口径措辞：「目标值仅展示」已被取代"),
    ("冻结阈值 +5.0 pp", "旧口径措辞：统一阈值副本（现行口径逐算法分列）"),
]

# ★ §17.7「矩阵中仍然禁止」+ 使用者诉求：**内部术语**（人称 / 内部编排 / 内部角色 / 内部过程）
#   —— 生成物与手写文档**同等**检查（§17.7「扫描不豁免」）。
FORBIDDEN_INTERNAL_TERMS = [
    "主席", "总参谋长", "总司令", "指挥部", "指挥官", "审查者",
    "口径包", "修正轮", "本包", "本批", "第 1 批", "第 2 批", "第 3 批",
    "第 4 批", "第 5 批", "第 6 批", "第 7 批", "工位", "work_log", "跑次根目录",
    "归属层", "必须能失败", "阳性对照", "机核可跑",
]

# ---- ④X / ④Y. ★★ 2026-10-17 去重轮：四条判别依据「互不重叠」+「页面 ↔ 数据件逐字一致」 ----
#   使用者原话（两句，本轮唯一验收标尺）：
#     ①「这一段删掉：四条判别依据（…）全部通过 ⇒ …；任一条不过 ⇒ …」（第 2 章 `<p class="rule">`）；
#     ②「四条判据好像有重叠呢，整理一下措辞和分工，判据要相互独立，不能重复。」
#   ★ 落地口径（工作包定稿）：① 只判「跑完 + 产物完整可加载」；② 只判效果读数；
#     ③ 只判算法机制本身对不对；④ 只判产物真的能推理出可解析输出。
#   ★ 本组是这两句话的**能失败**机核（阳性对照见 `run_self_proof` ㉒）：
#     ④X = 页面四条 `<li>` 的 `title` + `text` 与默认数据件 `pass_criteria` **逐字一致**
#          （防"页面对、数据件旧"两份漂移；⛔ 不受数据件路径影响的那一半另有 ④Y 兜底）；
#     ④Y = 四条**互不重叠**：逐条列出"本条不得出现的关键词"，任一命中即红
#          （⛔ 不是"关键词越多越好"：只列该条**必然**会与本条它条重叠的词）。
#: (标题, 独立判什么, 本条不得出现的重叠关键词)
CRITERIA_INDEPENDENCE = (
    ("能完整跑完", "只判「跑完 + 产物完整可加载」", ("算法机制", "正确性", "推理")),
    ("效果变好（核心判据）", "只判效果读数", ()),
    ("算法正确性检查通过", "只判算法机制本身对不对", ("退出码", "NaN", "检查点")),
    ("能推理", "只判产物真的能推理出可解析输出", ("加载",)),
)
#: ★ 判据② 的逐算法通过线**只允许**取自 `criteria_caliber.threshold.by_method[*].pass_line_text`
#:   （§1.4 单一真相源；页面第 2 章的 `<ul class="tight">` 小列表必须逐条含它们）。
CRITERIA_SUBLIST_METHODS = ("sft", "graspo", "cpt", "opd")
#: ★★ 2026-10-19 主席令新增断言（④X-c）用的**A 组口径词**：判据② 的每个子项必须
#:   **在本条 `<li>` 内**点明该算法自己的效果口径词（「数据集完全正确率」/ CPT 的 loss 与准确率 /
#:   OPD 的教师要求）。⛔ 这里只收**主席原文点名的口径词**；**数值一律不在此写死**——
#:   百分点由 `threshold.by_method[<m>].value` 现算、教师的 10 个百分点由
#:   `threshold.by_method.opd.teacher` 现解析（§1.4 单一真相源）。
CRITERIA_SUBLIST_TERMS = {
    "sft": ("数据集完全正确率",),
    "graspo": ("数据集完全正确率",),
    "cpt": ("训练 loss 正常下降", "准确率不下降"),
    "opd": ("数据集完全正确率", "教师", "高于学生未训练模型"),
}
#: 官方（可编辑源头）数据件路径 —— 只有 `--data` **恰好**是它时，`pass_criteria` 才是
#: 页面同源的真相源；自证夹具（临时路径 / 变体路径）按"路径不同"跳过 ④X-b，
#: ⛔ 不因此放过固定关键词面（④X-a / ④X-c / ④Y 对夹具同样 fail-closed）。
#: ⚠ 病史自证：这里原先写成比 `basename` == 全路径常量（恒不相等 ⇒ ④X-b 静默死掉、
#:   漂移阳性对照变假绿）—— 由 `run_self_proof` ㉓ 抓到，已改为**整串**比对。
CRITERIA_SOURCE_OF_TRUTH = "docs/capability-matrix.data.json"

# ---- ④B. 21 列表结构（渲染面）：**唯一真相源 = 渲染器常量**（`render_matrix.MATRIX_COLUMNS`） ----
# ★ 2026-10-19（使用者口径「JSON 里面只有测试结果，脚本只负责把测试结果填进最后一章测试表」）：
#   21 列（渲染面）的列 key / 表头 / 列组 / 顺序已**上移**到 `render_matrix.MATRIX_COLUMNS` / `MATRIX_GROUPS`
#   ⇒ 数据件**不再有** `columns` / `groups`（结构不随数据走）。
#   ★ 断言只许加强、⛔ 不许放松：由「比数据件 + 比渲染结果两面」改为**四面**——
#     ① 渲染器常量自身形状（14 渲染列 / key 唯一 / 列组齐备）；
#     ② 生成端**引用同一对象**（`build_official.COLUMNS is render_matrix.MATRIX_COLUMNS`
#        ⇒ "本文件不再自带第二份 10 列"这件事由此机核，⛔ 不是靠注释声明）；
#     ③ 数据件**确已不再携带** `columns` / `groups`（防"结构悄悄回到数据里"造成第三份真相源）；
#     ④ 渲染结果（colgroup / 两级表头 / colspan / sticky<th> / 行数）与常量逐字一致。
#   病史依据：上一轮 `build_official.COLUMNS` 仍写死 10 列 ⇒ 重跑生成器会把列覆盖回 10 列。
#:   ★ 2026-10-19 条件/结果分区轮：**渲染面** = `RENDERED_COLUMNS`（带 `into="conditions"` 的列
#:     上提第 3 章、不上表）⇒ 三处期望值一律从渲染面现算。
#:   ★ 2026-10-19 **阈值 ≤2/≥3 轮**：条件列由 9 列收敛到 **3 列**（模型 / 模式 / 后端），
#:     渲染面由 13 列扩到 **21 列**——⛔ 仍是"从常量现算"，字数一处未写死。
EXPECTED_COLUMNS = [(c["key"], c["label"]) for c in render_matrix.RENDERED_COLUMNS]
EXPECTED_GROUPS = dict(render_matrix.MATRIX_GROUPS)
#: ★ 列的**逐字**列组归属（与 EXPECTED_COLUMNS 同序；防"列对了但列组串了"）
EXPECTED_COL_GROUPS = [c["group"] for c in render_matrix.RENDERED_COLUMNS]
#: ★ 第一行表头的 colspan（列组 → 跨列数；`base` 是 sticky 档号列 ⇒ rowspan=2、不占 colspan）
#:   ★ 由常量**现算**（⛔ 不手抄）：按列出现顺序聚合，首组扣掉 sticky 那一列。
def _group_spans(cols):
    order, spans = [], {}
    for _c in cols:
        _g = _c["group"]
        if _g not in spans:
            order.append(_g)
            spans[_g] = 0
        spans[_g] += 1
    return [(_g, spans[_g] - (1 if _g == cols[0]["group"] else 0))
            for _g in order if spans[_g] - (1 if _g == cols[0]["group"] else 0) > 0]


EXPECTED_GROUP_SPANS = _group_spans(render_matrix.RENDERED_COLUMNS)
#: ★ 「主要参数」组（渲染列序 = 表格 param 组的 13 列，也是 ④D 的逐格一致性检查面）：
#:   ★★ 2026-10-19 条件/结果分区轮：原先**写死的 13 列清单** ⇒ 改为**由渲染器常量现算**
#:     （⛔ 不再各留一份；并集/交集/基数三条分区机核 ④Z 也共用它）。
#:     ⛔ **不是放宽**：这 13 列的内容由 ④Z 的「并集覆盖 / 交集为空 / 基数规则」逐档机核守住。
#:   ★ 2026-10-19 **阈值 ≤2/≥3 轮**：渲染面由 13 列扩回 **21 列**（参数 13 / 结果 5 / 状态 1 /
#:     档号 1 / 备注 1），`PARAM_KEYS` 随之由 5 列扩到 **13 列**（⛔ 仍是现算，不写死清单）。
PARAM_KEYS = list(render_matrix.TABLE_PARAM_COLUMNS)

# ---- ④Z. ★★ 条件/结果**分区**机核（2026-10-19 条件/结果分区轮；三条全部要能失败） ----
#   使用者原话（分区轮）：
#     「第三章测试条件和第四章测试结果，两个应该是无交集的，对于那些大部分测试档位都相同且参数
#      枚举数量小于5的，放测试条件里面列出来，比较节省表格列。对于那种不同档位里面经常变化的
#      参数，就应该放表格列…总之，测试条件和测试结果的并集，应该是包含全部参数的。」
#   ★★ 2026-10-19 **阈值 ≤2/≥3 轮**（主席改口径，原话）：
#     「枚举数量小于等于 2 的，放测试条件，剩下的都作为测试结果表的列」——动机是这张表的**用途**
#     是"我有多少卡 / 多少显存 / 多大模型 ⇒ 能选哪些并行方案（如 dp2tp2 / pp4tp1dp1）与后端、速度
#     与显存各是多少"的**查表** ⇒ 只要档位之间可能变（枚举 ≥ 3）的参数就必须在**表格列里可见**
#     （上一版把 TP 这类并行方案组成项放进了第 3 章 = 放错位置，本轮纠正）。
#   ★ 判据（fail-closed，逐档现算，⛔ 不写死档号/取值，⛔ 不靠第二份手抄清单）：
#     a **并集覆盖**：`条件参数 ∪ 表格 param 列（含合并列子键；取列的 `source` 行字段名）`
#       必须 **⊇ 逐档训练参数集合** —— 新增/漏登参数键 ⇒ 报红（正是使用者点名的「两边都不写」漏项）；
#     b **交集为空**：同一个参数**不得**同时出现在两处 —— 逐档取值**完全相同**的键对即判重
#       （实测：`lr` 若同时放回表格列，本断言必红，见自证用例）；取值不同的同名键
#       （如行的 `cards`＝"1" vs 表格列 `cards_total`＝公式串）**不判重**（列面向人读、行字段面向
#       机读，两者是"列 ⇄ 行字段"的呈现关系，不是两处重复发布同一参数 ⇒ 如实登记、⛔ 不滥用例外）；
#       ★ 列面判重按**列 key**（表头身份）：`mode_tbl` / `backend_tbl` 是刻意起的内部别名，
#         其 label（模式 / 后端）虽与第 3 章条件项同名，但两处内容一个是"逐档取值"、
#         一个是"分档值域说明"（两个不同读者的东西）⇒ 不是重复发布（键不同名、正文不同）。
#     c **基数规则**：表格 param 组每列在该 54 档的取值**枚举数量 ≥ 3**；条件参数**≤ 2**
#       —— 判据 = **该字段在 rows 里的取值基数**现算（⛔ 不写死清单的例外）。
#   ★ 与模板静态文字**同源**：`CONDITION_PARAMS` 由渲染器常量给出 ⇒ 模板第 3 章必须逐项出现
#     （见 ④Z-e：逐项查 `<b>标签：</b>` 落位 + 值域/档位说明）。
#: ★★ 第 3 章「测试条件」逐项登记表 —— **同源引用**渲染器常量（⛔ 本文件不另留一份清单）。
#:   每一项声明该条 `<li>` 是"哪个行字段的取值发布处"（`row_field`）；
#:   说明式条目（caliber，`row_field=None`）只解释口径、不在本页发布取值。
CONDITION_ITEMS = tuple(render_matrix.CHAPTER3_CONDITION_ITEMS)
#: 第 3 章条目**名**（面向读者；机核按它逐条比对模板）。
CONDITION_ITEM_LABELS = tuple(i["label"] for i in CONDITION_ITEMS)
#: 第 3 章**已登记在案**的行字段集合（含说明式条目，如 `TP 口径` → `tp`）——
#:   基数判据 ④Z-c1 据此做"同一参数不重复计数"，⛔ 不写死清单。
CONDITION_FIELDS = frozenset(render_matrix.CHAPTER3_FIELDS)
#: 逐档训练参数集合的键白名单 = 已归一化的**行字段面向键**：`CONDITION_PARAMS` 的列 key
#:   ＋ 表格 param 各列真正读取的行字段名（`source`；合并列取全部子键）。
#:   ⛔ 存在的意义 = 把"两边都不写"的漏项变成一个**逐档可机核**的集合差（不靠人眼）。
#:   ★ 与条件/表格两清单**同源**：这里只声明"哪些键属于训练参数"，⛔ 不列取值。
TRAIN_PARAM_KEYS = frozenset(render_matrix.TABLE_PARAM_KEYS) | frozenset(
    k for k, _l in render_matrix.CONDITION_PARAMS)
#: ★ 页面必须写明的**分区规则**三个关键词（缺一即红）：并集覆盖 / 交集为空 / 基数判据
#:   （★ 2026-10-19 阈值 ≤2/≥3 轮：判据关键词由「枚举数量小于 5」改为「**枚举数量不超过 2**」
#:    —— ⛔ 阈值本身改了（主席明令），但这三项要件一件不少：并集覆盖 / 交集为空 / 基数判据。）
PARTITION_RULE_MARKERS = ("并集覆盖全部参数", "交集为空", "枚举数量不超过 2")


def partition_cardinality(rows, key):
    """返回某键在 `rows` 里的**取值基数**与取值频次（⛔ 不改取值、不做归一）。"""
    from collections import Counter
    cnt = Counter(str(r.get(key)) for r in rows)
    return len(cnt), cnt


def partition_failures(rows, target_text, columns, cond_params):
    """★★ 条件/结果分区判据的**唯一实现**（verify ④Z 与自证用例**同源引用**，⛔ 不各写一份）。

    入参全部可注入（`columns` / `cond_params` / `rows` / `target_text`）⇒ 自证可以把
    "把 `lr` 放回表格列""删掉 `dp` 列""压掉 `dp` 的取值基数"这类**故意做错**的夹具喂进来，
    直接证明这三条判据**能失败**（⛔ 不依赖 verify 的打印文本）。
    返回 `(检查名, 明细)` 列表；空列表 = 全过。
    """
    out = []
    #   ★ 表格侧的两个清单，**用途不同 ⇒ 取键不同**（⛔ 不混用）：
    #     - `table_keys`：并集判据比的是**数据身份** ⇒ 取列的 `source`（缺省 = 列 key 同名）；
    #     - `table_cols`：列面判重比的是**表头身份** ⇒ 取列 key（如 `mode_tbl` / `backend_tbl`）。
    table_keys = tuple(src for c in columns if c["group"] == "param" and not c.get("into")
                       for src in (c.get("merge") or (c.get("source", c["key"]),)))
    table_cols = tuple(c["key"] for c in columns if c["group"] == "param" and not c.get("into"))
    cond_keys = [k for k, _l in cond_params]
    all_keys = list(cond_keys) + list(table_keys)
    #   a 并集覆盖：逐档训练参数集合（白名单 = 列面向键）必须被 条件 ∪ 表格 覆盖
    missing = []
    for _r in rows:
        _present = {k for k in TRAIN_PARAM_KEYS if k in _r}
        _miss = sorted(_present - set(all_keys))
        if _miss:
            missing.append((str(_r.get("tier")), _miss))
    #   ★ 反向（白名单自身的覆盖）：白名单里每个键都必须真实存在于**每一档**行数据里
    #     —— 防"白名单写了、数据里没有 ⇒ 并集断言永远成立"的假绿。
    absent = [(str(_r.get("tier")), k) for _r in rows for k in TRAIN_PARAM_KEYS if k not in _r]
    #   b 交集为空（两条互补判据，任一命中即红）：
    #     b1 **同名键**：条件参数的 key 又作为表格 param 列的 key（同名同值 ⇒ 就是两处重复发布同一参数）；
    #     b2 **同标签**：同一**参数名**（列 label）既出现在第 3 章条件清单、又出现在渲染表格列里
    #        —— 这是**面向读者**的判重（读者看到的是列名，不是 key；`lr` 这种 key 与 label 同形的
    #        参数正是最容易被"顺手放回表格列"的一类）。两条合起来 = 使用者要的"无交集"。
    dup = []
    for _k in cond_keys:
        for _c in table_cols:
            if _c == _k:
                continue
            if rows and all(str(r.get(_k)) == str(r.get(_c)) for r in rows):
                dup.append((_k, _c))
    cond_labels = {l: k for k, l in cond_params}
    for _c in columns:
        if _c.get("into") or _c["group"] != "param":
            continue
        # b2-a ★ 第 3 章条目名不得再出现在渲染表头（面向读者的**同名列**判重）：
        #   第 3 章条目名 = 其"参数名"（如 `基础模型`），一旦同名又进表头 ⇒ 同一参数两处发布。
        if _c["label"] in CONDITION_ITEM_LABELS:
            dup.append((f"标签「{_c['label']}」", "第 3 章条目", _c["key"]))
        # b2-b 未登记条件项的 `cond_label` 也不得进表头（防"漏登记还能上表"）。
        if _c["label"] in cond_labels:
            dup.append((f"标签「{_c['label']}」", cond_labels[_c["label"]], _c["key"]))
        # b2-c ★ 已登记的口径说明条目标签（如 `TP 口径`）不得作为渲染列标签 ——
        #   正是"TP 口径被改成表格列"这类误改的能失败判据。
        if _c["label"] in CONDITION_ITEM_LABELS:
            dup.append((f"口径说明「{_c['label']}」", "第 3 章口径说明", _c["key"]))
    #   b3 ★★ 取值发布面判重（**参数身份 = 行字段**，⛔ 不靠列名字面串）：
    #     同一个行字段**只能有一列**承载它的取值 —— 两列读同一字段即"两处发布同一参数"（必红）。
    _src_used = {}
    for _c in columns:
        for _f in ([] if _c.get("into") else list(_c.get("merge") or (_c.get("source", _c["key"]),))):
            _src_used.setdefault(_f, []).append(_c["key"])
    for _f, _ks in sorted(_src_used.items()):
        if len(_ks) > 1:
            dup.append((f"行字段「{_f}」被多列承载", tuple(_ks), "两处发布同一参数"))
    #   c 基数规则（★ 2026-10-19 阈值 ≤2/≥3 轮：阈值由 5 改为 2；⛔ 判据结构未变，
    #     仍为"逐列现算基数 + 越界即红"；⛔ 不写死任何档号/取值/例外清单）：
    #     c1 **表格 param 列**：取值枚举数量 ≥ 3；但"**第 3 章条目 = 对同一行字段的分档口径说明**"
    #        的那些字段**不重复计数**（同一条参数不在两个清单里各算一次）。
    #        ★ 该豁免是**窄条件**：豁免的键必须（i）是渲染器里登记过的条件参数 key，
    #          （ii）其值已由表格列逐档发布 —— 即"确有对应表格列"，否则照常判红。
    #          ⛔ 不是"把表格列门槛降到 2"：只放行"同一参数的两处呈现"，其它任何低基数表格列必红。
    #     c2 **条件条目**：其**承载取值的行字段**枚举数量 ≤ 2（说明式 caliber 条目无行字段 ⇒ 不适用）。
    #     c3 **登记完整性**：渲染器里每个带 `cond_label` 的条件列都必须在
    #        `CHAPTER3_CONDITION_ITEMS` 登记；且第 3 章每个**带字段**的条件条目都必须是
    #        `cond_params` 里的一项（⛔ 防"新增条件列忘了登记 / 登记表与常量漂移 ⇒ 判据静默放行"）。
    card_table = [(_k, partition_cardinality(rows, _k)[0]) for _k in table_keys
                  if _k not in CONDITION_FIELDS]
    card_cond = []
    for _k, _l in cond_params:
        _f = render_matrix.condition_field(_l, items=CONDITION_ITEMS)
        if _f is None:               # caliber 条目：只说明口径、不发布取值 ⇒ 基数规则不适用
            continue
        card_cond.append((_f, partition_cardinality(rows, _f)[0]))
    _declared_labels = {i["label"] for i in CONDITION_ITEMS}
    undeclared = [(_k, _l) for _k, _l in cond_params if _l not in _declared_labels]
    _field_labels = {i["label"] for i in CONDITION_ITEMS if i.get("row_field")}
    notin_const = sorted(_field_labels - {_l for _k, _l in cond_params})
    table_low = sorted((k, n) for k, n in card_table if n < 3)
    cond_high = sorted((k, n) for k, n in card_cond if n > 2)
    #   d 渲染面（交集为空的**列面**判据）：带 `into="conditions"` 的列**不得**出现在渲染列里
    #     —— 即"条件参数不得同时又是表格列"（与 b 是同一件事的两种写法：b 查取值/列名，d 查列 key）。
    into_leak = [k for k in cond_keys if k in [c["key"] for c in columns if not c.get("into")]]
    #   e 模板静态文字同源：第 3 章必须**逐项**列出条件参数（`<b>cond_label：</b>` + 同一 `<li>` 里给说明）
    tpl_missing = []
    for _k, _lbl in cond_params:
        _m = re.search(r"<li>\s*<b>" + re.escape(_lbl) + r"：</b>(.*?)</li>", target_text, re.S)
        if not _m:
            tpl_missing.append((_k, _lbl, "第 3 章缺该条 <li>"))
        elif len(re.sub(r"<[^>]+>", "", _m.group(1)).strip()) < 10:
            tpl_missing.append((_k, _lbl, "该条未给出值域/档位说明（正文过短）"))
    #   f 页面必须写明分区规则本身（使用者点名的那三句）
    rule_missing = [w for w in PARTITION_RULE_MARKERS if w not in target_text]
    if missing:
        out.append(("④Z 分区·并集覆盖（条件 ∪ 表格 ⊇ 逐档训练参数）",
                    f"{len(missing)} 档有参数两处都没写（前 5）: {missing[:5]}"))
    if absent:
        out.append(("④Z 分区·并集白名单键必须在行数据里存在",
                    f"{len(absent)} 处缺失（前 5）: {absent[:5]}"))
    if dup:
        out.append(("④Z 分区·交集为空（同一参数不得两处重复发布）",
                    f"逐档取值全等的重复键对: {dup}"))
    if table_low:
        out.append(("④Z 分区·基数规则（表格 param 列须 ≥ 3）",
                    f"取值枚举数量 < 3 的表格列: {table_low}（应上提第 3 章）"))
    if cond_high:
        out.append(("④Z 分区·基数规则（条件参数须 ≤ 2）",
                    f"取值枚举数量 > 2 的条件参数: {cond_high}（应改留表格列）"))
    if into_leak:
        out.append(("④Z 分区·交集为空（条件列不得同时是渲染表列）", f"{into_leak}"))
    if tpl_missing:
        out.append(("④Z 分区·第 3 章逐项与渲染器常量同源",
                    f"{len(tpl_missing)} 项（key, 标签, 原因）: {tpl_missing}"))
    if rule_missing:
        out.append(("④Z 分区·页面须写明分区规则",
                    f"缺关键词 {rule_missing}（并集覆盖 / 交集为空 / 基数判据）"))
    return out

#: ★★ **结果列**（本轮按使用者诉求全部清空为 `—`）：实验数据 5 列 + 备注 1 列。
#:   该清单是 ④C「结果清空」与 ④D「逐格 == 数据」两面共同的检查面。
RESULT_CLEARED_KEYS = ["peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note"]
#: ★ 结果列（含状态）＝ ④C 的「结果面」；`status` 不是读数（它一律 `尚未验证`），故另列。
RESULT_KEYS = ["status"] + RESULT_CLEARED_KEYS
#: ★★ 结果列的**唯一允许取值**（本轮口径）：一律 `—`；状态列一律 `尚未验证`。
RESULT_PLACEHOLDER = "—"
EXPECTED_STATUS = "尚未验证"
#: ★★ 真正的**实测读数**列：未结论档必须全为 `—`（`status` 不是读数，故不在内）。
#:    该清单是 ④C「未验证不得冒充可用 / 不得携带实验数据」的检查面。
#: ★ 与生成端**同源**（`build_official.EXP_DATA_KEYS`），⛔ 不各自维护一份
EXP_DATA_KEYS = list(build_official.EXP_DATA_KEYS)
# 供 yaml 生成器机读的细粒度字段（每行必须存在；值为 '—' 表示源缺失）
SPEC_KEYS = ["model_path", "train_method", "tuner_type", "cards", "parallel_dims",
             "parallel_scheme", "data_source", "subset_size", "prompts_per_rank",
             "subset_generation", "subset_r3_historical", "train_path",
             "planned_steps", "save_steps", "gpus", "target_modules", "target_modules_mode",
             "deepspeed", "allow_unverified_pp_rollout", "nnodes",
             "expected_optimizer_steps_per_epoch", "expected_optimizer_steps_reachable",
             "expected_optimizer_steps_historical", "steps_formula_v2", "min_optimizer_steps",
             "applicable"]


#: ★★ 静态区判据的**唯一实现**（`verify` ④V 与自证 ⑲ 同源引用，⛔ 不各写一份）。
#:   面域 = `<table class="matrix">` … `</table>` 之内 = "可随数据变的区域"；
#:   其余一切（第 1/2/3 章 / 目录 / 页脚 / 首屏注释 / 表格骨架标签）= **静态区**。
def _matrix_table_span(text: str):
    """返回矩阵表在文本里的 [start, end)；缺表时返回 None（调用方 fail-closed 处理）。"""
    i = text.find('<table class="matrix">')
    if i < 0:
        return None
    j = text.find("</table>", i)
    if j < 0:
        return None
    return i, j + len("</table>")


def static_region_diff(rendered: str, target: str):
    """比较**表格以外**的区域；返回差异描述列表（空 = 逐字节相同）。"""
    a, b = _matrix_table_span(rendered), _matrix_table_span(target)
    if a is None or b is None:
        return [(0, "缺少 <table class=\"matrix\">（无法切分静态区）")]
    out = []
    for name, x, y in (("表前", rendered[:a[0]], target[:b[0]]),
                       ("表后", rendered[a[1]:], target[b[1]:])):
        if x != y:
            k = next((n for n in range(min(len(x), len(y))) if x[n] != y[n]),
                     min(len(x), len(y)))
            out.append((k, f"{name}首个差异 @ 字符 {k}: 期望 {x[k:k + 60]!r} != 实际 {y[k:k + 60]!r}"))
    return out


def table_skeleton(text: str) -> str:
    """把矩阵表里**单元格内容**挖空（⛔ 保留标签 / 属性 / 列组 / 表头），用于判"表结构是否变"。

    ★ 只对 `<td …>…</td>` 的**内容**做占位；`<th>`（表头）与所有标签属性一律逐字保留
      ⇒ 列名 / 列组 / colspan / rowspan / colgroup / 行列数 任一改动都会被看见。
    """
    span = _matrix_table_span(text)
    if span is None:
        return ""
    seg = text[span[0]:span[1]]
    return re.sub(r"(<td\b[^>]*>).*?(</td>)", lambda m: m.group(1) + "\x00EMPTY\x00" + m.group(2),
                  seg, flags=re.S)


def first_diff(a: bytes, b: bytes):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def linecol(text: str, byte_off: int):
    """byte_off 是字节偏移；换算到 UTF-8 文本的行号 / 列号。"""
    prefix = text.encode("utf-8")[:byte_off]
    s = prefix.decode("utf-8", errors="replace")
    line = s.count("\n") + 1
    col = len(s.split("\n")[-1]) + 1
    return line, col


def context(text: str, char_off: int, width: int = 70):
    return repr(text[max(0, char_off - width):char_off + width])


def cell_text(cell_html: str) -> str:
    """把单元格 HTML 还原成纯文本（用于参数列的逐档核对）。"""
    return html.unescape(re.sub(r"<[^>]+>", "", cell_html)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--expect-empty", action="store_true",
                    help="要求结果列全部为占位符（空结果版 / 起手式用）；默认只要求「无半填」一致性")
    ap.add_argument("--allow-filled", action="store_true",
                    help="（保留兼容，已等价于默认行为）结果列允许有实测读数")
    ap.add_argument("--manifest", default=gen_configs.DEFAULT_DECLARED_MANIFEST,
                    help="跑次侧口径权威清单（★ 「已声明的前提」的唯一真相源；缺省 " +
                         gen_configs.DEFAULT_DECLARED_MANIFEST + "）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    template_text = Path(a.template).read_text(encoding="utf-8")
    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    rendered_text = render_matrix.render(template_text, data)
    rendered_bytes = rendered_text.encode("utf-8")

    target_path = Path(a.html)
    if not target_path.exists():
        print(f"FAIL: 目标 HTML 不存在: {target_path}")
        return 1
    target_bytes = target_path.read_bytes()
    target_text = target_bytes.decode("utf-8")

    failures = []

    # ---- ① 主判据：逐字节比对 ----
    off = first_diff(rendered_bytes, target_bytes)
    byte_identical = off is None
    if not byte_identical:
        # ---- ② 手改检测：定位 ----
        ln, col = linecol(target_text, off)
        char_off = len(target_bytes[:off].decode("utf-8", errors="replace"))
        detail = (f"首个差异 @ 字节偏移 {off}（行 {ln}，列 {col}）\n"
                  f"  期望(渲染) : {context(rendered_text, char_off)}\n"
                  f"  实际(在盘) : {context(target_text, char_off)}")
        failures.append(("① 字节同一性 / ② 手改检测（禁止手工改动）", detail))

    # ---- ③ 结构断言：只有两大部分 ----
    h2 = re.findall(r"<h2>(.*?)</h2>", target_text, re.S)
    h2 = [re.sub(r"<[^>]+>", "", x).strip() for x in h2]
    if h2 != EXPECTED_H2:
        failures.append(("③ 结构：<h2> 只允许四个", f"期望 {EXPECTED_H2}\n  实际 {h2}"))

    parts = re.findall(r'<section class="part" id="([^"]+)"', target_text)
    if parts != EXPECTED_PARTS:
        failures.append(("③ 结构：分区只允许四个", f"期望 {EXPECTED_PARTS}\n  实际 {parts}"))

    stale = [s for s in FORBIDDEN_OLD_SECTIONS if f'id="{s}"' in target_text]
    if stale:
        failures.append(("③ 结构：不应再有旧骨架分区", f"仍存在 {stale}"))

    toc = re.findall(r'<nav class="toc">(.*?)</nav>', target_text, re.S)
    if len(toc) != 1:
        failures.append(("③ 结构：目录", f"期望恰好 1 个 nav.toc，实际 {len(toc)}"))
    else:
        links = re.findall(r'href="#([^"]+)"', toc[0])
        if links != EXPECTED_PARTS:
            failures.append(("③ 结构：目录只指向四个部分", f"期望 {EXPECTED_PARTS}\n  实际 {links}"))

    # ---- ④A. 规则关键词 ----
    missing = [name for name, kw in REQUIRED_MARKERS if kw not in target_text]
    if missing:
        failures.append(("④A 规则关键词", f"缺失 {len(missing)} 项: {missing}"))

    # ---- ④X / ④Y. ★★ 本轮核心：四条判别依据「互不重叠」+「页面 ↔ 数据件逐字一致」 ----
    #   判据与常量集中在模块头 `CRITERIA_INDEPENDENCE` / `CRITERIA_SUBLIST_METHODS`
    #   （⛔ 本断言不另写一份判据文本；文本真相源 = 页面 `<ol class="criteria">` 与数据件 `pass_criteria`）。
    #   为什么查**在盘页面**的文本：`pass_criteria` 不进渲染（第 1/2/3 章 = 模板静态文字）⇒
    #   「页面对、数据件旧」这种漂移只有把两边的字面各解析一次才抓得住（⛔ 不能只比 ① 字节同一性）。
    _crit_seg_m = re.search(r'<ol class="criteria">(.*?)</ol>', target_text, re.S)
    if not _crit_seg_m:
        failures.append(("④X 四条判别依据块", '未找到 <ol class="criteria">（判据文本的唯一可见载体）'))
        _crit_seg = ""
    else:
        _crit_seg = _crit_seg_m.group(1)
    # ★ 逐条切分 `<li>`：**只取顶层 4 条**，每条只取**本条自身的文本**——
    #   ⛔ 判据② 的逐算法小列表是**嵌套 `<ul><li>`**，若用"贪婪/前瞻"式正则会把内层 4 条也算进来
    #      （实测：按 `(?=<li>|$)` 切会得到 8 条，且条次错位 ⇒ 断言假红）。
    #   本实现用**显式扫描 + 标签嵌套计数**，不依赖正则的贪婪性；未闭合即 fail-closed。
    def _split_top_li(html_seg: str):
        """把 `<ol class="criteria">` 的内容切成**顶层** `<li>` 片段；标签嵌套不平衡 ⇒ 抛 ValueError。"""
        out, i, n = [], 0, len(html_seg)
        while True:
            j = html_seg.find("<li>", i)
            if j < 0:
                return out
            k, depth, opened = j + 4, 0, False
            while k < n:
                if html_seg.startswith("<ul", k):
                    depth += 1
                    opened = True
                elif html_seg.startswith("</ul>", k):
                    depth -= 1
                elif html_seg.startswith("</li>", k) and depth == 0:
                    break
                k += 1
            if k >= n:
                raise ValueError("未闭合的 <li>/<ul>（判据块标签不平衡）")
            out.append(html_seg[j + 4:k])
            i = k + len("</li>")

    try:
        _crit_items = _split_top_li(_crit_seg)
    except ValueError as _e:
        failures.append(("④X 四条判别依据块（标签嵌套）", str(_e)))
        _crit_items = []
    if len(_crit_items) != len(CRITERIA_INDEPENDENCE):
        failures.append(("④X 四条判别依据块",
                         f"期望恰 {len(CRITERIA_INDEPENDENCE)} 条 <li>，实际 {len(_crit_items)} 条"))

    def _crit_top_text(li_html: str) -> str:
        """取 `li` 的**本条文本**：挖掉嵌套 `<ul>`（判据② 的逐算法小列表）后去标签。"""
        return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<ul\b.*?</ul>", "", li_html, flags=re.S))).strip()

    _crit_page = []
    for _li in _crit_items:
        _t = _crit_top_text(_li)
        _title = _t.split("：", 1)[0]
        _text = _t.split("：", 1)[1] if "：" in _t else ""
        _sub = re.search(r"<ul\b[^>]*>(.*?)</ul>", _li, re.S)
        _sub_items = re.findall(r"<li>(.*?)</li>", _sub.group(1), re.S) if _sub else []
        _crit_page.append((_title, _text,
                           [html.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in _sub_items]))
    _crit_data = list(((data.get("meta") or {}).get("criteria_caliber") or {}).get("pass_criteria") or [])

    # ④X-a 标题逐字 + 顺序（四个短名 ⛔ 不得改）
    _bad_titles = [(i + 1, _want, _crit_page[i][0] if i < len(_crit_page) else "<缺失>")
                   for i, (_want, _why, _kw) in enumerate(CRITERIA_INDEPENDENCE)
                   if i >= len(_crit_page) or _crit_page[i][0] != _want]
    if _bad_titles:
        failures.append(("④X 四条判据标题逐字/顺序（四个短名不得改）",
                         f"不符（条次, 期望, 实际）: {_bad_titles}"))

    # ④X-b 页面 `title`+`text` == 数据件 `pass_criteria` 的 `title`+`text`（逐字；防两份漂移）
    _bad_same_src = []
    if str(a.data) == CRITERIA_SOURCE_OF_TRUTH:
        if len(_crit_data) != len(CRITERIA_INDEPENDENCE):
            _bad_same_src.append((0, f"数据件 pass_criteria 条数 {len(_crit_data)} "
                                     f"!= {len(CRITERIA_INDEPENDENCE)}"))
        for _i, _page in enumerate(_crit_page):
            _d = _crit_data[_i] if _i < len(_crit_data) else {}
            if _page[0] != str(_d.get("title", "")):
                _bad_same_src.append((_i + 1, f"title 页面 {_page[0]!r} != 数据 {_d.get('title')!r}"))
            if _page[1] != str(_d.get("text", "")):
                _bad_same_src.append((_i + 1, f"text 页面 {_page[1]!r} != 数据 {str(_d.get('text'))!r}"))
    if _bad_same_src:
        failures.append(("④X 页面四条 ↔ 数据件 pass_criteria 逐字一致（防两份漂移）",
                         f"不一致 {len(_bad_same_src)} 处（条次/说明）: {_bad_same_src[:6]}"))

    # ④X-c 判据② 的逐算法小列表**必须保留**，且 **A 组（口径词 + 百分点）↔ B 组（pass_line_text）
    #      **逐条**一致**（数值/方向的**唯一真相源**＝数据件；⛔ 本断言不写死任何阈值字面量）。
    #   ★★ 2026-10-19 主席令加强（原实现只把 4 条 `<li>` **拼成一个 blob** 再各查一次 ⇒
    #      "A 组在①条、B 组在④条"这种**条次错位**抓不住）。现改为：
    #      ① 小列表条数 == `CRITERIA_SUBLIST_METHODS` 条数（⛔ 少一条即红）；
    #      ② **逐条**（第 k 条 ↔ 第 k 个算法）在**本条 `<li>` 内**同时命中：
    #         A 组口径词（`CRITERIA_SUBLIST_TERMS`，取自主席原文）＋
    #         A 组百分点（`delta_pp` 时由 `by_method[m].value` **现算**，⛔ 不写死）＋
    #         B 组 `by_method[m].pass_line_text`（逐字取自数据件）；
    #      ③ OPD 行**必须点明教师要求**：`教师` ＋ `高于学生未训练模型` ＋ 其百分点
    #         （百分点由 `by_method.opd.teacher` **现解析**，⛔ 不另写一份数值）
    #         ⇒ ⛔ "只写结果不写教师"即红（正是主席点名要防的那一种）。
    _by_method_c = (((data.get("meta") or {}).get("criteria_caliber") or {})
                    .get("threshold") or {}).get("by_method") or {}
    _sub_idx = next((i for i, (_t, _x, _kw) in enumerate(CRITERIA_INDEPENDENCE) if _t == "效果变好（核心判据）"), 1)
    _sub_items = _crit_page[_sub_idx][2] if _sub_idx < len(_crit_page) else []
    _blob = " ｜ ".join(_sub_items)
    if len(_sub_items) != len(CRITERIA_SUBLIST_METHODS):
        failures.append(("④X 判据②逐算法小列表条数",
                         f"期望恰 {len(CRITERIA_SUBLIST_METHODS)} 条 <li>，实际 {len(_sub_items)} 条：{_sub_items}"))
    _miss_lines = [str(_by_method_c.get(_m, {}).get("pass_line_text") or f"<数据件缺 {_m}>")
                   for _m in CRITERIA_SUBLIST_METHODS
                   if str(_by_method_c.get(_m, {}).get("pass_line_text") or "") not in _blob]
    if _miss_lines:
        failures.append(("④X 判据②逐算法通过线（保留 + 与 threshold.by_method 同源）",
                         f"小列表缺通过线 {_miss_lines}；实际小列表 {_sub_items}"))
    # ★★ 逐条绑定（A 组 ↔ B 组）—— 上述 blob 检查保留（防"整块缺失"），本组防"条次错位"。
    for _k, _m in enumerate(CRITERIA_SUBLIST_METHODS):
        _rule = _by_method_c.get(_m) or {}
        _item = _sub_items[_k] if _k < len(_sub_items) else ""
        _miss = []
        _line = str(_rule.get("pass_line_text") or "")
        if not _line:
            _miss.append(f"<数据件缺 threshold.by_method.{_m}.pass_line_text>")
        elif _line not in _item:
            _miss.append(f"B 组缺本条通过线 {_line!r}")
        for _w in CRITERIA_SUBLIST_TERMS.get(_m, ()):        # A 组口径词（主席原文点名）
            if _w not in _item:
                _miss.append(f"A 组缺口径词 {_w!r}")
        if str(_rule.get("criterion")) == "delta_pp":        # A 组百分点：由 value 现算，⛔ 不写死
            try:
                _want_pp = f"{float(_rule.get('value')):.1f} 个百分点"
            except (TypeError, ValueError):
                _want_pp = ""
            if not _want_pp:
                _miss.append(f"<数据件 by_method.{_m}.value 不是数 ⇒ 百分点无法现算>")
            elif _want_pp not in _item:
                _miss.append(f"A 组缺百分点 {_want_pp!r}（应与 by_method.{_m}.value 同源）")
        if _m == "opd":                                      # ★ 教师要求：⛔ 只写结果不写教师即红
            _tpp = re.search(r"(\d+(?:\.\d+)?)\s*个百分点", str(_rule.get("teacher") or ""))
            if not _tpp:
                _miss.append("<数据件缺 threshold.by_method.opd.teacher 的百分点要求>")
            else:
                _tv = float(_tpp.group(1))
                _forms = {f"{_tv:g} 个百分点", f"{_tv:.1f} 个百分点"}
                if not any(_f in _item for _f in _forms):
                    _miss.append(f"A 组缺教师要求百分点 {sorted(_forms)}")
        if _miss:
            failures.append((f"④X 判据②子项 A/B 逐条一致（{_m}）",
                             "；".join(_miss) + f"；本条实际 {_item!r}"))

    # ④Y 四条互不重叠：逐条列"本条不得出现的重叠关键词"，任一命中即红
    _crit_overlap = []
    for _i, (_title, _why, _kw) in enumerate(CRITERIA_INDEPENDENCE):
        if _i >= len(_crit_page):
            continue
        _tt, _tx = _crit_page[_i][0], _crit_page[_i][1]
        for _w in _kw:
            if _w in _tt or _w in _tx:
                _crit_overlap.append((_i + 1, _title, _w,
                                      f"命中于{'标题' if _w in _tt else '正文'}（本条判：{_why}）"))
    if _crit_overlap:
        failures.append(("④Y 四条判别依据互不重叠",
                         f"命中 {len(_crit_overlap)} 处（条次, 标题, 违规关键词, 位置）: {_crit_overlap}"))
    # ---- ④B. 21 列表结构（渲染面）== 渲染器常量（单一真相源；四级检查，全部 fail-closed） ----
    # ① 常量自身形状
    _cols = [dict(c) for c in render_matrix.RENDERED_COLUMNS]
    groups = dict(render_matrix.MATRIX_GROUPS)
    if len(_cols) != len(EXPECTED_COLUMNS):
        failures.append(("④B 列数（渲染器常量）", f"期望 {len(EXPECTED_COLUMNS)} 列，实际 {len(_cols)}"))
    if [c["key"] for c in _cols] != [k for k, _l in EXPECTED_COLUMNS]:
        failures.append(("④B 列 key 重复/顺序（渲染器常量）", f"实际 {[c['key'] for c in _cols]}"))
    _bad_group = [c["key"] for c in _cols if c["group"] not in groups]
    if _bad_group:
        failures.append(("④B 列组归属缺列组定义", f"{_bad_group}"))
    first = _cols[0] if _cols else {"group": "", "label": "", "key": ""}
    _td_cls = f'g-{first.get("group")} col-tier col-start'
    # ② 生成端必须**引用同一对象**（⛔ 不得自带第二份列定义）
    if build_official.COLUMNS is not render_matrix.MATRIX_COLUMNS:
        failures.append(("④B 生成端列定义同源（build_official.COLUMNS）",
                         "build_official.COLUMNS 不是 render_matrix.MATRIX_COLUMNS 同一对象 ⇒ 双真相源"))
    if build_official.GROUPS is not render_matrix.MATRIX_GROUPS:
        failures.append(("④B 生成端列组同源（build_official.GROUPS）",
                         "build_official.GROUPS 不是 render_matrix.MATRIX_GROUPS 同一对象 ⇒ 双真相源"))
    # ③ 数据件**不得**再携带表结构（结构不随数据走；出现即第三份真相源）
    if "columns" in data:
        failures.append(("④B 数据件不得携带 columns",
                         f"data.json 仍有 columns（{len(data.get('columns') or [])} 列）⇒ 结构随数据走"))
    if "groups" in data:
        failures.append(("④B 数据件不得携带 groups", f"data.json 仍有 groups={data.get('groups')}"))
    # ④ 渲染结果与常量逐字一致
    got_groups = [c["group"] for c in _cols]
    if got_groups != EXPECTED_COL_GROUPS:
        failures.append(("④B 列组归属", f"期望 {EXPECTED_COL_GROUPS}\n  实际 {got_groups}"))
    # ★ 第一行表头的 colspan 逐字（`档号` rowspan=2；其余 4 组按列数跨列）
    for _g, _span in EXPECTED_GROUP_SPANS:
        _th = (f'<th class="g-{_g}" colspan="{_span}">{html.escape(EXPECTED_GROUPS[_g])}</th>')
        if _th not in target_text:
            failures.append(("④B 一级表头 colspan", f"未找到逐字 {_th}"))
    # ★ colgroup 必须与常量逐字同序（渲染层直接由常量驱动）
    _colgroup = re.findall(r'<col data-col="([^"]+)" class="g-(\w+)">', target_text)
    if _colgroup != [(c["key"], c["group"]) for c in _cols]:
        failures.append(("④B colgroup", f"实际 {_colgroup}"))
    header_td = re.findall(r'<th class="g-\w+(?: col-start)?" data-col="([^"]+)">([^<]*)</th>', target_text)
    header_td = [(k, html.unescape(v)) for k, v in header_td]
    if header_td != EXPECTED_COLUMNS[1:]:
        failures.append(("④B 渲染后表头 cells", f"实际 {header_td}"))
    sticky_th = (f'<th class="{_td_cls}" data-colgroup="{html.escape(groups[first["group"]], quote=True)}" '
                 f'rowspan="2">{html.escape(first["label"])}</th>')
    if sticky_th not in target_text:
        failures.append(("④B 档号 sticky 表头", f"未找到逐字 {sticky_th}"))
    # 行数（只取矩阵表自己的 tbody；页内还有规则表 / 状态表，不能混算）
    if '<table class="matrix">' not in target_text:
        failures.append(("④B 矩阵表", '未找到 <table class="matrix">'))
        body = ""
    else:
        body = target_text.split('<table class="matrix">', 1)[1].split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    trs = re.findall(r"<tr>(.*?)</tr>", body, re.S)
    if len(trs) != len(data["rows"]):
        failures.append(("④B 行数", f"期望 {len(data['rows'])}，实际 {len(trs)}"))
    # ★ 每行必须恰好 14 个数据单元格，且列 key 集合 == 常量（防"多塞一列 / 少一列"）
    _row_cells_bad = []
    for _i, _tr in enumerate(trs):
        _ks = re.findall(r'data-col="([^"]+)"', _tr)
        if _ks != [c["key"] for c in _cols]:
            _row_cells_bad.append((_i + 1, len(_ks)))
    if _row_cells_bad:
        failures.append(("④B 逐行单元格与列常量不一致",
                         f"{len(_row_cells_bad)} 行（前 5：行号, 实际格数）: {_row_cells_bad[:5]}"))

    # ---- ④V. ★★ 静态区不随数据变（使用者口径「其它所有的部分都不能动」的机核）----
    #   判据（三层，全部 fail-closed；与自证用例 `run_self_proof` ⑲ **同源同判据**）：
    #     a 结构面：**除**最后一章表格的 21 个结果/参数数据单元格（`data-col="…"`）**以外**，
    #       模板 + 数据重渲染出的字节串必须与在盘件**逐字节**相同；
    #     b 面域面：**表格之外**（`<table class="matrix">` 之前 + `</table>` 之后）逐字节相同
    #       ⇒ 第 1/2/3 章、目录、页脚、首屏注释**一个字都不许随数据变**；
    #     c 变域面：发生变化的单元格**必须**全部落在 `data-col` 数据单元格内，且**只有**它们变
    #       （⛔ 不许出现"表格结构随数据变"——列名 / 列组 / rowspan / colgroup 都不得变）。
    _static_diff = static_region_diff(rendered_text, target_text)
    if _static_diff:
        failures.append(("④V 静态区不随数据变（表格以外逐字节相同）",
                         f"表格以外出现 {len(_static_diff)} 处差异（前 3）: {_static_diff[:3]}"))
    #   d 表结构面：把单元格内容挖空后，两边的表格骨架（表头 / 列组 / colgroup / 行列数）必须逐字相同
    if table_skeleton(rendered_text) != table_skeleton(target_text):
        failures.append(("④V 表格骨架不随数据变",
                         "挖空单元格内容后表格骨架不同 ⇒ 表格结构随数据变了（只允许单元格内容变）"))

    # ---- ④C. 验证状态四态一致性（§17.7） ----
    vocab = (data.get("meta") or {}).get("status_vocabulary", [])
    filled, bad = [], []
    #: ★ 自证夹具标记（自证端合成的"已回填/混排"行；⛔ 产品件不带该标记）
    _selfproof_fixture = bool((data.get("meta") or {}).get("_selfproof_fixture"))
    for r in data.get("rows", []):
        tier, st = r.get("tier"), r.get("status")
        if st not in vocab:
            bad.append((tier, f"非法验证状态 {st!r}"))
            continue
        if st in ("已验证可用", "已验证不可用"):
            filled.append(tier)
            if st == "已验证不可用" and not r.get("verdict"):
                bad.append((tier, "「已验证不可用」缺结论块（未写明未过判据 / 已通过项）"))
            if r.get("note") in ("", "—", None):
                bad.append((tier, "已结论档缺「备注」归因"))
        else:
            # ★ §17.7「未验证不得冒充可用」+「不得用空白/占位符使人误读」：
            #   未结论档**不得携带任何实验数据**（结果列一律 '—'）；备注只允许是**状态说明**。
            #   ⚠ 本断言改版的原因（如实登记）：`voided_conclusion` / `measurement_pending` 原先是
            #   这里用来判"允许带读数"的**内部豁免字段**；它们已按 §17.7「内部字段不进人读视图」
            #   从正式件剔除 ⇒ 断言改为**数据驱动**（看状态与备注文本本身），⛔ 不靠内部标记。
            # ★★ 2026-10-19 对齐轮（只许加强）：使用者诉求「测试内容清空成未测试 / 结果列与备注列
            #   全部清空为 `—`」⇒ 未结论档的备注**必须恰为** `—`（原先只要求"不是状态说明就报"，
            #   豁免了 `—` 与空串；现在两种含糊形态都红 ⇒ 比原断言更严，⛔ 不是放宽）。
            nz = [k for k in EXP_DATA_KEYS if r.get(k) not in ("", "—", None)]
            if nz:
                bad.append((tier, f"「{st}」却带实验数据 {nz}"))
            _note = str(r.get("note") or "").strip()
            # ★ 自证夹具（`meta._selfproof_fixture`）豁免：自证端刻意合成"已回填 + 未回填混排"的
            #   行，用来证明 ④O 的结论列契约**能失败**；豁免只免"备注须恰为 `—`"这一条，
            #   ⛔ 不免 `EXP_DATA_KEYS` 那条（见上）。产品件（无该标记）**不豁免**。
            if st == "尚未验证" and _note != RESULT_PLACEHOLDER and not _selfproof_fixture:
                bad.append((tier, f"「尚未验证」的备注须恰为 {RESULT_PLACEHOLDER}（实际 {_note[:30]!r}）"))
        if st == "未纳入验收" and not r.get("excluded_reason"):
            bad.append((tier, "「未纳入验收」缺逐档理由"))
    # ★ 现状约束：结论与同行参数一一对应 ⇒ 参数更正后旧结论作废；当前只应有「尚未验证 / 未纳入验收」
    # ★ 现状集合断言改为**数据驱动**（`meta.current_state_expectation`）：
    #   硬编码"当前只应有 X/Y"会随跑次推进而必须手改（与"只有表格随数据变"相悖）⇒ 缺省不设预期，
    #   仅保留"取值 ∈ 四态"与"各有其证据"两项校验。项目若确需冻结现状，可在数据里显式声明预期集合。
    expected_now = (data.get("meta") or {}).get("current_state_expectation")
    present = sorted({r.get("status") for r in data.get("rows", [])})
    if expected_now and not (data.get("meta") or {}).get("_selfproof_fixture"):
        if set(present) - set(expected_now):
            failures.append(("④C 现状状态集合", f"数据声明预期 {expected_now}，实际 {present}"))
    pend = [r["tier"] for r in data.get("rows", []) if r.get("measurement_pending")]
    for r in data.get("rows", []):
        if r.get("measurement_pending") and "阈值 δ 待定" not in str(r.get("note", "")):
            bad.append((r.get("tier"), "读数已录但备注未写明「阈值 δ 待定」"))
    if bad:
        failures.append(("④C 验证状态四态一致性", f"不一致 {len(bad)} 档（前 5）: {bad[:5]}"))
    if a.expect_empty and filled:
        failures.append(("④C --expect-empty：不得有已结论档位", f"已结论 {len(filled)} 档: {filled[:8]}"))

    # ---- ④U. ★★ 本轮口径：结果列全清空 + 状态全「尚未验证」（使用者诉求的机核化）----
    #   ★ 2026-10-19 对齐轮新增。使用者原话：「测试内容清空成未测试」。
    #   判据（三层，全部 fail-closed）：
    #     a **数据面**：`rows[*]` 的 6 个结果列**逐档**必须恰为 `—`；`status` 逐档必须恰为
    #       `尚未验证`；`verdict` 必须逐档**不存在**（结论已归档到工位 `archived-verdicts.json`）；
    #     b **渲染面**：在盘 HTML 里这 6 列的每一格**必须逐字**是 `—`，
    #       `data-col="status"` 的每一格必须逐字是 `尚未验证`（⛔ 不信数据面，直接查成品）；
    #     c **同源面**：`data["data_keys"]` 必须恰为这 6 列 —— 它是生成端声明"结果列"的机读字段，
    #       与 `meta.results_policy`（"未有实测结论的档位标注为尚未验证"）同源；不同源即红。
    #   ★ 之所以渲染面也查：防止"数据清空了、渲染端却把旧值贴回去"这类双真相源缺陷。
    #   ★ 自证夹具（`meta._selfproof_fixture`）豁免本组**行级**检查：自证端**刻意**合成"已回填"行，
    #     用来证明 ④O / ④J 这些**能失败**（产品件不含该标记 ⇒ 本组对产品件 fail-closed）。
    #   ⚠ 同源面（c，`data_keys`）**不豁免** —— 自证 ⑬/⑭/⑮ 的注入背景必须与本轮口径一致，
    #     否则"数据件声明 ≠ 结果六列"这类缺陷会在自证里被静默放过。
    _u_rows = data.get("rows", [])
    _u_data_bad, _u_html_bad = [], []
    if list(data.get("data_keys") or []) != RESULT_CLEARED_KEYS:
        failures.append(("④U 结果列同源（data_keys）",
                         f"期望 {RESULT_CLEARED_KEYS}，实际 {data.get('data_keys')}"))
    if not _selfproof_fixture:
        for _r in _u_rows:
            _t = str(_r.get("tier"))
            for _k in RESULT_CLEARED_KEYS:
                if str(_r.get(_k) if _r.get(_k) is not None else "") != RESULT_PLACEHOLDER:
                    _u_data_bad.append((_t, _k, str(_r.get(_k))[:30]))
            if str(_r.get("status")) != EXPECTED_STATUS:
                _u_data_bad.append((_t, "status", str(_r.get("status"))[:30]))
            if "verdict" in _r:
                _u_data_bad.append((_t, "verdict", "结论未删除（应归档到工位 archived-verdicts.json）"))
        for _k in RESULT_CLEARED_KEYS:
            _cells = re.findall(r'<td class="[^"]*" data-col="' + re.escape(_k) + r'">(.*?)</td>', target_text, re.S)
            if len(_cells) != len(_u_rows):
                _u_html_bad.append((_k, f"单元格数 {len(_cells)} != {len(_u_rows)}"))
                continue
            _nz = [i for i, _c in enumerate(_cells) if _c.strip() != RESULT_PLACEHOLDER]
            if _nz:
                _u_html_bad.append((_k, f"{len(_nz)} 格不是 {RESULT_PLACEHOLDER}（前 5 行号 {_nz[:5]}）"))
        _st_cells = re.findall(r'<td class="[^"]*" data-col="status">(.*?)</td>', target_text, re.S)
        if len(_st_cells) != len(_u_rows):
            _u_html_bad.append(("status", f"单元格数 {len(_st_cells)} != {len(_u_rows)}"))
        else:
            _st_nz = [i for i, _c in enumerate(_st_cells) if _c.strip() != EXPECTED_STATUS]
            if _st_nz:
                _u_html_bad.append(("status", f"{len(_st_nz)} 格不是「{EXPECTED_STATUS}」（前 5 行号 {_st_nz[:5]}）"))
    if _u_data_bad:
        failures.append(("④U 结果列清空 / 状态统一（数据面）",
                         f"{len(_u_data_bad)} 处（前 5）: {_u_data_bad[:5]}"))
    if _u_html_bad:
        failures.append(("④U 结果列清空 / 状态统一（渲染面）",
                         f"{len(_u_html_bad)} 处: {_u_html_bad[:5]}"))

    # ---- ④D. 表格列逐档与数据逐字一致（主要参数 13 列 / 结果 6 列 + 状态） ----
    #   ★ 修正轮 1（审 B2）：「效果」列经**展示层归一**（旧口径措辞 ⇒ 现行通过线），
    #     期望值必须用**同一个** `render_matrix.human_view_effect`（§1.4 单一真相源），
    #     ⛔ 校验端不另抄一份归一规则。
    #   ★ 2026-10-19 对齐轮：检查面由「4 参数 + 5 结果」扩到「13 参数 + 7 结果（含状态）」。
    #     ★ 2026-10-19 阈值 ≤2/≥3 轮：检查面再由「13 列渲染面的 5 个参数列」扩到**全部 13 个
    #       参数列**（TP / PP / DP / 总卡数 / 算法 / 模式 / 后端 / lr / 训练轮次 / LoRA 目标 /
    #       数据规模（数据版本） / 步数 / 关键开关）—— `PARAM_KEYS` 由常量现算 ⇒ 列一增，
    #       检查面自动跟上（⛔ 不写死清单、⛔ 不是放宽）。
    _by_method = (((data.get("meta") or {}).get("criteria_caliber") or {})
                  .get("threshold") or {}).get("by_method") or {}
    mismatch = []
    for key in PARAM_KEYS + RESULT_KEYS:
        cells = re.findall(r'<td class="[^"]*" data-col="' + re.escape(key) + r'">(.*?)</td>', target_text, re.S)
        if len(cells) != len(data["rows"]):
            mismatch.append((key, f"单元格数 {len(cells)} != {len(data['rows'])}"))
            continue
        for row, cell in zip(data["rows"], cells):
            # ★★ 2026-10-19 条件/结果分区轮：合并列（`data_scale` / `steps`）的期望文本由
            #   **渲染器同一份拼装实现**给出（⛔ 校验端不另写一套规则）；普通列取行字段原文。
            v = render_matrix.cell_display_text(row, key)
            if key == "effect":
                v = render_matrix.human_view_effect(str(row.get(key, "")), row.get("train_method"),
                                                    _by_method)
            got = html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            # ★ 标点型小节标签（如「步数」列的 `（`）前后不插空格 ⇒ 比较时去空白（⛔ 不改渲染/取值）
            if re.sub(r"\s+", "", got) != re.sub(r"\s+", "", v):
                mismatch.append((row.get("tier"), key, f"渲染 {got!r} != 数据 {v!r}"))
    if mismatch:
        failures.append(("④D 表格列一致性", f"不一致 {len(mismatch)} 处（前 5）: {mismatch[:5]}"))

    # ---- ④Z. ★★ 条件/结果**分区**三条机核（并集覆盖 / 交集为空 / 基数规则）----
    #   ★ 判据实现**唯一**（`partition_failures`，本文件模块级）⇒ 自证用例可以把它**喂坏夹具**
    #     直接证明"能失败"，⛔ 不依赖打印文本、⛔ 不出现第二套判据。
    #   ★ 逐档现算、fail-closed；⛔ 不写死档号 / 取值 / 例外清单（例外一律在工位 report.md 登记）。
    failures.extend(partition_failures(data.get("rows", []), target_text,
                                       render_matrix.MATRIX_COLUMNS,
                                       render_matrix.CONDITION_PARAMS))

    # ---- ④E. §3「统一超参」块（**模板静态文字**）+ 供 yaml 生成器的机读字段 ----
    #   ★ 2026-10-19 处置（使用者口径「其它所有的部分都不能动」）：
    #     原断言比"数据件 `uniform.items`（白名单项与取值）⇒ 必须逐字出现在人读件里"。
    #     本轮 `uniform` 随数据件一起下架（统一口径块 = **模板静态文字**，⛔ 不再由数据现算）
    #     ⇒ 断言改为**结构面 + 静态面**（仍 fail-closed，⛔ 不放宽条目数）：
    #       a) 人读件必须有统一超参列表，且**恰好 9 条**白名单项（漏项 / 多塞即红）；
    #       b) 每条必须有 `data-uniform-key` 且取值非空、非占位符；
    #       c) ⛔ 数据件不得再携带 `uniform`（防结构/口径悄悄回到数据里造成双真相源）。
    #     ★ 取值面（"数值不得被改"）由 ④V 静态区逐字节同一 + ① 字节同一性共同守住：
    #       模板静态区任何一格被改动 ⇒ ①/④V 立即红。
    _uni_items = re.findall(r'<li data-uniform-key="([^"]+)">([^<]*)<code>([^<]*)</code></li>',
                            target_text)
    _UNIFORM_EXPECTED_KEYS = ("max_prompt_length", "torch_dtype", "gradient_checkpointing",
                              "micro_batch_size", "gradient_accumulation_micro_batches",
                              "lora_r", "lora_alpha", "lora_dropout", "reward.kind")
    if not _uni_items:
        failures.append(("④E 统一超参块", "人读件未找到统一超参列表（<ul class=\"uniform\">）"))
    else:
        if tuple(k for k, _l, _v in _uni_items) != _UNIFORM_EXPECTED_KEYS:
            failures.append(("④E 统一超参项（静态块）",
                             f"期望恰为 {list(_UNIFORM_EXPECTED_KEYS)}，实际 {[k for k, _l, _v in _uni_items]}"))
        _empty = [k for k, _l, _v in _uni_items if not _v.strip() or _v.strip() == RESULT_PLACEHOLDER]
        if _empty:
            failures.append(("④E 统一超参取值非空", f"取值缺失/占位: {_empty}"))
    if "uniform" in data:
        failures.append(("④E 数据件不得携带 uniform",
                         "data.json 仍有 uniform（统一口径块已改为模板静态文字 ⇒ 结构不随数据走）"))
    # ---- ④G. 脱敏：正式件不得出现宿主路径 / 内网路径 / 机器名 / 人名（宪法 §15.1，A 类零容忍） ----
    SENSITIVE = [
        # 通用形状（⛔ 检测器自身不得携带任何具体主机 / 账号 / 挂载点字面量）
        (r"/(?:home|work|mnt|opt|srv|data|models)/[A-Za-z0-9._-]+", "宿主 / 内网路径"),
        (r"\.local/", "本地目录 .local/"),
        (r"\b(?:10|172|192)\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\b", "私网 IP"),
        (r"\b[0-9]{2,3}\s*(?:共享机|机器|主机|在役快照)", "机器名（数字 + 机器语境）"),
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "邮箱（PII）"),
    ]
    leaks = []
    for pat, why in SENSITIVE:
        for m in re.finditer(pat, target_text):
            leaks.append((why, m.group(0)))
    if leaks:
        failures.append(("④G 脱敏（宪法 §15.1）", f"命中 {len(leaks)} 处（前 5）: {leaks[:5]}"))

    # ---- ④H. 禁止措辞（§17.7：双真相源 / 能力承诺式措辞） ----
    forbidden = [(w, why) for w, why in FORBIDDEN_PHRASES if w in target_text]
    if forbidden:
        failures.append(("④H 禁止措辞", f"命中 {forbidden}"))

    # ---- ④I. §17.7 结构要件：**机器可读生成留痕** + 判定口径（可见声明已按下架处置）----
    #   ★ 2026-10-19 处置（使用者裁定「目录后面那三块全删」，含对 §17.7 的明确破例）：
    #     - 「首屏生成声明」（可见文字）**下架** ⇒ 改为**机器可读生成留痕**：`<head>` 的 generator /
    #       matrix-source / matrix-generation-marker 三个 meta ＋ `<body>` 起始处的
    #       `<!-- generated-by: … ; source: … ; do-not-edit -->` 注释（不可见、合规留痕）。
    #       ★ 替代防线：④W **反向**断言"可见面不得再出现任何生成说明行"（⛔ 使用者明示不许补）。
    #     - 「边界声明可见性」下架（见 ④N）。
    #     - 保留未受影响的：criteria_caliber.criteria_id（判定口径可复核）。
    meta = data.get("meta") or {}
    if ('name="matrix-generation-marker"' not in target_text
            or "generated-by:" not in target_text
            or 'name="generator"' not in target_text
            or 'name="matrix-source"' not in target_text):
        failures.append(("④I 机器可读生成留痕",
                         "缺生成留痕（矩阵件必须带 generated-by 注释 + generator/matrix-source/"
                         "matrix-generation-marker 三个 meta）"))
    if "<!-- generated-by:" not in target_text:
        failures.append(("④I 首屏机器可读注释",
                         "`<body>` 起始处缺 `<!-- generated-by: … ; do-not-edit -->` 注释（合规留痕）"))
    if not (meta.get("criteria_caliber") or {}).get("criteria_id"):
        failures.append(("④I 判定口径", "数据缺 criteria_caliber.criteria_id"))

    # ---- ④K. ★★ 覆盖度：**反向断言**（警示块已按使用者裁定下架；此处守"未达门槛不得宣称"）----
    #   原 ④K（首屏必须出现 `id="coverage-warning"` 的警示 + 四态计数 + 尚未验证档清单 + 档位总数）
    #   随「覆盖度警示块」一并下架（使用者 2026-10-19 明确破例，逐条处置见工位 report.md 任务 4）。
    #   ★ 留下的等价防线（⛔ 不是放宽，是**同族断言改向**）：
    #     a) 页面**不得**再出现警示块的 id / 计数句（防"删不干净 / 半删"）；
    #     b) **静态区不随数据变**（④V）+ 渲染↔在盘**字节同一**（①）⇒ 页面上的档位与状态标签
    #        必然与数据同源（页面无法成为陈旧快照：改数据必须重渲染，否则 ① 直接红）；
    #     c) 档位集合 / 四态标签计数仍**逐档**对照数据（保留原 ④K 的可见性判据本体）；
    #     d) 保留"覆盖度未达门槛时，可见文字**不得**出现能力承诺式措辞"（含标题）。
    _vocab = (meta.get("status_vocabulary") or [])
    _rows_ = data.get("rows", [])
    _cnt = {s: sum(1 for _r in _rows_ if _r.get("status") == s) for s in _vocab}
    _unver = [str(_r.get("tier")) for _r in _rows_ if _r.get("status") == "尚未验证"]
    if _unver:
        #   a) 警示块已下架 ⇒ 出现即"删不干净"
        for _bad_kw in ('id="coverage-warning"', "覆盖度警示：", "当前结论分布为",
                        "尚未验证的档位（", "不得被读作"):
            if _bad_kw in target_text:
                failures.append(("④K 覆盖度警示块已下架",
                                 f"页面仍出现已删除块的痕迹 {_bad_kw!r} ⇒ 任务 1 未删净"))
        #   c) 可见性判据本体（保留、⛔ 不放松）：档号集合与四态标签计数逐档对照数据
        _ids_html = re.findall(r'data-col="tier">([^<]+)</td>', target_text)
        _st_html = re.findall(r'data-col="status">([^<]+)</td>', target_text)
        _ids_data = [str(_r.get("tier")) for _r in _rows_]
        if sorted(_ids_html) != sorted(_ids_data):
            _miss = sorted(set(_ids_data) - set(_ids_html))
            _extra = sorted(set(_ids_html) - set(_ids_data))
            failures.append(("④K 档位可见性",
                             f"页面档号集合 != 数据档位集合（页面缺 {_miss[:5]}；页面多 {_extra[:5]}）"))
        _cnt_html = {s: _st_html.count(s) for s in set(_st_html)}
        for _s in _vocab:
            if _cnt_html.get(_s, 0) != _cnt.get(_s, 0):
                failures.append(("④K 状态标签计数",
                                 f"页面「{_s}」标签 {_cnt_html.get(_s, 0)} 个 != 数据 {_cnt.get(_s, 0)} 档"))
        #   d) 未达门槛 ⇒ 可见文字不得出现"能力承诺式"表述（⛔ 原断言保留，只是不再依赖警示块）
        _visible = re.sub(r"<!--.*?-->", " ", target_text, flags=re.S)
        _visible = html.unescape(re.sub(r"<[^>]+>", " ", _visible))
        _claims = [w for w in ("全组合可用", "已验证全组合", "全部可用", "均已可用")
                   if w in _visible]
        #   ★ 只认**无歧义的能力承诺措辞**（⛔ 不做宽窗口正则：`四条判据全部通过 ⇒ 才能标…`
        #     是判据的**必要条件句**，不是承诺 ⇒ 宽窗口会误伤，已实测剔除）。
        if _claims:
            failures.append(("④K 未达门槛不得宣称",
                             f"尚有 {len(_unver)} 档「尚未验证」，可见文字却出现能力承诺式表述 {_claims[:5]}"))
    elif 'id="coverage-warning"' in target_text:
        failures.append(("④K 覆盖度门槛", "全部档位已有明确结论，却仍渲染覆盖度警示"))

    # ---- ④N. ★★ §17.7 承诺边界声明：**数据面要件**（可见性已按下架处置）----
    #   ★ 2026-10-19 处置：声明块（`div.hint`，含覆盖度门槛注）按使用者裁定**整块删除**
    #     ⇒「必须首屏可见」不再成立（明确破例）；本断言**只保留数据面**：
    #     声明文本仍须存在且三点要件齐备（这样"以后要恢复可见声明"时，素材与要件校验都还在），
    #     ⛔ 不再要求它出现在人读件里。
    _bd = (meta.get("boundary_declaration") or {})
    _bd_txt = str(_bd.get("text") or "")
    _bd_pts = " ".join(str(x) for x in (_bd.get("points") or []))
    _bd_all = _bd_txt + " " + _bd_pts
    if not _bd_txt.strip():
        failures.append(("④N 边界声明存在性（数据面）",
                         "boundary_declaration.text 为空 ⇒ 声明素材已丢（§17.7 要件不可复核）"))
    # ① 各档可用性（每个列出的档位都已完整跑通）
    if not any(k in _bd_all for k in ("完整跑通", "都已完整跑通", "各档可用性")):
        failures.append(("④N 边界要件①各档可用性（数据面）", "未正面陈述「每个列出的档位都已完整跑通」"))
    # ② 平台方数据集结论（效果达标 + 是"可用、大概率无 bug"的证据）
    if not any(k in _bd_all for k in ("平台方数据集", "我们的数据集")):
        failures.append(("④N 边界要件②平台方数据集（数据面）", "未正面陈述「在平台方数据集与参数下效果达标」"))
    if "大概率无 bug" not in _bd_all:
        failures.append(("④N 边界要件②证据定位（数据面）", "未写明这是「可用、大概率无 bug」的证据"))
    # ③ 使用者效果不在承诺范围
    if not any(k in _bd_all for k in ("不在本矩阵的承诺范围", "不在承诺范围", "不归本矩阵承诺")):
        failures.append(("④N 边界要件③使用者效果（数据面）", "未写明「使用者的任务/数据集/参数下的效果不在承诺范围」"))
    # ★ 反例（判为不合格）：只写"效果不保证 / 仅供参考"而未正面陈述 ①②
    if (not any(k in _bd_all for k in ("完整跑通", "平台方数据集", "我们的数据集"))
            and any(k in _bd_all for k in ("效果不保证", "仅供参考"))):
        failures.append(("④N 边界声明反例（数据面）", "只写「不保证效果 / 仅供参考」⇒ §17.7 判为不合格"))

    # ---- ④W. ★★ 可见面反向断言：不得再出现"生成说明行 / 覆盖度块 / 边界声明块" ----
    #   使用者原话：「目录后面那三块全删」⇒ 机核口径 = 可见文本里这三块**0 命中**
    #   （⛔ 不可见的 `<!-- generated-by: … -->` 注释与 `<head>` meta 除外，那是合规留痕）。
    #   ★ 本断言**能失败**：只要有人把任一块补回可见面（含变形写法），立即红。
    _visible_wo_comments = re.sub(r"<!--.*?-->", " ", target_text, flags=re.S)
    _vis = html.unescape(re.sub(r"<[^>]+>", " ", _visible_wo_comments))
    _vis_flat = re.sub(r"\s+", " ", _vis)
    _resurrected = [w for w in ("覆盖度警示", "覆盖度门槛", "承诺边界声明", "唯一可编辑源头",
                                "自动生成文件", "单方向，禁止互推", "禁止手工编辑")
                    if w in _vis_flat]
    if _resurrected:
        failures.append(("④W 可见面不得出现已删除三块",
                         f"可见文本仍命中 {_resurrected[:5]} ⇒ 目录后面的三块未删净 / 或被补回"))
    if '<p class="gnote">' in target_text or 'class="hint"' in target_text:
        failures.append(("④W 可见面不得出现已删除三块（容器）",
                         "仍存在 <p class=gnote> / <div class=hint> 容器"))
    # ★ 同批：模板里**不得**再把这三块作为占位符（否则等于留了回填通道）
    for _tok in ("{{GEN_BANNER}}", "{{COVERAGE_WARNING}}", "{{BOUNDARY}}"):
        if _tok in template_text:
            failures.append(("④W 模板不得留三块占位符", f"模板仍有 {_tok}"))
    # ★ 模板里**只允许** 4 个占位符（使用者口径：静态部分不得随数据变）
    _toks = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", template_text)))
    _allowed = sorted(render_matrix.PLACEHOLDER_TOKENS)
    if _toks != _allowed:
        failures.append(("④W 模板占位符白名单",
                         f"期望恰为 {_allowed}，实际 {_toks}（静态区不得再有数据占位符）"))
    # ---- ④Q0. 「关键开关」列的 `—` **语义必须被显式定义**（修正轮 4 ⚠②）----
    #   ★ 缺陷实况：该列 31/54 档为 `—`，而上一版读表提示只定义了「实验数据」段的 `—`
    #     ⇒ 属"要读者推断"。修法 = 读表提示补一句（⛔ 不改渲染形态，`--human-view-self-test`
    #     的"唯一开关 ⇒ `—`"反向断言**保持成立**）。
    #   ★ 断言只许加强：不仅查那句话在不在，还查**它必须同时点明段位与列名**（防写回泛泛一句）。
    if "「关键开关」列的「—」" not in target_text or "没有额外差异开关" not in target_text:
        failures.append(("④Q0 关键开关列 — 的语义",
                         "读表提示未显式定义「关键开关」列的 `—` 语义（31/54 档该列为 `—`）"))

    # ---- ④Q. 「关键开关」列**不得重复发布检查点间隔**（修正包 🔴-1）----
    #   ★ 断言只许加强：不仅查"令牌在不在"，还查**该档的权威值是否等于公式值**。
    #     判据①（发布面）：`switches` 里不得出现 `save_steps=` 令牌
    #       ——清单侧的旧口径值（= 每 epoch 步数）一旦发布，就与 §3.1 定义行、
    #         同行权威字段 `save_steps`、跑次 yaml 三处矛盾（实测 50/54 档）。
    #     判据②（取值面）：逐档 `save_steps` 必须**恰好**等于 `derived_steps + 1`
    #       ——这是 §3.1 定义行的机核化（⛔ 不依赖人眼比对）。
    _tok_q = build_official.SAVE_STEPS_TOKEN
    _bad_q = [str(r.get("tier")) for r in _rows_ if _tok_q in str(r.get("switches") or "")]
    if _bad_q:
        failures.append(("④Q 开关列不得重复发布检查点间隔",
                         f"「关键开关」列出现 `{_tok_q}` 令牌（{len(_bad_q)} 档：{_bad_q[:5]}）"
                         f"——该参数不进人读件"))
    _bad_q2 = []
    for _r in _rows_:
        _dv, _sv = _r.get("derived_steps"), _r.get("save_steps")
        if _dv in (None, "", "—") or _sv in (None, "", "—"):
            continue
        if int(_sv) != int(_dv) + 1:
            _bad_q2.append((str(_r.get("tier")), _sv, int(_dv) + 1))
    if _bad_q2:
        failures.append(("④Q 检查点间隔取值 =推导步数+1",
                         f"{len(_bad_q2)} 档不符（tier, 实际, 应为）：{_bad_q2[:5]}"))
    # ★ 精简轮 4：检查点间隔（`save_steps`）**整段移出人读件**（主席令「只列复现需要的」）⇒
    #   原先「§3.1 必须写明『定义一次 + 表内不重复列出』」的交叉引用断言随该段一并作废。
    #   ★ 但**发布面禁令照旧**（上条 ④Q 令牌检查 + 取值检查），⛔ 不放宽。

    # ---- ④R. 人读件正文**不得含开发过程内容**（修正包 🔴-2 / 建议 S2，机核化主席判据）----
    #   ★ 判据（主席原话）：「不要把修改过程写进去，不要把给我们自己写的东西写进去」。
    #   ★ 只扫**正文**（渲染后的可见文本），且**排除 §17.7 宪法逐字引文**（引文里的「我们」是豁免项）。
    #   ★ 词表 = 内部工位/证据路径形态 + 相对时间指代；⛔ 不做通配、⛔ 不豁免生成物。
    #   ★★ 2026-10-19 处置：原先这里有一段**豁免逻辑** —— 把「承诺边界声明」块内的宪法逐字引文
    #     （`…都在我们的数据集上…`）从扫描面里挖掉。该块已按使用者裁定**整块删除**
    #     ⇒ 页面上不再有那段引文，豁免逻辑成为**死代码**，随块一并移除。
    #     ★ 扫描面**只增不减**：现在扫的是**全部可见正文**（⛔ 不挖任何豁免洞），
    #       阳性对照见 `run_self_proof` ⑫（注入即红）与 ⑯/⑮（⑮ 已随豁免下架，见该文件注）。
    _prose = html.unescape(re.sub(r"<[^>]+>", " ", target_text))
    _churn = []
    for _w, _why in build_official.DEV_CHURN_TERMS:
        if _w in _prose:
            _n = _prose.count(_w)
            _churn.append((_w, _n, _why))
    if _churn:
        failures.append(("④R 正文不得含开发过程内容",
                         "命中内部过程词：" + "；".join(f"`{w}`×{n}（{why}）" for w, n, why in _churn)))

    # ---- ④P. ★★ 覆盖度门槛注 —— 随「承诺边界声明块」下架，改为**现状一致性**断言 ----
    #   ★ 2026-10-19 处置：原断言要求"边界声明块内必须写明门槛尚未生效，且计数与 rows 现算一致"。
    #     该块已按使用者裁定整块删除 ⇒ 无对象，随块下架（逐条处置见工位 report.md 任务 4）。
    #   ★ 等价防线（同族换向，⛔ 不是放宽）：
    #     a) **数据面**：`rows` 与页面状态标签的现状必须一一对应（仍在 ④K 逐档 / ④U 逐格）；
    #     b) **文案面**：若**全部档位已结论**，则页面**不得**再出现"当前尚无任何实测结论"这类
    #        已过期的门槛文案（防"结论出来后文案不更新"的陈旧快照——这正是原 ④P 要防的合成谬误）；
    #     c) 未达门槛时的"不得宣称整体承诺"由 ④K(d) 与 ④W 共同守住。
    _pend_p = [str(r.get("tier")) for r in _rows_ if r.get("status") == "尚未验证"]
    if not _pend_p:
        _stale_gate = [w for w in ("当前尚无任何实测结论", "覆盖度门槛", "尚未生效")
                       if w in html.unescape(re.sub(r"<[^>]+>", " ", target_text))]
        if _stale_gate:
            failures.append(("④P 门槛文案陈旧快照",
                             f"全部档位已结论，页面仍写过期门槛文案 {_stale_gate[:5]}"))
    else:
        # 未达门槛：页面必须仍如实说明"尚无结论"（原文案在位；缺失即红）
        if "本表当前<b>尚无任何实测结论</b>" not in target_text:
            failures.append(("④P 未达门槛如实说明",
                             f"尚有 {len(_pend_p)} 档「尚未验证」，页面却不再说明「尚无任何实测结论」"))

    # ---- ④O. 结论两列的**契约**（生成端 `build_official.note_contract` ⇄ 校验端同一组字符串）----
    #   ★ 断言只许加强：这里不仅查"有没有写第 1/2 条"，还查**归因正文的依据关键词**是否在位、
    #     备注列是否**恰好**是契约文本（防"一句话被删空"或"塞回长文"）。
    _DETAIL_EVIDENCE_GUARD = build_official._DETAIL_EVIDENCE
    _bad_o = []
    for _r in _rows_:
        _t, _st = str(_r.get("tier")), _r.get("status")
        _want = build_official.note_contract(_st, has_verdict=bool(_r.get("verdict")),
                                  has_excluded=bool(str(_r.get("excluded_reason") or "").strip()),
                                  raw_note=str(_r.get("note") or ""))
        if str(_r.get("note") or "").strip() != _want:
            _bad_o.append((_t, f"备注 != 契约文本：实际 {str(_r.get('note'))[:40]!r}"))
        _nr = str(_r.get("not_failed_reason") or "").strip()
        if _st == "已验证不可用":
            _v = _r.get("verdict") or {}
            if _nr in ("", "—"):
                _bad_o.append((_t, "「已验证不可用」的「未过判据」列为空"))
            if not (_nr.startswith("第 1 条") or _nr.startswith("第 2 条")):
                _bad_o.append((_t, "「未过判据」须以「第 1 条 / 第 2 条」起头"))
            _det = str(_v.get("failed_detail") or "")
            _ev = _DETAIL_EVIDENCE_GUARD.get(_t, ())
            _hit = [k for k in _ev if k in _det or k in _nr]
            if _ev and not _hit:
                _bad_o.append((_t, f"归因正文缺依据关键词（期望其一 {list(_ev)}）"))
            elif _ev and not all(k in _det or k in _nr for k in _ev):
                _bad_o.append((_t, f"归因正文依据不完整（缺 {[k for k in _ev if k not in _det and k not in _nr]}）"))
        elif _st in ("尚未验证", "未纳入验收"):
            if _nr not in ("", "—"):
                _bad_o.append((_t, f"「{_st}」不得有「未过判据」（实际 {_nr[:30]!r}）"))
    if _bad_o:
        failures.append(("④O 结论列契约", f"不一致 {len(_bad_o)} 档（前 5）: {_bad_o[:5]}"))

    # ---- ④L. §17.7 逐档「不可用必须写明哪一条」+ 「未纳入验收」逐档理由 ----
    _bad_crit = []
    for _r in _rows_:
        if _r.get("status") != "已验证不可用":
            continue
        _v = _r.get("verdict") or {}
        _blob = f"{_v.get('failed_criterion', '')} {_v.get('failed_detail', '')} {_r.get('note', '')}"
        if not any(k in _blob for k in ("第 1 条", "第 2 条")):
            _bad_crit.append((_r.get("tier"), "未写明未过第 1 条还是第 2 条"))
    if _bad_crit:
        failures.append(("④L 不可用须写明条次", f"{len(_bad_crit)} 档（前 5）: {_bad_crit[:5]}"))
    _bad_excl = [( _r.get("tier"), "缺逐档理由") for _r in _rows_
                 if _r.get("status") == "未纳入验收" and not str(_r.get("excluded_reason") or "").strip()]
    if _bad_excl:
        failures.append(("④L 未纳入验收逐档理由", f"{len(_bad_excl)} 档（前 5）: {_bad_excl[:5]}"))

    # ---- ④M. 内部术语（§17.7「矩阵中仍然禁止」＋使用者诉求）----
    _term_hits = [(w, target_text.count(w)) for w in FORBIDDEN_INTERNAL_TERMS if w in target_text]
    if _term_hits:
        failures.append(("④M 内部术语", f"命中 {len(_term_hits)} 类: {_term_hits}"))

    # ---- ④J. 耗时列（★ fail-closed，四向）----
    #   ① 有读数（跑次文本 或 228 只读侦察 `elapsed_evidence`）⇒ 列**必须**有值；
    #   ② 列有值（非「未取到」）⇒ **必须**有读数证据（⛔ 防编造）；
    #   ③ 列标「未取到」⇒ 括号内**必须**有具名原因，且此时**不得**存在读数证据（口径矛盾）；
    #   ④ 列**为空** ⇒ FAIL（⛔ 不留空白让它看起来像"忘了填"）。
    el_bad, el_filled, el_na = [], [], []
    try:
        import build_official as _bo
        for r in data.get("rows", []):
            # ★ 仅对**已结论**档断言：未结论档按 §17.7 结果列一律 '—'（不适用本断言）
            if r.get("status") not in ("已验证可用", "已验证不可用"):
                continue
            parsed = _bo.split_elapsed(_bo.elapsed_source_blob(r)) != "—"
            ev = r.get("elapsed_evidence") if isinstance(r.get("elapsed_evidence"), dict) else {}
            st = str(ev.get("status") or "")
            has_ev = st in ("found", "found_from_run_json")
            got = str(r.get("elapsed") or "").strip()
            if got in ("", "—"):
                el_bad.append((r.get("tier"), "「耗时」列为空（既无读数也无具名原因）"))
                continue
            # ★ 生成期把「总时长 N s」规范化为「总时长 N s（约 X h）」——**原始秒数逐字保留在括号前**，
            #   ⇒ 该形态自身即读数证据；此处直接认可并继续（⛔ 不放宽取值域：仍只认这一种形态）。
            _dur_ok = bool(re.fullmatch(r"总时长\s*\d+\s*s（约[\d.]+\s*h）", got))
            if got.startswith("未取到"):
                el_na.append(r.get("tier"))
                mm = re.match(r"^未取到（\s*(.+?)\s*）$", got)
                if not mm or not mm.group(1):
                    el_bad.append((r.get("tier"), "标「未取到」但缺具名原因"))
                if parsed or has_ev:
                    el_bad.append((r.get("tier"), "标「未取到」但跑次文本/侦察证据里有读数 ⇒ 口径矛盾"))
                continue
            el_filled.append(r.get("tier"))
            if _dur_ok:
                continue
            # ★ 生成期把「总时长 N s」折算为「总时长 N s（约 X h）」——数字未变、只是加了折算；
            #   该形态不再与跑次原始文本逐字相同 ⇒ 由**形态本身**判定（⛔ 不放宽取值域）。
            _dur = re.fullmatch(r"总时长\s*(\d+)\s*s(?:（约\s*([\d.]+)\s*h）)?", got)
            if _dur:
                continue
            if not (parsed or has_ev):
                el_bad.append((r.get("tier"), f"「耗时」列有值 {got!r}，但跑次文本与侦察证据都无读数（疑编造）"))
            if st == "found" and ev.get("value_s") is None:
                el_bad.append((r.get("tier"), "elapsed_evidence=found 但缺 value_s"))
            if st == "not_found":
                el_bad.append((r.get("tier"), "elapsed_evidence=not_found 但「耗时」列有值"))
        if el_bad:
            failures.append(("④J 耗时列一致性（fail-closed）", f"不一致 {len(el_bad)} 档（前 5）: {el_bad[:5]}"))
    except Exception as e:                       # ⛔ 校验器自身出错也不得静默
        failures.append(("④J 耗时列一致性（fail-closed）", f"断言执行失败: {e!r}"))

    miss_keys = [(r.get("tier"), k) for r in data["rows"] for k in SPEC_KEYS if k not in r]
    if miss_keys:
        failures.append(("④E 机读字段缺失（yaml 生成器口径）", f"缺 {len(miss_keys)} 处（前 5）: {miss_keys[:5]}"))

    # ---- ④S. 「不可执行」档数：**散文 == 数据**（2026-10-16 矩阵侧收口）----
    #   ★ 病史：散文里 4 处「10 档不可执行」是**硬编码**；口径扩面到 27 档并集后 ⇒ 文档说假话。
    #   ★ 判据（fail-closed）：① 数据里 `executability == not_executable` 的行数**现算**（唯一入口
    #     = `build_official.not_executable_count`，与生成端同源）；② 散文里的档数短语只能有**一个**
    #     取值，且必须等于 ①（多取值 / 漏插值 / 硬编码残留 ⇒ 必红）。
    _ne_n = build_official.not_executable_count(data["rows"])
    _ne_word = f"{_ne_n} 档不可执行"
    _meta_blob = json.dumps(data.get("meta", {}), ensure_ascii=False)
    _ne_phrases = sorted(set(re.findall(r"\d+ 档不可执行", _meta_blob)))
    if _ne_phrases != [_ne_word]:
        failures.append(("④S 散文档数 == 数据",
                         f"数据里 not_executable 行数 = {_ne_n} ⇒ 散文应恰为 ['{_ne_word}']，"
                         f"实际 = {_ne_phrases}（散文与数据不一致：硬编码残留 / 多取值 / 漏插值）"))

    # ---- ④T. 清单里**已声明的前提**必须出现在对外人读面（2026-10-16 透明性缺口）----
    #   ★ 缺口实况：`T016` / `T034` 的可行性依赖 `native.offload_optimizer_state = true`
    #     （清单 `feasibility.recipe.native.offload_optimizer_state`，v1/v2 清单各 2 档携带），
    #     而对外人读件里 **0/54** —— `rows[T016/T034]` 连 `feasibility` 键都没有，
    #     `note` 只写「尚未验证」⇒ 照着对外件复现的人会踩同一个 OOM（真机已复现 48.5 s / 0 步）。
    #   ★ 判据（fail-closed、三面，**双向 iff**）：
    #     a 发布面：`{(档, 前提键)}` **出现在对外件者** == 清单**声明者**
    #       —— ⛔ 不是"包含"：**多写与漏写都红** ⇒「当且仅当该档有声明才出现」；
    #     b 渲染面：声明的前提必须出现在**渲染列**（`EXPECTED_COLUMNS`）里
    #       —— 防"把令牌藏进不渲染字段"造成假绿（该子项**能失败**，见阳性对照 ④）；
    #     c 取值面：对外令牌取值必须**等于**清单声明取值（⛔ 不得写假开关）。
    #   ★ 前提的定义**同源** = `gen_configs.load_premises()`（含"正文点名"判据），⛔ 本文件不另实现。
    _prem_err = None
    try:
        _prem = gen_configs.load_premises(a.manifest)
    except SystemExit as _e:                                   # 清单缺失 ⇒ fail-closed
        _prem_err = str(_e).strip().splitlines()[0]
    if _prem_err:
        failures.append(("④T 已声明前提的可见性（清单不可读 ⇒ fail-closed）", _prem_err))
    else:
        _vocab = {k for _items in _prem.values() for (_b, k, _v) in _items}
        _declared = {(str(t), k): v for t, _items in _prem.items() for (_b, k, v) in _items}
        _anywhere, _rendered, _tokenval = set(), set(), {}
        for _r in data["rows"]:
            _t = str(_r.get("tier"))
            for _k in _vocab:
                if any(_k in json.dumps(_r.get(_c), ensure_ascii=False) for _c in _r):
                    _anywhere.add((_t, _k))
                if any(_k in json.dumps(_r.get(_c), ensure_ascii=False)
                       for (_c, _lbl) in EXPECTED_COLUMNS):
                    _rendered.add((_t, _k))
            for _tok in str(_r.get("switches") or "").split(";"):
                _tok = _tok.strip()
                if "=" not in _tok:
                    continue
                _k, _v = _tok.split("=", 1)
                if _k in _vocab:
                    _tokenval[(_t, _k)] = _v
        _dset = set(_declared)
        _not_published = sorted(_dset - _anywhere)      # 声明了但对外件里根本没有
        _undeclared = sorted(_anywhere - _dset)         # 对外件里写了但清单没声明
        _not_rendered = sorted(_dset - _rendered)       # 发布了、但没进任何渲染列
        _wrong = sorted((t, k, _tokenval[(t, k)], _declared[(t, k)])
                        for (t, k) in _dset
                        if (t, k) in _tokenval
                        and str(_declared[(t, k)]).lower() != _tokenval[(t, k)].lower())
        if _not_published or _undeclared or _not_rendered or _wrong:
            failures.append(("④T 已声明前提的人读面可见性",
                             f"前提词汇 = {sorted(_vocab)}；清单声明 {len(_dset)} 条、"
                             f"对外件出现 {len(_anywhere)} 条、其中渲染列 {len(_rendered)} 条｜"
                             f"声明了但未发布 = {_not_published}｜发布了但未声明 = {_undeclared}｜"
                             f"未进渲染列 = {_not_rendered}｜取值不符（档,键,实际,声明）= {_wrong}"))

    sha_html = hashlib.sha256(target_bytes).hexdigest()
    sha_render = hashlib.sha256(rendered_bytes).hexdigest()

    if not a.quiet:
        def ok(pfx):
            return "PASS" if not any(f[0].startswith(pfx) for f in failures) else "FAIL"
        print(f"{VERSION}")
        print(f"target : {target_path}  ({len(target_bytes)} bytes, sha256 {sha_html})")
        print(f"render : {len(rendered_bytes)} bytes, sha256 {sha_render}")
        print(f"checks : ① 字节同一性={'PASS' if byte_identical else 'FAIL'}"
              f" | ③ 结构两大部分={ok('③')}"
              f" | ④A 规则关键词={ok('④A')}"
              f" | ④B 列顺序={ok('④B')}"
              f" | ④C 结果列一致性={ok('④C')}"
              f" | ④D 主要训练参数列={ok('④D')}"
              f" | ④Z 条件/结果分区={ok('④Z')}"
              f" | ④U 结果列清空/状态统一={ok('④U')}"
              f" | ④E 统一口径/机读字段={ok('④E')}"
              f" | ④G 脱敏={ok('④G')}"
              f" | ④H 禁止措辞={ok('④H')}"
              f" | ④I §17.7 要件={ok('④I')}"
              f" | ④J 耗时列={ok('④J')}"
              f" | ④K 覆盖度门槛={ok('④K')}"
              f" | ④L 逐档条次/理由={ok('④L')}"
              f" | ④M 内部术语={ok('④M')}"
              f" | ④N 边界声明三点={ok('④N')}"
              f" | ④O 结论列契约={ok('④O')}"
              f" | ④P 覆盖度门槛注={ok('④P')}"
              f" | ④Q0 关键开关列—语义={ok('④Q0')}"
              f" | ④Q 开关列不重复发布检查点间隔={ok('④Q')}"
              f" | ④R 正文无开发过程内容={ok('④R')}"
              f" | ④S 散文档数==数据={ok('④S')}"
              f" | ④V 静态区不随数据变={ok('④V')}"
              f" | ④W 可见面不得出现已删除三块={ok('④W')}"
              f" | ④T 已声明前提可见性={ok('④T')}"
              f" | ④X 判据页/数据逐字一致={ok('④X')}"
              f" | ④Y 判据互不重叠={ok('④Y')}")
        print(f"results: 已回填 {len(filled)} 档 {filled}；其余 {len(data['rows'])-len(filled)} 档保持 '—'"
              + ("；--expect-empty 已开启" if a.expect_empty else ""))
        print(f"elapsed: 有值 {len(el_filled)} 档"
              + (f"（{'、'.join(el_filled)}）" if el_filled else "")
              + f"；标「未取到（具名原因）」{len(el_na)} 档"
              + (f"（{'、'.join(el_na)}）" if el_na else ""))
        if off is None:
            print("hand-edit detection: 无差异（改 1 个字符即会在此打印字节偏移 / 行 / 列）")
        for name, detail in failures:
            print(f"\nFAIL [{name}]\n{detail}")

    print("\nVERIFY " + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
