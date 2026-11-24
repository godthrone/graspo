"""tool call 标注：期望 mark 序列逐字符比对（非正则）。

流程（对应方案 §6）：
1. think 标签处理：``<think>`` 与 ``</think>`` 标记 → S，内容 → T
2. 构建期望 mark 序列（来自 targets + 模板常量）：
   ``<tool_call> <function=NAME> { <parameter=NAME> </parameter> }* </function> </tool_call>``
3. 逐 mark 与模型输出**逐字符比对**：
   - 字符相等 → S（含结构标签后的换行，数据集约定 ``<tool_call>\\n`` 整体 S）
   - 首个不相等字符 → E，其后全部 D（严格对齐截断）
4. 结构不完整（截断/缺闭合）：已有字符全部正确 → 全 S，不标 E
   （"结构不完整"信号由 reward 层表达，避免误标正确 token）

关键设计（v0.20.0 修订）：
- **按字符比对，不按 XML 元素整体标 E**：拼错标签（``</parametr>``）时，
  与正确标签共享的前缀字符（``</paramet``）标 S，仅首个不匹配字符（``r``）标 E
- tokenizer 无关：标注定义在字符级，token 由 offset_mapping 派生，
  任何 tokenizer 都不会把纯正确前缀字符标成 E

## ★★ P1（2026-11-18）：工具名（``<function=NAME>`` 的 NAME）是一个**可评分字段**

改前缺陷（真实 rollout 实测）：工具名整段走逐字符比对 ⇒ 只有**首个不匹配字符**标 E(−1.0)，
其后全 D，而名字**前面**的 22 个结构 token（``<tool_call>`` + 换行 + ``<function=``）都是
+1.0 ⇒ 工具名写错的 completion **净得 +21（正的）**。后果：把工具名叫错反而拿正梯度。

修法：NAME 段不再当"结构字符"逐个比对，而按**身份字段**处理（工具名对就整段对、错就整段
错，没有"写对前 8 个字母"的部分分）：

- NAME 与 GT 一致 ⇒ 整段标 ``VALUE`` + ``field=TOOL_NAME_FIELD`` ⇒ ``field_score`` 判满分
  ⇒ 该段每个 token 拿 **+1.0**（写对就是正分）。**不是** S——理由见下面 B1（2026-11-24）
- NAME 写错 ⇒ 从**首个不匹配字符**起整段标 ``VALUE`` + ``field=TOOL_NAME_FIELD``
  ⇒ :func:`graspo.ripple.annotation.advantages.field_score` 判 0 ⇒ 该段每个 token 拿 −1.0；
  随后 ``drop_from``（「第一个错误之后扔掉」——工具名错时后面的参数没有训练意义）
- NAME 比 GT 短（模型在"名字该继续"处写了 ``>``）⇒ 该处标 E + drop（残缺名字，
  与参数残缺标签同一语义）
- **名字后缺 ``>``**（名字段里撞见 ``<``，说明找到的 ``>`` 属于后面的标签）⇒ 同上一条：
  E 标在「应出现 ``>`` 的位置」（即那个 ``<`` 处）+ drop，**不产生 V 段**

最后一条是 B1 修复（2026-11-18 复审）：``_line_local_gt`` 只保证 ``>`` 与本行同名，
``<function=rotate_arm<parameter=angle_deg>`` 会拿到 ``angle_deg>`` 的 ``>`` ⇒
名字段成了 ``rotate_arm<parameter=angle_deg``，被当成"写错的值"整段 −1（实测一条 28 个 −1、
Σadv −4.0），把**写对的工具名**与后面的结构字符一起罚进负分。判据「名字段含 ``<`` ⇒ 缺 ``>``」
把它拉回既有口径。GT 名自身含 ``<`` 的极端情况不受影响（先比 ``model_name == gt_name``，
相等即判对）：形如 ``<function=a<b>`` 且 GT 名就是 ``a<b`` 时仍是整段 S。

⛔ 不动「第一个错误之后 drop」这条原则本身（参数/闭合标签拼错仍走它）；
⛔ 工具名写错**不产生 E 标签**——保住既有跨用例不变式「E 之后必须全部为 D」。

## ★★ B1 修复（2026-11-24 复审）：写对的工具名也标 V —— 工具名是**值**，不是结构

负责人定的划分口径是「**按模型是选错了还是写错了**分」：**写错 = 结构**（JSON/XML 语法、
括号、引号、字段名拼写、参数名）｜**选错 = 值**（**工具名**、参数值）。

P1 只改了「名字写错 ⇒ V」，名字**写对**时仍是逐字符 S ⇒ 工具名进了**结构**域。于是
``advantages.all_correct_structure_chars``（P3：``k = 组内各条 S 字符数的最小值``，前 k 个
S 字符的 advantage 置 0）会伸进工具名内部，把**写对的工具名**也 mask 掉。反例（一组两条）：
好那条名字全对（20 个字符全 S）、坏那条写成 ``robot_atomic_controX``（S 字符 = 22 + 19）
⇒ ``k = 41`` ⇒ 好那条的**工具名前 19 个字符** advantage 变成 0.0。

修法：名字写对时整段标 ``VALUE`` + ``field=TOOL_NAME_FIELD``（与写错那支**标签层面完全
同构**，只是取值 1.0 而非 0.0）⇒ 名字字符不再计入 S 字符数 ⇒ 不可能落进 P3 的「组内全对
结构前缀」。**这不是给工具名打折**：S 与 V(raw=1.0) 的 advantage 都是 ``+1.0``，除 P3 mask
之外任何读数都不变。

⛔ 工具名前后的格式符（``<function=`` / ``>``）仍是结构 S——它们是语法，不是选择。
"""

from dataclasses import dataclass
from typing import Any

from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.parsing.qwen_tool_parser import (
    FUNCTION_CLOSE,
    FUNCTION_OPEN,
    PARAMETER_CLOSE,
    PARAMETER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
)

# 标签格式常量单一真相源（2026-08-07 方案 A）：从模型族 parser
# （qwen_tool_parser）读取，避免标注器与解析器两套格式理解漂移；
# 新模型族 = 新 parser 族 = 标注器自动获得其格式。
_OPEN_TAG = TOOL_CALL_OPEN
_OPEN_TAG_LEN = len(_OPEN_TAG)

_THINK_OPEN = THINK_OPEN
_THINK_CLOSE = THINK_CLOSE

_FUNCTION_MARK = FUNCTION_OPEN
_PARAMETER_MARK = PARAMETER_OPEN
_FUNCTION_CLOSE = FUNCTION_CLOSE
_PARAMETER_CLOSE = PARAMETER_CLOSE
_TOOL_CALL_CLOSE = TOOL_CALL_CLOSE

#: 工具名字段的 field 名（P1）。``<function=NAME>`` 的 NAME 是可评分的身份字段，
#: ``advantages.gt_value_for_field`` 见到它时取 ``tool_calls[0].name`` 作 GT。
#: 双下划线包裹的保留名：参数名来自数据集，不会以 ``__`` 开头，故不会与真实字段撞车。
TOOL_NAME_FIELD = "__tool_name__"


def _blank(text: str, start: int, end: int) -> bool:
    """[start, end) 区间是否全为空白（\\n / 空格 / \\t）。"""
    return start >= end or not text[start:end].strip()


def _first_non_blank(text: str, start: int, end: int) -> int:
    """[start, end) 区间首个非空白字符下标；全空白返回 end。"""
    for i in range(start, min(end, len(text))):
        if not text[i].isspace():
            return i
    return min(end, len(text))


def _target_fn_name(targets: list[dict[str, Any]]) -> str | None:
    """从 targets 提取首个 tool call 的函数名。"""
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if not isinstance(output, dict):
            continue
        for tc in output.get("tool_calls") or []:
            if isinstance(tc, dict) and isinstance(tc.get("name"), str):
                return tc["name"]
    return None


def _target_param_order(targets: list[dict[str, Any]], fn_name: str) -> list[str]:
    """从 targets 提取指定函数名的参数顺序（GT 顺序）。"""
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if not isinstance(output, dict):
            continue
        for tc in output.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("name") == fn_name:
                arguments = tc.get("arguments")
                if isinstance(arguments, dict):
                    return list(arguments.keys())
    return []


def _target_param_types(targets: list[dict[str, Any]], fn_name: str) -> dict[str, Any]:
    """从 targets 提取参数名 → GT 值（用于值类型校验）。"""
    types: dict[str, Any] = {}
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if not isinstance(output, dict):
            continue
        for tc in output.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("name") == fn_name:
                arguments = tc.get("arguments")
                if isinstance(arguments, dict):
                    types.update(arguments)
    return types


class _Annotator:
    """工具类：持有 tags/fields 并维护当前推进位置。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.tags: list[CharTag] = [CharTag.WASTE] * self.n
        self.fields: list[str | None] = [None] * self.n

    def mark(self, start: int, end: int, tag: CharTag, consume_newline: bool = True) -> int:
        """标 [start, end) 为 tag；STRUCTURE 且 end 后紧跟换行时把换行一并标 S。"""
        e = min(end, self.n)
        for i in range(start, e):
            self.tags[i] = tag
        if consume_newline and tag == CharTag.STRUCTURE and e < self.n and self.text[e] == "\n":
            self.tags[e] = tag
            return e + 1
        return e

    def drop_from(self, start: int) -> None:
        for i in range(start, self.n):
            self.tags[i] = CharTag.DROPPED

    def match_expected(
        self,
        pos: int,
        expected: str,
        *,
        consume_newline: bool = True,
    ) -> tuple[bool, int]:
        """从 pos 起（跳过空白）逐字符比对期望 mark。

        :param pos: 当前推进位置
        :param expected: 期望 mark 字符串（如 ``<function=example_tool>``）
        :param consume_newline: 匹配成功后是否吞并紧随换行（think 闭合为 False）
        :return: (matched, new_pos)。matched=False 时已标 E 并截断（或文本提前结束
            结构不完整——此时不标 E，全 S，返回 (True, n) 终止语义由调用方处理）
        """
        first = _first_non_blank(self.text, pos, self.n)
        # 逐字符比对
        for i, ec in enumerate(expected):
            idx = first + i
            if idx >= self.n:
                # 文本在 mark 中途结束：已有字符全正确 → 结构不完整，无 E
                return True, self.n
            if self.text[idx] != ec:
                # 首个不匹配字符 → E，其后 D
                self.tags[idx] = CharTag.ERROR
                self.drop_from(idx + 1)
                return False, idx
            self.tags[idx] = CharTag.STRUCTURE
        # 匹配成功：吞并紧随换行
        end = first + len(expected)
        if consume_newline and end < self.n and self.text[end] == "\n":
            self.tags[end] = CharTag.STRUCTURE
            end += 1
        return True, end

    def match_function_name(self, pos: int, gt_name: str | None) -> tuple[bool, int]:
        """比对 ``<function=NAME>``：``<function=`` 与 ``>`` 是结构，**NAME 是可评分字段**。

        P1 的落点：NAME 若走 ``match_expected``（逐字符），写错时只有首个不匹配字符吃 −1、
        其后全 D，而名字前面的结构 token 全是 +1 ⇒ 整条 completion 仍是正的（实测 +21）。
        这里把 NAME 当身份字段：整段对 ⇒ 整段 ``VALUE(__tool_name__)``（+1.0 每 token）；
        写错 ⇒ 写错那段整段 ``VALUE``（−1 每 token）+ drop。

        ⚠ 名字写对时也是 ``VALUE`` 而**不是** S（B1 修复，2026-11-24）：工具名算值，不算结构
        ⇒ 不计入 ``all_correct_structure_chars`` 的 S 字符数 ⇒ P3 的 mask 伸不进工具名。

        :param pos: 当前推进位置（``<tool_call>`` 之后）
        :param gt_name: GT 工具名；``None``（targets 里没有工具名）时退回改前的逐字符比对
        :return: ``(matched, new_pos)``；``matched=False`` 时已标错并截断，调用方应停止
        """
        first = _first_non_blank(self.text, pos, self.n)

        # 1. ``<function=`` 前缀是结构：拼错仍按「首个不匹配 E + 其后 D」
        for offset, expected_char in enumerate(_FUNCTION_MARK):
            idx = first + offset
            if idx >= self.n:
                # 文本在前缀中途结束：已有字符全正确 → 结构不完整，无 E
                return True, self.n
            if self.text[idx] != expected_char:
                self.tags[idx] = CharTag.ERROR
                self.drop_from(idx + 1)
                return False, idx
            self.tags[idx] = CharTag.STRUCTURE
        name_start = first + len(_FUNCTION_MARK)

        # 2. 名字段的边界：``>`` 必须与本行同名（跨行 find 会把后面的 ``>`` 误当闭合）
        name_end = _line_local_gt(self.text, name_start, self.n)
        if gt_name is None or name_end < 0:
            # 无 GT 名 / 名字段无法界定 ⇒ 退回改前的逐字符比对（含结构不完整语义）
            return self.match_expected(name_start, f"{gt_name}>" if gt_name else "?>")

        model_name = self.text[name_start:name_end]
        if model_name == gt_name:
            # 名字写对 ⇒ 整段 ``VALUE(__tool_name__)`` ⇒ raw 满分 ⇒ 每 token +1.0。
            # ⛔ 不标 S（B1，2026-11-24）：工具名算**值**（选错），不是结构（写错）。
            #   标 S 会让名字字符进入 P3 的 S 字符计数，``k`` 就可能伸进名字内部，
            #   把**写对的工具名** mask 成 0.0（反例见模块 docstring 的 B1 段）。
            for i in range(name_start, name_end):
                self.tags[i] = CharTag.VALUE
                self.fields[i] = TOOL_NAME_FIELD
            end = name_end + 1
            self.tags[name_end] = CharTag.STRUCTURE  # '>' 是结构（语法，不是选择）
        elif "<" in model_name:
            # 名字段里出现 ``<`` ⇒ 找到的 ``>`` 不是名字的闭合，而是**后面标签**（
            # ``<parameter=…`` / ``</function>``）的 ``>`` ⇒ 名字后缺 ``>``。
            # 按既有 T31/T32 口径（`<parameter=` 残缺标签）处理：在应出现 ``>`` 的位置
            # 标 E + drop，**不产生 V 段**。
            # 为什么不能走下面的 P1 分支：那会把 ``<parameter=angle_deg`` 这些**结构字符**
            # 当 ``VALUE(__tool_name__)`` 按「值错」扣 −1（实测一条 28 个 −1、Σadv −4.0），
            # 而名字本身可能写对了 ⇒ 正确 token 吃负分，正是 P1 要避免的误伤。
            # 顺序上先比 ``model_name == gt_name``：GT 名字自身含 ``<``（极端情况）时仍按
            # 「名字写对」处理，不被这条判据误伤。
            return self.match_expected(name_start, f"{gt_name}>")
        else:
            common = _common_prefix_length(model_name, gt_name)
            for i in range(name_start, name_start + common):
                self.tags[i] = CharTag.STRUCTURE  # 写对的那部分仍是结构（+1.0）
            if common < len(model_name):
                # 从首个不匹配字符起整段是「错的值」：整段吃 −1，随后全 D
                for i in range(name_start + common, name_end):
                    self.tags[i] = CharTag.VALUE
                    self.fields[i] = TOOL_NAME_FIELD
                self.drop_from(name_end)
            else:
                # 名字比 GT 短：模型在"名字该继续"处写了 '>'（残缺名字）⇒ E 于该处 + drop
                self.tags[name_end] = CharTag.ERROR
                self.drop_from(name_end + 1)
            return False, name_end
        # 3. 匹配成功：吞并紧随换行（与 ``match_expected`` 的既有行为一致）
        if end < self.n and self.text[end] == "\n":
            self.tags[end] = CharTag.STRUCTURE
            end += 1
        return True, end


def _line_local_gt(text: str, start: int, n: int) -> int:
    """从 start 起**同一行内**首个 ``>`` 的下标；同行无 ``>`` 返回 -1。

    为什么限行内：名字段若跨行 ``find(">")``，会把后面 ``</function>`` 的 ``>`` 当成名字闭合
    （与 ``_peek_param_chunk`` 同一防呆理由，T32 修复口径）。

    ⚠ 同行也仍可能拿到**后面标签**的 ``>``（``<function=name<parameter=x>``）——这一层由
    调用方 :meth:`_Annotator.match_function_name` 用「名字段含 ``<``」判据兜住（B1 修复）。
    """
    newline = text.find("\n", start)
    gt = text.find(">", start)
    if gt < 0 or gt >= n or (newline >= 0 and gt > newline):
        return -1
    return gt


def _common_prefix_length(left: str, right: str) -> int:
    """两段文本的公共前缀长度（用于定位「写错的那一段」）。"""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class _ParamChunk:
    """参数循环中模型输出的下一个块（切块阶段的结果，lookahead）。

    切块与核对分离（v3.2）：本类只描述"模型写了什么块"，不判断对错；
    期望驱动的核对由 annotate_tool_call 的参数循环按块类型决策。
    """

    first: int  # 块起点（<parameter= 的 <，跳过空白后）
    name_start: int  # 参数名起点（<parameter= 之后）
    name_end: int  # 参数名终点；-1 = 残缺（当前行内无 >，标签未闭合）


def _peek_param_chunk(text: str, pos: int, n: int) -> _ParamChunk | None:
    """从 pos 跳过空白，向前看下一个块是否为参数标签块（切块阶段）。

    规则（通用文本形态，无内容特判）：
    - 下一个非空白以 ``<parameter=`` 开头 → 参数标签块；
    - ``>`` 必须与 ``<parameter=`` 同行（跨行 find 会把后续 ``</function>`` 的
      ``>`` 误当闭合——T32 修复）：当前行内无 ``>`` → 残缺块（name_end=-1）；
    - 其他（闭合标签/文本/文本结束）→ None（参数循环结束或跳闭合）。
    """
    first = _first_non_blank(text, pos, n)
    if first >= n or not text.startswith(_PARAMETER_MARK, first):
        return None
    name_start = first + len(_PARAMETER_MARK)
    nl = text.find("\n", name_start)
    gt_pos = text.find(">", name_start)
    name_end = gt_pos if (gt_pos >= 0 and (nl < 0 or gt_pos < nl)) else -1
    return _ParamChunk(first=first, name_start=name_start, name_end=name_end)


def annotate_tool_call(
    text: str,
    targets: list[dict[str, Any]],
    check_think: bool = False,
) -> tuple[list[CharTag], list[str | None]]:
    """对 tool call 格式 completion 做逐字符标注。

    :return: ``(tags, fields)``，两列表长度均 == len(text)
    """
    a = _Annotator(text)
    n = a.n

    pos = 0

    # ---- 1. think 标签 ----
    if check_think:
        th_open = text.find(_THINK_OPEN, pos)
        if th_open >= 0:
            th_close = text.find(_THINK_CLOSE, th_open + len(_THINK_OPEN))
            if th_close >= 0:
                a.mark(th_open, th_open + len(_THINK_OPEN), CharTag.STRUCTURE)
                a.mark(th_open + len(_THINK_OPEN), th_close, CharTag.THINK)
                pos = a.mark(
                    th_close,
                    th_close + len(_THINK_CLOSE),
                    CharTag.STRUCTURE,
                    consume_newline=False,
                )

    # ---- 2. 开标签 ----
    open_pos = text.find(_OPEN_TAG, pos)
    if open_pos < 0:
        # 无结构 → 全部 WASTE（纯乱码 / 前导文本）
        return a.tags, a.fields
    a.mark(open_pos, open_pos + _OPEN_TAG_LEN, CharTag.STRUCTURE)
    pos = open_pos + _OPEN_TAG_LEN
    if pos < n and text[pos] == "\n":
        a.tags[pos] = CharTag.STRUCTURE
        pos += 1

    fn_name = _target_fn_name(targets)
    param_order = _target_param_order(targets, fn_name) if fn_name else []
    param_types = _target_param_types(targets, fn_name) if fn_name else {}

    # ---- 3. 期望 mark 序列逐字符比对 ----
    # 3.1 <function=NAME>：NAME 是**可评分字段**（P1，见模块 docstring）
    ok, pos = a.match_function_name(pos, fn_name)
    if not ok:
        return a.tags, a.fields

    # 3.2 参数循环：<parameter=NAME> value </parameter>（切块 → 核对，v3.2）
    # 参数匹配为**无序集合**（tool call 参数顺序无关，T16 顺序颠倒不标 E）；
    # 参数名不在 GT 集合 → 多余字段 E（T07）；GT 参数有缺失 → 缺失检测（T06）
    param_set = set(param_order) if param_order else set()
    seen_params: set[str] = set()
    while True:
        # ── 切块阶段：模型的下一个块是什么 ──
        chunk = _peek_param_chunk(text, pos, n)
        if chunk is None:
            # 非参数标签块（闭合标签/文本/文本结束）→ 参数循环结束（T01 回归点）
            break
        first = chunk.first
        name_start = chunk.name_start

        # ── 核对阶段 1：残缺标签（期望参数标签 vs 模型残缺标签）──
        # v3.1 用户裁定：E 于"应出现 > 的位置"（参数名后首字符，通常 \n 或行尾），
        # 而非标签首字符；`<parameter=` 与参数名本身是正确结构 → S。
        if chunk.name_end < 0:
            a.mark(first, name_start, CharTag.STRUCTURE)  # <parameter= 前导
            nl_pos = text.find("\n", name_start)
            param_len = (nl_pos - name_start) if nl_pos >= 0 else (n - name_start)
            a.mark(
                name_start, name_start + param_len, CharTag.STRUCTURE, consume_newline=False
            )  # 参数名（不吞并后随换行）
            e_pos = name_start + param_len  # 参数名后第一个字符（应出现 > 处）
            if e_pos < n:
                a.tags[e_pos] = CharTag.ERROR
                a.drop_from(e_pos + 1)
            else:
                # 参数名到文本末尾（无后续字符可标）→ 标签首字符兜底
                a.tags[first] = CharTag.ERROR
                a.drop_from(first + 1)
            return a.tags, a.fields
        param_name = text[name_start : chunk.name_end]
        if param_set and param_name not in param_set:
            # 多余/未知参数块（v3.0 语义，2026-08-07 用户裁定）：
            # `<parameter=` 前导结构本身正确 → S；错误本体是参数名 →
            # E 定位在参数名首字符（T07 修订：不再从 </function> 分叉点 p 开始）
            a.mark(first, name_start, CharTag.STRUCTURE)
            if name_start < n:
                a.tags[name_start] = CharTag.ERROR
                a.drop_from(name_start + 1)
            return a.tags, a.fields
        p_mark = f"{_PARAMETER_MARK}{param_name}>"
        # 参数开标签逐字符比对（不吞并后随换行——值前空白是格式自由区）
        ok, pos = a.match_expected(pos, p_mark, consume_newline=False)
        if not ok:
            return a.tags, a.fields
        seen_params.add(param_name)

        # value span：<parameter=NAME> 后跳过空白到值。
        # v3.2（2026-08-07 用户裁定）：值前后空白标 W（格式自由区，不训练）——
        # 标 V 会污染字段值（value_spans/field_score 解析带空白），
        # 标 S 会过度约束空白形态（空格/换行/无都是合法分隔）。
        v_start = pos
        while v_start < n and text[v_start].isspace():
            a.tags[v_start] = CharTag.WASTE
            v_start += 1
        # 值段（值本身 = 非空白连续段，遇标签形态 </parameter> 停止）
        v_end = v_start
        while (
            v_end < n and not text[v_end].isspace() and not text.startswith(_PARAMETER_CLOSE, v_end)
        ):
            v_end += 1
        if v_end == v_start and v_start >= n:
            # 标签后文本结束（无后续字符可标值）→ 结构不完整，不标 E
            pass
        elif v_end == v_start:
            # v3.1（2026-08-07 用户裁定）：空参数值 = 格式错误。
            # E 于值位置（空白段最后一个字符，即"该有值却为空"处），E 后全 D——
            # 后续即使正确（如 angle_deg=40.0）也不参与训练（基于格式错误
            # 前缀 token 的训练无意义，坚持"E 后全 D"原则）。
            e_pos = v_start - 1 if v_start > pos else pos
            a.tags[e_pos] = CharTag.ERROR
            a.drop_from(e_pos + 1)
            return a.tags, a.fields
        if v_end > v_start:
            # 值类型校验：GT 是数字但模型值不是 → E（T12）
            if (
                param_name is not None
                and isinstance(param_types.get(param_name), (int, float))
                and not isinstance(param_types.get(param_name), bool)
            ):
                try:
                    float(text[v_start:v_end])
                except ValueError:
                    a.tags[v_start] = CharTag.ERROR
                    a.drop_from(v_start + 1)
                    return a.tags, a.fields
            pos = a.mark(v_start, v_end, CharTag.VALUE)
            if param_name is not None:
                for i in range(v_start, v_end):
                    a.fields[i] = param_name
        # 值尾（跳过空白）后必须立即是 </parameter>；值后空白标 W（v3.2）
        tail = _first_non_blank(text, v_end, n)
        for i in range(v_end, tail):
            a.tags[i] = CharTag.WASTE
        ok, pos = a.match_expected(tail, _PARAMETER_CLOSE)
        if not ok:
            return a.tags, a.fields

    # 3.3 缺失检测：GT 参数未全部出现 → 期望此处是缺失参数 <parameter=NAME>，
    # 与模型实际输出（通常是 </function>）逐字符比对 → 分叉点 E（T06）
    missing = [p for p in param_order if p not in seen_params]
    if missing:
        p_mark = f"{_PARAMETER_MARK}{missing[0]}>"
        ok, pos = a.match_expected(pos, p_mark)
        if not ok:
            return a.tags, a.fields

    # 3.4 闭合：</function> </tool_call>
    ok, pos = a.match_expected(pos, _FUNCTION_CLOSE)
    if not ok:
        return a.tags, a.fields
    ok, pos = a.match_expected(pos, _TOOL_CALL_CLOSE, consume_newline=False)
    if not ok:
        return a.tags, a.fields

    # 3.5 闭合后多余内容检测（v5.0，用户裁定 2026-08-08——取代 v3.0 T33 紧邻检测）：
    # "</tool_call>" 之后第一个字符（含换行符）直接 E，其后全 D——一刀切。
    # 语义：工具调用之后没有思考/结构，只可能是乱码，要压制而不是放弃
    # （v21 实测多调用 0.5%→9.8% 增长，根因是间隔杂散标签形态全部漏检）。
    # 千问模板无 </tool_call> 后换行规定（实测 1312 条合法样本闭合后均空串）。
    # 统一规则：换行标注跟随其后内容——后面是结构→S、后面是值→W（v3.2
    # 值空白裁定）、后面是"结束后的多余"→E。
    if pos < n:
        a.tags[pos] = CharTag.ERROR
        a.drop_from(pos + 1)
    return a.tags, a.fields
