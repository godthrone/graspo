"""tool call 标注：mark 定位 + 恒等式逐步推进（非正则）。

流程（对应方案 §6）：
1. think 标签处理：``<think>`` 与 ``</think>`` 标记 → S，内容 → T
2. 找开标签 ``<tool_call>``（由 chat template 决定，Qwen 家族统一为 ``<tool_call>``）
3. 从开标签起，逐步定位下一个 mark（``<function=`` / ``<parameter=`` / 闭合标签），
   两个 mark 之间的非空白文本做恒等式检查——不恒等即首个 E，其后全部 D
4. 缺失 mark（结构不完整/截断）→ 最后一个结构字符标 E

标注语义（对齐方案 §7 严格对齐）：
- 结构开始后必须严格对齐，结构之间插入废话直接失败
- 首错点 E（下游 -1.0），其后所有字符 D（不训练）
- 结构标签**及紧随其后的换行**标 S（数据集约定 ``<tool_call>\\n`` 整体 S）
"""

from typing import Any

from .roles import CharTag

# Qwen 家族 tool call 开标签（chat template 决定；模型族注册表在此扩展）
_OPEN_TAG = "<tool_call>"
_OPEN_TAG_LEN = len(_OPEN_TAG)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_FUNCTION_MARK = "<function="
_PARAMETER_MARK = "<parameter="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_CLOSE = "</parameter>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _blank(text: str, start: int, end: int) -> bool:
    """[start, end) 区间是否全为空白（\\n / 空格 / \\t）。"""
    return start >= end or not text[start:end].strip()


def _first_non_blank(text: str, start: int, end: int) -> int:
    """[start, end) 区间首个非空白字符下标；全空白返回 end。"""
    for i in range(start, min(end, len(text))):
        if not text[i].isspace():
            return i
    return min(end, len(text))


class _Annotator:
    """工具类：持有 tags/fields 并维护当前推进位置。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.tags: list[CharTag] = [CharTag.WASTE] * self.n
        self.fields: list[str | None] = [None] * self.n

    def mark(self, start: int, end: int, tag: CharTag, consume_newline: bool = True) -> int:
        """标 [start, end) 为 tag；STRUCTURE 且 end 后紧跟换行时把换行一并标 S。

        :param consume_newline: 是否吞并紧随的换行（think 闭合标记后为 False——
            数据集约定 ``</think>`` 后的 ``\n`` 是 WASTE 夹缝，不是结构换行）
        """
        e = min(end, self.n)
        for i in range(start, e):
            self.tags[i] = tag
        if (
            consume_newline
            and tag == CharTag.STRUCTURE
            and e < self.n
            and self.text[e] == "\n"
        ):
            self.tags[e] = tag
            return e + 1
        return e

    def drop_from(self, start: int) -> None:
        for i in range(start, self.n):
            self.tags[i] = CharTag.DROPPED


def _target_fn_names(targets: list[dict[str, Any]]) -> set[str]:
    """从 targets 提取期望函数名集合。"""
    names: set[str] = set()
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if not isinstance(output, dict):
            continue
        for tc in output.get("tool_calls") or []:
            if isinstance(tc, dict) and isinstance(tc.get("name"), str):
                names.add(tc["name"])
    return names


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


def _target_param_types(
    targets: list[dict[str, Any]],
    fn_name: str,
) -> dict[str, Any]:
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
                    th_close, th_close + len(_THINK_CLOSE), CharTag.STRUCTURE,
                    consume_newline=False,
                )

    # ---- 2. 开标签 ----
    open_pos = text.find(_OPEN_TAG, pos)
    if open_pos < 0:
        # 无结构 → 全部 WASTE（纯乱码 / 前导文本）
        return a.tags, a.fields
    pos = a.mark(open_pos, open_pos + _OPEN_TAG_LEN, CharTag.STRUCTURE)

    # ---- 3. <function= ----
    fn_pos = text.find(_FUNCTION_MARK, pos)
    if fn_pos < 0:
        # 结构不完整（开标签后无 function）→ 首个非空白字符 E
        e = _first_non_blank(text, pos, n)
        if e < n:
            a.tags[e] = CharTag.ERROR
            a.drop_from(e + 1)
        return a.tags, a.fields
    if not _blank(text, pos, fn_pos):
        e = _first_non_blank(text, pos, fn_pos)
        a.tags[e] = CharTag.ERROR
        a.drop_from(e + 1)
        return a.tags, a.fields
    fn_close = text.find(">", fn_pos + len(_FUNCTION_MARK))
    if fn_close < 0:
        a.tags[fn_pos] = CharTag.ERROR
        a.drop_from(fn_pos + 1)
        return a.tags, a.fields
    fn_name = text[fn_pos + len(_FUNCTION_MARK):fn_close]
    fn_names = _target_fn_names(targets)
    if fn_names and fn_name not in fn_names:
        # 函数名不恒等 → 函数名首字符 E，其后 D（T13）
        a.mark(fn_pos, fn_pos + len(_FUNCTION_MARK), CharTag.STRUCTURE)
        a.tags[fn_pos + len(_FUNCTION_MARK)] = CharTag.ERROR
        a.drop_from(fn_pos + len(_FUNCTION_MARK) + 1)
        return a.tags, a.fields
    pos = a.mark(fn_pos, fn_close + 1, CharTag.STRUCTURE)

    # ---- 4. 参数循环：<parameter=NAME> value </parameter> ----
    # GT 参数顺序校验（T16 参数顺序颠倒 → 期望参数迟到时 E）
    param_order = _target_param_order(targets, fn_name)
    order_index = 0  # GT 参数顺序中的当前位置
    saw_order_mismatch = False  # 已出现"期望参数被跳过"（顺序颠倒）

    while True:
        p_pos = text.find(_PARAMETER_MARK, pos)
        if p_pos < 0:
            break
        if not _blank(text, pos, p_pos):
            e = _first_non_blank(text, pos, p_pos)
            a.tags[e] = CharTag.ERROR
            a.drop_from(e + 1)
            return a.tags, a.fields
        p_close = text.find(">", p_pos + len(_PARAMETER_MARK))
        if p_close < 0:
            a.tags[p_pos] = CharTag.ERROR
            a.drop_from(p_pos + 1)
            return a.tags, a.fields
        param_name = text[p_pos + len(_PARAMETER_MARK):p_close]
        if not param_name:
            a.tags[p_pos] = CharTag.ERROR
            a.drop_from(p_pos + 1)
            return a.tags, a.fields
        # 参数名校验：不在 GT 参数集合中 → E（T07 多余字段）
        if param_order and param_name not in param_order:
            a.tags[p_pos] = CharTag.ERROR
            a.drop_from(p_pos + 1)
            return a.tags, a.fields
        # 顺序校验：期望参数被前面的参数跳过 → 期望参数迟到即 E（T16）
        if param_order and order_index < len(param_order):
            if param_name == param_order[order_index]:
                if saw_order_mismatch:
                    # 期望参数迟到（顺序颠倒）→ E
                    a.tags[p_pos] = CharTag.ERROR
                    a.drop_from(p_pos + 1)
                    return a.tags, a.fields
                order_index += 1
            else:
                # 当前参数不是期望的下一个 → 记录错序，但本身合法照常标注
                saw_order_mismatch = True
        pos = a.mark(p_pos, p_close + 1, CharTag.STRUCTURE)
        # value span：<parameter=NAME> 后跳过空白到值
        v_start = _first_non_blank(text, pos, n)
        v_close = text.find(_PARAMETER_CLOSE, v_start)
        if v_close < 0:
            # 值未闭合（缺 </parameter> / 截断）→ 值首字符 E
            if v_start < n:
                a.tags[v_start] = CharTag.ERROR
                a.drop_from(v_start + 1)
            return a.tags, a.fields
        # value 段（值本身；值后的空白归结构——数据集约定 V 后紧跟的 \n 是 S）
        v_end = v_start
        while v_end < n and not text[v_end].isspace() and not text.startswith(
            _PARAMETER_CLOSE, v_end
        ):
            v_end += 1
        # 先标值本身（值正确则 V，值类型错则 E）——值标注独立于后续闭合检查
        if v_end > v_start:
            # 值类型校验：GT 是数字但模型值不是 → E（T12）
            gt_value = _target_param_types(targets, fn_name).get(param_name)
            if isinstance(gt_value, (int, float)) and not isinstance(gt_value, bool):
                try:
                    float(text[v_start:v_end])
                except ValueError:
                    a.tags[v_start] = CharTag.ERROR
                    a.drop_from(v_start + 1)
                    return a.tags, a.fields
            pos = a.mark(v_start, v_end, CharTag.VALUE)
            for i in range(v_start, v_end):
                a.fields[i] = param_name
        # 值尾（跳过空白）后必须立即是 </parameter>；拼错（</parametr>）→ E（T04）
        tail = _first_non_blank(text, v_end, n)
        # 值后的空白（\n）标 S（属于结构换行）
        for i in range(v_end, tail):
            a.tags[i] = CharTag.STRUCTURE
        if not text.startswith(_PARAMETER_CLOSE, tail):
            if tail < n:
                a.tags[tail] = CharTag.ERROR
                a.drop_from(tail + 1)
            return a.tags, a.fields
        v_close = tail
        # </parameter>
        pos = a.mark(v_close, v_close + len(_PARAMETER_CLOSE), CharTag.STRUCTURE)

    # ---- 5. 闭合：</function> </tool_call> ----
    fnc_pos = text.find(_FUNCTION_CLOSE, pos)
    if fnc_pos < 0:
        last = pos - 1
        if last < n:
            a.tags[last] = CharTag.ERROR
            a.drop_from(last + 1)
        return a.tags, a.fields
    if not _blank(text, pos, fnc_pos):
        e = _first_non_blank(text, pos, fnc_pos)
        a.tags[e] = CharTag.ERROR
        a.drop_from(e + 1)
        return a.tags, a.fields
    pos = a.mark(fnc_pos, fnc_pos + len(_FUNCTION_CLOSE), CharTag.STRUCTURE)

    tcc_pos = text.find(_TOOL_CALL_CLOSE, pos)
    if tcc_pos < 0:
        # 缺 </tool_call>：若存在部分标签（</tool_call 截断）→ 其 '<' 处 E（T20）；
        # 否则从最后一个闭合标签（</function>）起点 E（T14 结构不完整）
        partial = text.find("<", pos)
        if partial >= 0:
            a.tags[partial] = CharTag.ERROR
            a.drop_from(partial + 1)
        else:
            a.tags[fnc_pos] = CharTag.ERROR
            a.drop_from(fnc_pos + 1)
        return a.tags, a.fields
    if not _blank(text, pos, tcc_pos):
        e = _first_non_blank(text, pos, tcc_pos)
        a.tags[e] = CharTag.ERROR
        a.drop_from(e + 1)
        return a.tags, a.fields
    a.mark(tcc_pos, tcc_pos + len(_TOOL_CALL_CLOSE), CharTag.STRUCTURE, consume_newline=False)
    # 闭合后的尾随文本保持 WASTE（不截断）
    return a.tags, a.fields
