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


def _target_param_order(
    targets: list[dict[str, Any]], fn_name: str
) -> list[str]:
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
    targets: list[dict[str, Any]], fn_name: str
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

    def match_expected(
        self,
        pos: int,
        expected: str,
        *,
        consume_newline: bool = True,
    ) -> tuple[bool, int]:
        """从 pos 起（跳过空白）逐字符比对期望 mark。

        :param pos: 当前推进位置
        :param expected: 期望 mark 字符串（如 ``<function=robot_atomic_control>``）
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
        if (
            consume_newline
            and end < self.n
            and self.text[end] == "\n"
        ):
            self.tags[end] = CharTag.STRUCTURE
            end += 1
        return True, end


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
    a.mark(open_pos, open_pos + _OPEN_TAG_LEN, CharTag.STRUCTURE)
    pos = open_pos + _OPEN_TAG_LEN
    if pos < n and text[pos] == "\n":
        a.tags[pos] = CharTag.STRUCTURE
        pos += 1

    fn_name = _target_fn_name(targets)
    param_order = _target_param_order(targets, fn_name) if fn_name else []
    param_types = _target_param_types(targets, fn_name) if fn_name else {}

    # ---- 3. 期望 mark 序列逐字符比对 ----
    # 3.1 <function=NAME>
    fn_mark = f"{_FUNCTION_MARK}{fn_name}>" if fn_name else f"{_FUNCTION_MARK}?>"
    ok, pos = a.match_expected(pos, fn_mark)
    if not ok:
        return a.tags, a.fields

    # 3.2 参数循环：<parameter=NAME> value </parameter>
    # 参数匹配为**无序集合**（tool call 参数顺序无关，T16 顺序颠倒不标 E）；
    # 参数名不在 GT 集合 → 多余字段 E（T07）；GT 参数有缺失 → 缺失检测（T06）
    param_set = set(param_order) if param_order else set()
    seen_params: set[str] = set()
    while True:
        # 参数循环结束条件：下一个非空白不是 <parameter= （参数已全部处理完，
        # 或模型直接跳到闭合标签）。此时不应触发多余参数分支（T01 回归点）。
        first = _first_non_blank(text, pos, n)
        if first >= n or not text.startswith(_PARAMETER_MARK, first):
            break
        # 读取参数名：> 限制在当前行（跨行 find 会把后续 </function> 的 > 误当
        # 标签闭合，参数名含换行走错分支——T32 修复：标签未闭合按 name_end<0 处理）
        name_start = first + len(_PARAMETER_MARK)
        nl = text.find("\n", name_start)
        gt_pos = text.find(">", name_start)
        name_end = gt_pos if (gt_pos >= 0 and (nl < 0 or gt_pos < nl)) else -1
        if name_end < 0:
            a.tags[first] = CharTag.ERROR
            a.drop_from(first + 1)
            return a.tags, a.fields
        param_name = text[name_start:name_end]
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
        # 参数开标签逐字符比对
        ok, pos = a.match_expected(pos, p_mark)
        if not ok:
            return a.tags, a.fields
        seen_params.add(param_name)

        # value span：<parameter=NAME> 后跳过空白到值
        v_start = _first_non_blank(text, pos, n)
        # 值段（值本身；值后的空白归结构——数据集约定 V 后紧跟的 \n 是 S）
        v_end = v_start
        while v_end < n and not text[v_end].isspace() and not text.startswith(
            _PARAMETER_CLOSE, v_end
        ):
            v_end += 1
        if v_end > v_start:
            # 值类型校验：GT 是数字但模型值不是 → E（T12）
            if param_name is not None and isinstance(
                param_types.get(param_name), (int, float)
            ) and not isinstance(param_types.get(param_name), bool):
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
        # 值尾（跳过空白）后必须立即是 </parameter>（拼错 → E 于首个不匹配字符）
        tail = _first_non_blank(text, v_end, n)
        for i in range(v_end, tail):
            a.tags[i] = CharTag.STRUCTURE
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

    # 3.5 多余调用检测（v3.0：T33——"每轮只输出一个工具调用"约束违反）：
    # 闭合后尾随非空白且以 <tool_call> 开头 → 第二个调用首字符 E，其后 D。
    # （v20 毒药形态：完整首调用 + 尾部第二/三调用曾被标全 S/W 无 E 被正强化）
    after_close = _first_non_blank(text, pos, n)
    if after_close < n and text.startswith(_OPEN_TAG, after_close):
        a.tags[after_close] = CharTag.ERROR
        a.drop_from(after_close + 1)
        return a.tags, a.fields

    # 闭合后的尾随文本保持 WASTE（不截断；自然语言尾随 T17 语义不变）
    return a.tags, a.fields
