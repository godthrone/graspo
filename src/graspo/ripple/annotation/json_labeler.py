"""JSON 标注：围栏提取 + 结构扫描（非正则，状态机逐字符）。

流程（对应方案 §5）：
1. 围栏处理：check_json_markdown=True 时找 `````json```` 围栏：
   - 缺失且文本以 JSON 结构字符（{ [ " 数字）开头 → 结构失败：首非空白 E（J06）
   - 缺失且文本是乱码（无结构起点）→ 全部 WASTE（J13）
   - 围栏标记本身 S，围栏后换行归 S
2. 状态机扫描 JSON body：
   - 标点 ``{ } [ ] : ,`` → S
   - 字段名（对象内 ``{`` 或 ``,`` 之后）→ S，且与 GT key 对比：
     精确匹配 → S；前缀匹配（拼错）→ 首个不匹配字符 E；多余 → 首字符 E
   - 值（冒号后 / 数组元素）→ V（含引号），且与 GT 值类型对比：
     GT 数字 vs 模型字符串（带引号）→ E（J09）
   - 结构语法错误（缺逗号 / 字符串未闭合 / 意外字符）→ E
3. E 之后全部 D（截断）
4. 闭合围栏（``\n``` ``）→ S；其后的尾随文本 → W

值错误（内容对不上 GT）**不**触发 E——那是下游打分的事；只有**结构/类型**错误触发 E。
"""

from typing import Any

from graspo.ripple.annotation.char_tag import CharTag

_FENCE = "```json"
_FENCE_CLOSE = "```"

# JSON 结构起点字符（用于区分"围栏缺失但结构存在"与"纯乱码"）
_JSON_START_CHARS = ("{", "[", '"', "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def _mark(tags: list[CharTag], start: int, end: int, tag: CharTag) -> None:
    for i in range(start, min(end, len(tags))):
        tags[i] = tag


def _drop(tags: list[CharTag], start: int) -> None:
    for i in range(start, len(tags)):
        tags[i] = CharTag.DROPPED


def _first_non_blank(text: str, start: int, end: int) -> int:
    for i in range(start, min(end, len(text))):
        if not text[i].isspace():
            return i
    return min(end, len(text))


def _find_string_end(text: str, start: int) -> int:
    """从 start（指向 ``"``）找字符串闭合引号（处理 \\ 转义）；未闭合返回 -1。"""
    i = start + 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i
        i += 1
    return -1


def _extract_gt(targets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从 targets 提取 GT JSON 对象（首个含 content 的对象）。"""
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if isinstance(output, dict) and isinstance(output.get("content"), dict):
            return output["content"]
    return None


def _child_gt(ctx: dict[str, Any] | None, pending_key: str | None) -> dict[str, Any] | None:
    """取嵌套子对象的 GT dict（沿 pending_key 下降一层）。

    进入 ``{`` 时调用：GT 中 pending_key 的值是 dict → 返回它（内层 key 用该
    子 dict 校验）；值是 list（数组元素为对象）→ 返回首个 dict 元素；
    其余（GT 缺失 / 标量 / 非 dict）→ None（内层不校验，保持顶层缺失的
    现有行为）。
    """
    if ctx is None or pending_key is None:
        return None
    val = ctx.get(pending_key)
    if isinstance(val, dict):
        return val
    if isinstance(val, list):
        for elem in val:
            if isinstance(elem, dict):
                return elem
    return None


def _check_key(key: str, gt: dict[str, Any] | None) -> tuple[str, str | None]:
    """字段名校验（gt 为当前上下文子对象，支持嵌套）。

    :return: (verdict, gt_key)。verdict ∈ {"ok", "typo", "extra"}
    """
    if gt is None or key in gt:
        return "ok", None
    # 前缀匹配（拼错）：与某个 GT key 共享 ≥2 字符前缀
    best = None
    best_len = 0
    for gk in gt:
        common = 0
        for a, b in zip(key, gk):
            if a != b:
                break
            common += 1
        if common >= 2 and common > best_len:
            best, best_len = gk, common
    if best is not None:
        return "typo", best
    return "extra", None


def annotate_json(
    text: str,
    targets: list[dict[str, Any]],
    check_json_markdown: bool = True,
) -> tuple[list[CharTag], list[str | None]]:
    """对 JSON 格式 completion 做逐字符标注。"""
    n = len(text)
    tags: list[CharTag] = [CharTag.WASTE] * n
    fields: list[str | None] = [None] * n
    gt = _extract_gt(targets)

    pos = 0
    if check_json_markdown:
        fpos = text.find(_FENCE)
        if fpos < 0:
            first = _first_non_blank(text, 0, n)
            if first < n and text[first] in _JSON_START_CHARS:
                # 围栏缺失但结构存在 → 结构失败：首非空白 E（J06）
                tags[first] = CharTag.ERROR
                _drop(tags, first + 1)
            # 否则（乱码无结构起点）→ 全部 WASTE（J13）
            return tags, fields
        _mark(tags, fpos, fpos + len(_FENCE), CharTag.STRUCTURE)
        pos = fpos + len(_FENCE)
        if pos < n and text[pos] == "\n":
            tags[pos] = CharTag.STRUCTURE
            pos += 1

    # ---- 状态机扫描 JSON body ----
    i = pos
    stack: list[str] = []  # "o" | "a"
    gt_ctx: list[dict[str, Any] | None] = [gt]  # 与 stack 平行的 GT 上下文栈
    after_colon = False
    pending_key: str | None = None

    while i < n:
        # 闭合围栏检测：遇到 ``` 提前结束扫描
        if text.startswith(_FENCE_CLOSE, i):
            break
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "{":
            tags[i] = CharTag.STRUCTURE
            if stack:
                # 嵌套对象：GT 上下文随 pending_key 下降一层（J16 嵌套 key 校验）
                gt_ctx.append(_child_gt(gt_ctx[-1], pending_key))
            else:
                # 顶层对象：上下文就是 gt 本身
                gt_ctx.append(gt_ctx[-1])
            stack.append("o")
            after_colon = False
            i += 1
            continue
        if c == "[":
            tags[i] = CharTag.STRUCTURE
            stack.append("a")
            after_colon = False
            i += 1
            continue
        if c == "}":
            if not stack or stack[-1] != "o":
                tags[i] = CharTag.ERROR
                _drop(tags, i + 1)
                return tags, fields
            tags[i] = CharTag.STRUCTURE
            stack.pop()
            gt_ctx.pop()
            after_colon = False
            pending_key = None
            i += 1
            continue
        if c == "]":
            if not stack or stack[-1] != "a":
                tags[i] = CharTag.ERROR
                _drop(tags, i + 1)
                return tags, fields
            tags[i] = CharTag.STRUCTURE
            stack.pop()
            after_colon = False
            i += 1
            continue
        if c == ":":
            tags[i] = CharTag.STRUCTURE
            after_colon = True
            i += 1
            continue
        if c == ",":
            tags[i] = CharTag.STRUCTURE
            after_colon = False
            i += 1
            continue
        if c == '"':
            # 语法检查：对象内 ] 或 } 之后直接 " 是缺逗号（J05）
            if stack and stack[-1] == "o" and not after_colon and i > 0:
                prev = text[i - 1]
                if prev in "]}":
                    tags[i] = CharTag.ERROR
                    _drop(tags, i + 1)
                    return tags, fields
            end = _find_string_end(text, i)
            if end < 0:
                # 未闭合字符串（截断）：已有字符全正确 → 值内容标 V，无 E
                # （结构不完整信号由 reward 层表达，避免误标正确 token）
                _mark(tags, i, n, CharTag.VALUE)
                if pending_key is not None:
                    for j in range(i, n):
                        fields[j] = pending_key
                return tags, fields
            in_obj_key = bool(stack) and stack[-1] == "o" and not after_colon
            if in_obj_key:
                key = text[i + 1 : end]
                verdict, gt_key = _check_key(key, gt_ctx[-1])
                if verdict == "ok":
                    _mark(tags, i, end + 1, CharTag.STRUCTURE)
                    pending_key = key
                elif verdict == "typo" and gt_key is not None:
                    # 拼错：前缀匹配部分 S，首个不匹配字符 E
                    common = sum(1 for a, b in zip(key, gt_key) if a == b)
                    _mark(tags, i, i + 1 + common, CharTag.STRUCTURE)
                    tags[i + 1 + common] = CharTag.ERROR
                    _drop(tags, i + 1 + common + 1)
                    return tags, fields
                else:
                    # 多余字段：首字符 E
                    tags[i] = CharTag.ERROR
                    _drop(tags, i + 1)
                    return tags, fields
            else:
                in_array = bool(stack) and stack[-1] == "a"
                # 类型校验：GT 数字 vs 模型字符串值（带引号）→ E
                ctx = gt_ctx[-1]
                if (
                    ctx is not None
                    and pending_key is not None
                    and not in_array
                    and isinstance(ctx.get(pending_key), (int, float))
                    and not isinstance(ctx.get(pending_key), bool)
                ):
                    tags[i] = CharTag.ERROR
                    _drop(tags, i + 1)
                    return tags, fields
                _mark(tags, i, end + 1, CharTag.VALUE)
                if pending_key is not None and not in_array:
                    for j in range(i, end + 1):
                        fields[j] = pending_key
                pending_key = None
            i = end + 1
            continue
        # 数字 / true / false / null（值）
        if c in "-0123456789" or text.startswith(("true", "false", "null"), i):
            if not after_colon and not (stack and stack[-1] == "a"):
                tags[i] = CharTag.ERROR
                _drop(tags, i + 1)
                return tags, fields
            j = i
            while j < n and not text[j].isspace() and text[j] not in ",}]":
                j += 1
            _mark(tags, i, j, CharTag.VALUE)
            if pending_key is not None and not (stack and stack[-1] == "a"):
                for k in range(i, j):
                    fields[k] = pending_key
            pending_key = None
            i = j
            continue
        # 意外字符
        tags[i] = CharTag.ERROR
        _drop(tags, i + 1)
        return tags, fields

    # ---- 闭合围栏 ----
    if check_json_markdown:
        fend = text.find(_FENCE_CLOSE, pos)
        if fend >= 0:
            if fend > 0 and text[fend - 1] == "\n":
                tags[fend - 1] = CharTag.STRUCTURE
            _mark(tags, fend, fend + len(_FENCE_CLOSE), CharTag.STRUCTURE)
    # 闭合围栏后保持 WASTE（尾随文本）

    return tags, fields
