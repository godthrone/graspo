"""标注主入口：completion + targets → 逐字符标注（List[CharTag] + field 数组）。

职责：只做分类与对齐判定，不做打分。
"""

from dataclasses import dataclass
from typing import Any

from graspo.ripple.annotation.char_tag import CharTag


@dataclass(frozen=True)
class AnnotationInput:
    """标注输入。format_type 由上层 Sample.expects_tool_calls 传入，不猜测。"""

    completion_text: str
    targets: list[dict[str, Any]]
    tokenizer: Any
    format_type: str  # "tool_call" | "json"
    check_json_markdown: bool = True
    check_think: bool = False


@dataclass(frozen=True)
class AnnotationResult:
    """标注输出：逐字符标注 + 并行 field 数组（长度均 == len(completion_text)）。"""

    tags: list[CharTag]
    fields: list[str | None]


def annotate(inp: AnnotationInput) -> AnnotationResult:
    """对 completion 做逐字符标注。"""
    if inp.format_type == "tool_call":
        from graspo.ripple.annotation.tool_call_labeler import annotate_tool_call

        tags, fields = annotate_tool_call(
            text=inp.completion_text,
            targets=inp.targets,
            check_think=inp.check_think,
        )
    elif inp.format_type == "json":
        from graspo.ripple.annotation.json_labeler import annotate_json

        tags, fields = annotate_json(
            text=inp.completion_text,
            targets=inp.targets,
            check_json_markdown=inp.check_json_markdown,
        )
    else:  # pragma: no cover - 上层已校验
        raise ValueError(f"unknown format_type: {inp.format_type!r}")
    return AnnotationResult(tags=tags, fields=fields)
