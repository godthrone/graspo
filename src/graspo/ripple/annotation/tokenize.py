"""字符标注 → token 标注：通过 tokenizer 的 offset_mapping 映射。

设计（方案 §3.3）：
- 字符标注正确 → token 标注自动正确（offset_mapping 保证字符区间与 token 一一对应）
- 一个 token 跨多个字符时，token 标注取各字符标注的"合并语义"：
  - 任一字符是 E → token 标 E（错误优先，保证截断语义传导）
  - 否则若字符包含 V → token 标 V（值优先于结构）
  - 否则取首字符标注
- field 取首个非 None 的字符 field
"""

from dataclasses import dataclass
from typing import Any

from .roles import CharTag


@dataclass(frozen=True)
class TokenLabel:
    """token 级标注。"""

    token_index: int  # 生成区域内 token 下标
    tag: CharTag
    field: str | None


def _merge_tag(chars: list[CharTag]) -> CharTag:
    """合并一个 token 覆盖的多个字符标注为一个 token 标注。"""
    if any(c == CharTag.ERROR for c in chars):
        return CharTag.ERROR
    if any(c == CharTag.VALUE for c in chars):
        return CharTag.VALUE
    if any(c == CharTag.STRUCTURE for c in chars):
        return CharTag.STRUCTURE
    if any(c == CharTag.THINK for c in chars):
        return CharTag.THINK
    if any(c == CharTag.DROPPED for c in chars):
        return CharTag.DROPPED
    return CharTag.WASTE


def char_to_token_labels(
    completion_text: str,
    tags: list[CharTag],
    fields: list[str | None],
    tokenizer: Any,
) -> list[TokenLabel]:
    """把字符标注转换为生成区域内的 token 标注。

    :param completion_text: 生成区域文本
    :param tags: 字符标注（长度 == len(completion_text)）
    :param fields: 字符 field 数组
    :param tokenizer: HF tokenizer（需支持 offset_mapping）
    :return: TokenLabel 列表（长度 == 生成 token 数）
    """
    if not completion_text:
        return []
    encoding = tokenizer(completion_text, return_offsets_mapping=True)
    offsets = encoding.get("offset_mapping")
    if offsets is None:  # pragma: no cover - 防御
        raise ValueError("tokenizer must return offset_mapping (return_offsets_mapping=True)")

    labels: list[TokenLabel] = []
    for token_index, (start, end) in enumerate(offsets):
        # BPE token 覆盖 [start, end)；特殊 token 可能是 (0, 0)
        if end <= start:
            continue
        char_tags = tags[start:end]
        char_fields = fields[start:end]
        field = next((f for f in char_fields if f is not None), None)
        labels.append(TokenLabel(token_index=token_index, tag=_merge_tag(char_tags), field=field))
    return labels


__all__ = ["TokenLabel", "char_to_token_labels"]
