"""标注枚举（CharTag）：逐字符结构角色。

设计原则（v0.20.0）：
- 不做 FORMAT/CONTENT 分类（那是整体打分时代的产物），每个 token 只关注
  "在当前前缀条件下与模板结构的对齐状态"
- E 与 D 分离是"严格对齐截断"语义的显式化：首错点给负分，其后所有字符
  （包括本可能正确的结构）不再训练，因为其条件概率建立在错误前缀之上
"""

from enum import StrEnum


class CharTag(StrEnum):
    """逐字符标注枚举。

    值使用单字符代号（S/V/T/W/E/D），便于测试数据集逐字符比对与可视化。
    """

    STRUCTURE = "S"
    """结构字符（XML 标签 / JSON 标点与字段名），与模板对齐正确。下游 +1.0。"""

    VALUE = "V"
    """值字符（参数值 / JSON value），field 见并行数组。下游按相似度打分 0~1。"""

    THINK = "T"
    """think 内容（``<think>`` 与 ``</think>`` 标记之间）。0 分，不训练。"""

    WASTE = "W"
    """前导/尾随/夹缝多余文本。0 分，不训练，不触发截断。"""

    ERROR = "E"
    """首个结构对齐失败点（废话 / 拼错 / 类型错 / 结构不完整等）。下游 -1.0。"""

    DROPPED = "D"
    """E 之后的所有字符（截断，不训练）。0 分。"""

    @property
    def trainable(self) -> bool:
        """该角色是否进入训练（E 参与训练作为负样本，D 不参与）。"""
        return self in (CharTag.STRUCTURE, CharTag.VALUE, CharTag.ERROR)


# 兼容别名（旧文档用全小写）
STRUCTURE = CharTag.STRUCTURE
VALUE = CharTag.VALUE
THINK = CharTag.THINK
WASTE = CharTag.WASTE
ERROR = CharTag.ERROR
DROPPED = CharTag.DROPPED
