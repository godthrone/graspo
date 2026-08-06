"""标注模块（Annotation）：rollout 完成后对 completion 的逐字符结构标注。

职责边界：
- 只做**分类与对齐判定**，不做打分（打分由 advantage 层消费枚举后完成）
- 输出 ``List[CharTag]``（长度 == len(completion)）+ 并行 field 数组
- 支持两种格式：JSON（围栏可有可无）与 tool call（Qwen XML 等）
- 严格对齐截断：首个结构错误点标 E，其后所有字符标 D（不训练）
"""

from .labeler import AnnotationInput, AnnotationResult, annotate
from .roles import CharTag

__all__ = ["CharTag", "AnnotationInput", "AnnotationResult", "annotate"]
