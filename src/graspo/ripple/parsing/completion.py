"""ParsedCompletion：模型输出解析结果数据模型（reward 前置）。"""

from pydantic import BaseModel, ConfigDict, Field


class ParsedCompletion(BaseModel):
    """模型输出解析后的结构化数据（不可变）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str
    think_text: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    answer_text: str = ""
    parser_name: str = "raw"
    parse_errors: list[str] = Field(default_factory=list)
    extra_text: str = ""

    def to_dict(self) -> dict:
        """兼容旧调用方（等价于 ``model_dump()``）。"""
        return self.model_dump()


def raw_parsed_completion(text: str, *, parser_name: str = "raw") -> ParsedCompletion:
    return ParsedCompletion(
        raw_text=text,
        answer_text=text,
        parser_name=parser_name,
        extra_text="",
    )
