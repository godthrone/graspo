from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedCompletion:
    raw_text: str
    think_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    answer_text: str = ""
    parser_name: str = "raw"
    parse_errors: list[str] = field(default_factory=list)
    extra_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def raw_parsed_completion(text: str, *, parser_name: str = "raw") -> ParsedCompletion:
    return ParsedCompletion(
        raw_text=text,
        answer_text=text,
        parser_name=parser_name,
        extra_text="",
    )
