"""Qwen 系列模型的公共配置基类。"""


from typing import Any


class NativeQwenConfig:
    """Qwen 原生模型配置，从 HuggingFace config 提取关键字段。

    通过 ``family`` 和 ``key_prefix`` 参数区分不同模型族。
    """

    def __init__(self, values: dict[str, Any], *, family: str, key_prefix: str) -> None:
        self.family: str = family
        self.key_prefix: str = key_prefix
        for key, value in values.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'NativeQwenConfig' object has no attribute '{name}'")
