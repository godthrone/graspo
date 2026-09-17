"""ARD ``anchor_bank.jsonl`` ↔ graspo SFT 契约适配器（共享基座，决策 D4）。

**职责边界**

- **本模块**：把 ARD 记录（``messages`` + ``targets[{content, reasoning}]``，
  ``content`` 为 **str**）转成 graspo 训练管线能直接消费的样本
  （``output.content`` 为 **dict**），以及反方向的回流；顺带产出 ms-swift
  instruction-tuning 形态（query/response）的纯数据载荷。
- **不负责**：文件读写（调用方做）、tokenize/chat template（需要 tokenizer，
  属设施层，ms-swift 侧由 E2 接）、reward 打分、logprob（ARD 无 logprob 是
  特性，本模块不发明）。

**为什么需要它**：ARD 落盘 ``output.content`` 是纯文本 str，而 graspo 消费端
``ripple/reward/normalize.py`` 要求 ``content`` 是 JSON object；同时
``build_sft_target_text``（``ripple/parsing/xml.py``）对 **str** content 返回
空串 → SFT 立即 ``ValueError("primary target has no content or tool_calls")``。
适配器把 str 无损包成 ``{"text": <str>}`` 并从 ``output`` 内保留 ``reasoning``，
使同一条 ARD 记录可同时喂 SFT 与 graspo RL（D4 共享基座）。

**方向语义（一名一物）**

- ``GraspoToMsSwiftAdapter``：ARD 记录（str content）→ graspo 合法样本（dict content）
- ``MsSwiftToGraspoAdapter``：graspo/ms-swift 样本 → ARD v3 记录（str content）

两者互为逆：``MsSwiftToGraspoAdapter().convert_sample(
GraspoToMsSwiftAdapter().convert_sample(ard))`` 字段级逐字符还原 ``ard``。
"""

from __future__ import annotations

import copy
from typing import Any

from graspo.flow.msswift._ard_contract import (
    ARD_TOP_LEVEL_FIELDS,
    OUTPUT_REASONING_KEY,
    ArdContractError,
    ard_target_to_graspo,
    extract_ard_metadata,
    graspo_sample_to_ard,
    validate_ard_messages,
)

__all__ = [
    "ARD_TOP_LEVEL_FIELDS",
    "OUTPUT_REASONING_KEY",
    "ArdContractError",
    "GraspoToMsSwiftAdapter",
    "MsSwiftToGraspoAdapter",
]


class GraspoToMsSwiftAdapter:
    """ARD / graspo-JSONL 记录 → graspo 合法训练样本（并给出 ms-swift 形态）。

    输入是 **ARD v3 锚点记录**（``output.content`` 为 str），输出满足 graspo
    ``core.schema.Sample`` 的契约：

    - ``messages``：ARD 形状不变量校验通过后原样保留（含多模态 content 块）
    - ``targets[i].output.content``：str → ``{"text": str}``（无损）
    - ``targets[i].output.reasoning``：保留（可为 None）
    - 顶层路由字段（``id``/``source``/``data_source``/``schema_version``/
      ``teacher_id``/``input_generator_model``/``anchor_meta``）→ ``metadata``
    """

    def __init__(self) -> None:
        pass

    def convert_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """转换单条 ARD 记录为 graspo 合法样本。

        Args:
            sample: ARD v3 ``anchor_bank.jsonl`` 单行解析出的 dict。

        Returns:
            ``{"messages": [...], "targets": [...], "metadata": {...}}``；
            输入含 ``tools`` 时一并透传。

        Raises:
            ArdContractError: 记录不符合 ARD 共享基座契约（非法样本被拒绝，
                不做静默修补——宪法 §2.3）。
        """
        if not isinstance(sample, dict):
            raise ArdContractError(f"sample must be a JSON object, got {type(sample).__name__}")

        messages = validate_ard_messages(sample.get("messages"))

        targets = sample.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ArdContractError("sample must contain a non-empty 'targets' list")

        converted: dict[str, Any] = {
            "messages": messages,
            "targets": [
                ard_target_to_graspo(target, path=f"targets[{idx}]")
                for idx, target in enumerate(targets)
            ],
        }
        if "tools" in sample and sample["tools"] is not None:
            converted["tools"] = copy.deepcopy(sample["tools"])
        converted["metadata"] = extract_ard_metadata(sample)
        return converted

    def convert_dataset(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """批量转换（逐条独立；任一条非法即整批失败并指出 index）。

        Raises:
            ArdContractError: 第 ``i`` 条样本非法（错误信息带 index）。
        """
        if not isinstance(samples, list):
            raise ArdContractError(f"samples must be a list, got {type(samples).__name__}")
        converted: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            try:
                converted.append(self.convert_sample(sample))
            except ArdContractError as exc:
                raise ArdContractError(f"samples[{index}]: {exc}") from exc
        return converted

    def to_ms_swift_record(self, sample: dict[str, Any], *, response: str) -> dict[str, Any]:
        """产出 ms-swift instruction-tuning 形态的纯数据载荷。

        ms-swift 的 SFT 数据是 ``{"messages": [...]}`` 或
        ``{"query": str, "response": str}``；``query`` 由调用方（持有
        tokenizer/processor 的设施层）渲染 chat template 得到，本方法只负责
        组装记录形状并做字段守恒校验。

        Args:
            sample: ARD 记录（原始输入即可）。
            response: 模型应生成的文本（由 ``build_sft_target_text`` 或
                ms-swift 侧渲染得到）。

        Returns:
            ``{"messages": [...], "response": str, **ard_metadata}``。

        Note:
            E1 只保证"形状可用 + 字段不丢"；真实 ms-swift 数据集对象
            （``AutoPreprocessor`` 消费）的对接属 E2。
        """
        converted = self.convert_sample(sample)
        if not isinstance(response, str) or not response.strip():
            raise ArdContractError("response must be a non-empty string")
        record: dict[str, Any] = {
            "messages": converted["messages"],
            "response": response,
        }
        record.update(converted["metadata"])
        return record


class MsSwiftToGraspoAdapter:
    """graspo 合法样本 / ms-swift 记录 → ARD v3 记录（``GraspoToMsSwiftAdapter`` 的逆）。

    接受两种输入形状（都还原为 ARD ``output.content`` 的纯文本 str）：

    1. **graspo 样本**：``targets[i].output.content`` 为
       ``{"text": str}``（``GraspoToMsSwiftAdapter`` 的产物，可逐字符还原）；
    2. **ms-swift 记录**：带顶层 ``response``（str）+ ``messages``。

    顶层路由字段从 ``metadata``（或输入的顶层同名键）取回。
    """

    def __init__(self) -> None:
        pass

    def convert_sample(self, sample: Any) -> dict[str, Any]:
        """转换单条样本为 ARD v3 记录。

        Args:
            sample: ``core.schema.Sample``、graspo 样本 dict，或带 ``response``
                的 ms-swift 记录 dict。

        Returns:
            ARD v3 记录：``{**ARD_TOP_LEVEL_FIELDS, "messages", "targets"}``。

        Raises:
            ArdContractError: 缺失 ``messages``、无法还原纯文本 content、
                或 ``response`` 为空串。
        """
        if hasattr(sample, "model_dump"):
            record: dict[str, Any] = copy.deepcopy(sample.model_dump())
        elif isinstance(sample, dict):
            record = copy.deepcopy(sample)
        else:
            raise ArdContractError(f"sample must be a Sample or dict, got {type(sample).__name__}")

        if "messages" not in record:
            raise ArdContractError("sample must contain 'messages'")

        # ms-swift 形态（顶层 response）→ 退化为单 target 的 graspo 形状
        if "response" in record:
            response = record.pop("response")
            if not isinstance(response, str) or not response.strip():
                raise ArdContractError("sample.response must be a non-empty string")
            record["targets"] = [
                {"id": None, "output": {"content": {"text": response}}}
            ]

        if "targets" not in record:
            raise ArdContractError("sample must contain 'targets' (or a top-level 'response')")

        ard_record = graspo_sample_to_ard(record)
        # 回流场景：输入顶层显式同名键优先于 metadata 中的副本
        for key in ARD_TOP_LEVEL_FIELDS:
            if key in record:
                ard_record[key] = copy.deepcopy(record[key])
        return ard_record

    def convert_dataset(self, samples: list[Any]) -> list[dict[str, Any]]:
        """批量转换（逐条独立；任一条非法即整批失败并指出 index）。"""
        if not isinstance(samples, list):
            raise ArdContractError(f"samples must be a list, got {type(samples).__name__}")
        converted: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            try:
                converted.append(self.convert_sample(sample))
            except ArdContractError as exc:
                raise ArdContractError(f"samples[{index}]: {exc}") from exc
        return converted
