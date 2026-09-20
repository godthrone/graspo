"""ARD ↔ graspo 数据契约的纯转换原语（计算层，零 IO / 零设施依赖）。

**职责边界**：本模块只做字段级转换与校验，输入输出都是普通 ``dict``/``str``。
文件读取、日志、报告渲染在 ``adapter.py``（适配器）与调用方（CLI/训练器）里。

**为什么独立成文件**：ARD 契约是 graspo ↔ ms-swift ↔ ARD 三方的共享基座
（决策 D4），转换逻辑必须能脱离 ``flow/__init__`` 的 torch 导入链独立单测
（宪法 §1.3 计算与设施分离）。

**契约来源**：``decision-record.md`` D4 —— 同一份 ``anchor_bank.jsonl``
（``messages`` + ``targets[{content, reasoning}]``，无 logprob）同时供 SFT 与
graspo RL 消费。ARD 落盘 ``output.content`` 为 **str**，而 graspo 消费端
（``ripple/reward/normalize.py``）要求 ``output.content`` 为 **dict** ——
本模块是两侧之间唯一的转换口径（单一真相源，宪法 §1.4）。

**ARD 无 logprob 是特性**（OPD 阶段由教师现场给出），适配器不发明也不补造
logprob 字段。
"""

from __future__ import annotations

import copy
from typing import Any

# ── 契约常量（全项目唯一真相源，禁止在别处硬编码同名字符串）────────────────

#: graspo ``targets[i].output`` 中承载纯文本答案的键。
#: ARD 的 ``content`` 是一整段纯文本（markdown 等），无内部结构；graspo 的
#: ``output.content`` 必须是 JSON object。用**唯一一个键**包裹可保证往返无损
#: （字典仅此一个键 ⟺ 它承载原始纯文本），避免"猜哪个键是答案"的隐式约定。
GRASPO_TEXT_CONTENT_KEY = "text"

#: ``output`` 中承载教师推理链的键（ARD ``output.reasoning`` 的落点）。
#: 与 ``content`` 平级且位于 ``output`` **之内**——``_normalize_target`` 只读取
#: ``output`` 内的键，放在 ``output`` 之外会被静默丢弃。
OUTPUT_REASONING_KEY = "reasoning"

#: ARD 顶层字段：必须完整透传到 graspo ``metadata`` 的路由/溯源字段。
ARD_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "id",
    "source",
    "data_source",
    "schema_version",
    "anchor_meta",
    "input_generator_model",
    "teacher_id",
)

#: ARD 已知 ``data_source`` 受控枚举（ARD ``core/types.py``）。
ARD_DATA_SOURCES: tuple[str, ...] = ("ard_text", "ard_multi")


class ArdContractError(ValueError):
    """ARD 样本不符合共享基座契约时抛出（边界校验，宪法 §2.3 防线）。"""


# ── 方向 1：ARD → graspo ───────────────────────────────────────────────────


def ard_content_to_graspo(content: Any, *, path: str = "output.content") -> dict[str, Any]:
    """ARD 的 ``output.content``（str）→ graspo 要求的 dict。

    包裹形状为 ``{"text": <原字符串>}``（``GRASPO_TEXT_CONTENT_KEY``）。
    这是**无损**映射：往返转换（``graspo_content_to_ard``）可逐字符还原。

    Raises:
        ArdContractError: ``content`` 不是非空字符串。
    """
    if not isinstance(content, str):
        raise ArdContractError(
            f"{path} must be a string (ARD v3 emits a plain-text answer), "
            f"got {type(content).__name__}"
        )
    if not content.strip():
        raise ArdContractError(f"{path} must be a non-empty string")
    return {GRASPO_TEXT_CONTENT_KEY: content}


def graspo_content_to_ard(content: Any, *, path: str = "output.content") -> str:
    """graspo 的 ``output.content``（dict）→ ARD 的纯文本 str。方向 1 的逆。

    Raises:
        ArdContractError: ``content`` 不是恰好只含 ``GRASPO_TEXT_CONTENT_KEY``
            的 dict（结构性答案无法还原为 ARD 纯文本，拒绝而非猜测）。
    """
    if not isinstance(content, dict):
        raise ArdContractError(
            f"{path} must be a JSON object (graspo contract), got {type(content).__name__}"
        )
    if list(content.keys()) != [GRASPO_TEXT_CONTENT_KEY]:
        raise ArdContractError(
            f"{path} cannot be reduced to plain text: expected exactly "
            f"{{'{GRASPO_TEXT_CONTENT_KEY}': str}}, got keys {sorted(content.keys())}"
        )
    return _check_text(content[GRASPO_TEXT_CONTENT_KEY], path)


def _check_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArdContractError(f"{path}.{GRASPO_TEXT_CONTENT_KEY} must be a non-empty string")
    return value


def ard_output_to_graspo(output: Any, *, path: str = "output") -> dict[str, Any]:
    """ARD ``targets[i].output`` → graspo ``targets[i].output``。

    - ``content``：str → ``{"text": str}``（无损）
    - ``reasoning``：**原样保留**在 ``OUTPUT_REASONING_KEY``（可为 None，宪法 §2.2
      None 是唯一合法空值）；ARD 字段缺失时不发明该键。
    - 其余键不识别 → 抛错（防呆：契约漂移必须显式暴露，而非静默丢弃）。
    """
    if not isinstance(output, dict):
        raise ArdContractError(f"{path} must be a JSON object, got {type(output).__name__}")
    unknown = sorted(set(output) - {"content", OUTPUT_REASONING_KEY})
    if unknown:
        raise ArdContractError(
            f"{path} has unrecognised key(s) {unknown}; "
            f"ARD contract allows content + optional {OUTPUT_REASONING_KEY}"
        )
    if "content" not in output:
        raise ArdContractError(f"{path} must contain 'content'")

    converted: dict[str, Any] = {
        "content": ard_content_to_graspo(output["content"], path=f"{path}.content")
    }
    if OUTPUT_REASONING_KEY in output:
        reasoning = output[OUTPUT_REASONING_KEY]
        if reasoning is not None and not isinstance(reasoning, str):
            raise ArdContractError(
                f"{path}.{OUTPUT_REASONING_KEY} must be a string or null, "
                f"got {type(reasoning).__name__}"
            )
        converted[OUTPUT_REASONING_KEY] = reasoning
    return converted


def graspo_output_to_ard(output: Any, *, path: str = "output") -> dict[str, Any]:
    """graspo ``output`` → ARD ``output``（方向 1 的逆，逐字符可逆）。"""
    if not isinstance(output, dict):
        raise ArdContractError(f"{path} must be a JSON object, got {type(output).__name__}")
    converted: dict[str, Any] = {
        "content": graspo_content_to_ard(output.get("content"), path=f"{path}.content")
    }
    if OUTPUT_REASONING_KEY in output:
        converted[OUTPUT_REASONING_KEY] = output[OUTPUT_REASONING_KEY]
    return converted


def ard_target_to_graspo(target: Any, *, path: str = "targets[i]") -> dict[str, Any]:
    """ARD ``targets[i]`` → graspo ``targets[i]``（校验 id + 转换 output）。"""
    if not isinstance(target, dict):
        raise ArdContractError(f"{path} must be a JSON object, got {type(target).__name__}")
    unknown = sorted(set(target) - {"id", "output"})
    if unknown:
        raise ArdContractError(f"{path} has unrecognised key(s) {unknown}; allowed: id, output")
    if "output" not in target:
        raise ArdContractError(f"{path} must contain 'output'")
    target_id = target.get("id")
    if target_id is not None and not isinstance(target_id, str):
        raise ArdContractError(f"{path}.id must be a string when provided")
    return {
        "id": target_id,
        "output": ard_output_to_graspo(target["output"], path=f"{path}.output"),
    }


def graspo_target_to_ard(target: Any, *, path: str = "targets[i]") -> dict[str, Any]:
    """graspo ``targets[i]`` → ARD ``targets[i]``（方向 1 的逆）。"""
    if not isinstance(target, dict):
        raise ArdContractError(f"{path} must be a JSON object, got {type(target).__name__}")
    if "output" not in target:
        raise ArdContractError(f"{path} must contain 'output'")
    return {
        "id": target.get("id"),
        "output": graspo_output_to_ard(target["output"], path=f"{path}.output"),
    }


# ── 方向 2：graspo → ARD（记录级）───────────────────────────────────────────


def graspo_sample_to_ard(sample: Any) -> dict[str, Any]:
    """graspo 样本（dict 或 ``Sample``）→ ARD v3 记录。

    ms-swift / graspo 侧训练完回流数据集时使用。``metadata`` 中的
    ``ARD_TOP_LEVEL_FIELDS`` 会被还原为 ARD 顶层字段；``targets`` 走
    ``graspo_target_to_ard``。

    Args:
        sample: graspo 样本，可以是 ``core.schema.Sample`` 或等价 dict，
            需含 ``messages`` 与 ``targets``。

    Raises:
        ArdContractError: 缺 ``messages``/``targets``，或 ``targets`` 为空。
    """
    if hasattr(sample, "model_dump"):
        record = copy.deepcopy(sample.model_dump())
    elif isinstance(sample, dict):
        record = copy.deepcopy(sample)
    else:
        raise ArdContractError(f"sample must be a Sample or dict, got {type(sample).__name__}")

    if not isinstance(record.get("messages"), list) or not record["messages"]:
        raise ArdContractError("sample must contain a non-empty 'messages' list")
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ArdContractError("sample must contain a non-empty 'targets' list")

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ArdContractError("sample.metadata must be a JSON object when provided")

    ard_record: dict[str, Any] = {
        key: copy.deepcopy(metadata[key]) for key in ARD_TOP_LEVEL_FIELDS if key in metadata
    }
    ard_record["messages"] = copy.deepcopy(record["messages"])
    ard_record["targets"] = [
        graspo_target_to_ard(target, path=f"targets[{idx}]") for idx, target in enumerate(targets)
    ]
    if "tools" in record and record["tools"] is not None:
        ard_record["tools"] = copy.deepcopy(record["tools"])
    return ard_record


def extract_ard_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """从 ARD 记录中抽出需透传的顶层字段（浅校验后原样取值）。

    未知顶层键也在返回值内——graspo 的 ``sample_from_record`` 会把
    ``messages``/``targets``/``tools`` 之外的所有键透传为 metadata，
    适配器保持同样口径，不擅自裁剪上游字段。
    """
    if not isinstance(record, dict):
        raise ArdContractError(f"record must be a JSON object, got {type(record).__name__}")
    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in {"messages", "targets", "tools", "metadata"}
    }


# ── 校验（边界防线，宪法 §2.3）─────────────────────────────────────────────


def validate_ard_messages(messages: Any) -> list[dict[str, Any]]:
    """校验 ARD ``messages`` 形状不变量，返回深拷贝。

    不变量（与 ARD ``domain/anchor_shape.py`` 一致）：

    - 非空 list[dict]；``role`` 非空字符串；``content`` 键存在
    - ``system`` 只允许出现在 index 0（至多一条）
    - 首、尾消息均为 ``user``
    - 角色严格交替（``user``/``assistant``/``tool`` 轮转；``tool`` 仅出现在
      ``assistant`` 之后）
    - **末条不是 assistant**（末条是教师答案，不允许出现在 prompt 里）

    Raises:
        ArdContractError: 任一不变量被违反。
    """
    if not isinstance(messages, list) or not messages:
        raise ArdContractError("messages must be a non-empty list")
    cleaned: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ArdContractError(f"messages[{idx}] must be a JSON object")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ArdContractError(
                f"messages[{idx}].role is required and must be a non-empty string"
            )
        if "content" not in message and role != "tool":
            # ``tool`` 消息用 ``tool_call_id`` 定位（OpenAI 标准允许 content 缺省）。
            raise ArdContractError(f"messages[{idx}].content is required")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ArdContractError(
                    f"messages[{idx}] (tool) must have a non-empty 'tool_call_id'"
                )
        if role == "system" and idx != 0:
            raise ArdContractError(f"messages[{idx}]: 'system' role is only allowed at index 0")
        cleaned.append(copy.deepcopy(message))

    # 首个非 system 消息必须是 user（ARD 形状：可选 system@[0] + user 起头）。
    first_dialogue = next(
        (idx for idx, message in enumerate(cleaned) if message["role"] != "system"), None
    )
    if first_dialogue is None:
        raise ArdContractError("messages must contain at least one non-system message")
    if cleaned[first_dialogue]["role"] != "user":
        raise ArdContractError(
            f"messages[{first_dialogue}].role must be 'user' (optional system at index 0), "
            f"got {cleaned[first_dialogue]['role']!r}"
        )
    if cleaned[-1]["role"] != "user":
        raise ArdContractError(
            f"messages[-1].role must be 'user' (final assistant message would leak the target), "
            f"got {cleaned[-1]['role']!r}"
        )

    for idx in range(1, len(cleaned)):
        prev, current = cleaned[idx - 1]["role"], cleaned[idx]["role"]
        if current == "system":
            raise ArdContractError(f"messages[{idx}]: 'system' role is only allowed at index 0")
        if current == prev:
            raise ArdContractError(
                f"messages[{idx}]: roles must alternate strictly; "
                f"{prev!r} cannot be followed by {current!r}"
            )
    return cleaned
