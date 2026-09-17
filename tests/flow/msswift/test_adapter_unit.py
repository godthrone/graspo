"""L1 单元测试：ARD ↔ graspo 契约纯转换（不依赖 GPU / 网络 / 文件）。

方案 §9.1：断言 ARD ``content: str`` → 目标结构、``reasoning`` 保留、
非法样本被拒绝。全部输入用内存对象构造（宪法 §1.3 计算与设施分离）。

被测对象：``graspo.flow.msswift._ard_contract``（纯计算原语）与
``graspo.flow.msswift.adapter``（适配器类，同样不碰 IO）。
"""

from __future__ import annotations

import copy

import pytest

from graspo.flow.msswift._ard_contract import (
    GRASPO_TEXT_CONTENT_KEY,
    OUTPUT_REASONING_KEY,
    ArdContractError,
    ard_content_to_graspo,
    graspo_content_to_ard,
    validate_ard_messages,
)
from graspo.flow.msswift.adapter import GraspoToMsSwiftAdapter, MsSwiftToGraspoAdapter


def _ard_record() -> dict:
    """最小合法 ARD v3 记录（结构照 ARD ``bank.py`` 输出，内容为占位）。"""
    return {
        "id": "anchor_" + "0" * 16,
        "source": "ard",
        "data_source": "ard_text",
        "schema_version": "3.0.0",
        "messages": [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Explain X briefly."},
        ],
        "targets": [
            {
                "id": "primary",
                "output": {"content": "## Answer\n\nBecause of **reasons**.", "reasoning": "Hmm."},
            }
        ],
        "anchor_meta": {"language": "English", "has_image": False},
        "input_generator_model": "placeholder-generator",
        "teacher_id": "placeholder-teacher",
    }


# ── 方向 1：content str → dict（核心断言：ARD str → 目标结构）───────────────


def test_ard_content_str_becomes_json_object():
    """ARD 的 str content 变成 graspo 要求的 dict。"""
    converted = ard_content_to_graspo("plain answer")

    assert isinstance(converted, dict)
    assert converted == {GRASPO_TEXT_CONTENT_KEY: "plain answer"}


def test_ard_content_round_trip_is_character_exact():
    """str → dict → str 逐字符无损（含 markdown / 换行 / 非 ASCII / 引号）。"""
    original = '# 标题\n\n- 项目 "A"\n- 项目 B\n\n```json\n{"k": 1}\n```\n'

    assert graspo_content_to_ard(ard_content_to_graspo(original)) == original


def test_ard_content_rejects_non_string():
    """非 str 的 content 被拒绝（不静默强转）。"""
    with pytest.raises(ArdContractError, match="must be a string"):
        ard_content_to_graspo({"already": "a dict"})


def test_ard_content_rejects_empty_and_whitespace():
    """空串 / 纯空白被拒绝（SFT 会因空 target 失败，边界上就拦住）。"""
    with pytest.raises(ArdContractError, match="non-empty"):
        ard_content_to_graspo("")
    with pytest.raises(ArdContractError, match="non-empty"):
        ard_content_to_graspo("   \n\t ")


def test_graspo_content_to_ard_rejects_structural_content():
    """结构性 dict（多键）无法还原为 ARD 纯文本 → 拒绝而非猜测。"""
    with pytest.raises(ArdContractError, match="cannot be reduced to plain text"):
        graspo_content_to_ard({"name": "Alice", "age": 30})

    with pytest.raises(ArdContractError, match="cannot be reduced to plain text"):
        graspo_content_to_ard({"text": "a", "extra": "b"})


# ── reasoning 保留 ─────────────────────────────────────────────────────────


def test_reasoning_is_preserved_inside_output():
    """``reasoning`` 保留在 ``output`` 之内（放外面会被 normalize 静默丢弃）。"""
    converted = GraspoToMsSwiftAdapter().convert_sample(_ard_record())

    output = converted["targets"][0]["output"]
    assert output[OUTPUT_REASONING_KEY] == "Hmm."
    assert output["content"] == {GRASPO_TEXT_CONTENT_KEY: "## Answer\n\nBecause of **reasons**."}


def test_reasoning_null_is_preserved_as_none():
    """ARD 的 ``reasoning: null`` 保留为 None（None 是唯一合法空值，§2.2）。"""
    record = _ard_record()
    record["targets"][0]["output"][OUTPUT_REASONING_KEY] = None

    output = GraspoToMsSwiftAdapter().convert_sample(record)["targets"][0]["output"]

    assert OUTPUT_REASONING_KEY in output
    assert output[OUTPUT_REASONING_KEY] is None


def test_reasoning_absent_stays_absent():
    """ARD 没给 reasoning 时不发明该键。"""
    record = _ard_record()
    del record["targets"][0]["output"][OUTPUT_REASONING_KEY]

    output = GraspoToMsSwiftAdapter().convert_sample(record)["targets"][0]["output"]

    assert OUTPUT_REASONING_KEY not in output


def test_reasoning_rejects_non_string():
    record = _ard_record()
    record["targets"][0]["output"][OUTPUT_REASONING_KEY] = 123

    with pytest.raises(ArdContractError, match="must be a string or null"):
        GraspoToMsSwiftAdapter().convert_sample(record)


# ── 非法样本被拒绝 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("不是 dict", lambda r: ["not", "a", "record"]),
        ("缺 targets", lambda r: {k: v for k, v in r.items() if k != "targets"}),
        ("空 targets", lambda r: {**r, "targets": []}),
        ("缺 messages", lambda r: {k: v for k, v in r.items() if k != "messages"}),
        ("空 messages", lambda r: {**r, "messages": []}),
        ("target 缺 output", lambda r: {**r, "targets": [{"id": "primary"}]}),
        ("output 非 dict", lambda r: {**r, "targets": [{"id": "p", "output": "text"}]}),
        ("output 缺 content", lambda r: {**r, "targets": [{"id": "p", "output": {}}]}),
        ("output 有未识别键", lambda r: {**r, "targets": [{"id": "p", "output": {"content": "x", "junk": 1}}]}),
        ("target id 非 str", lambda r: {**r, "targets": [{"id": 7, "output": {"content": "x"}}]}),
        ("末条是 assistant", lambda r: {**r, "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}),
        ("首条不是 user", lambda r: {**r, "messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]}),
        ("角色不交替", lambda r: {**r, "messages": [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}]}),
        ("system 不在 index 0", lambda r: {**r, "messages": [{"role": "user", "content": "q1"}, {"role": "system", "content": "s"}, {"role": "user", "content": "q2"}]}),
        ("message 缺 content", lambda r: {**r, "messages": [{"role": "user"}]}),
        ("message role 为空", lambda r: {**r, "messages": [{"role": "", "content": "q"}]}),
    ],
)
def test_invalid_samples_are_rejected(label, mutate):
    """非法样本在边界上被拒绝（宪法 §2.3 边界校验即防呆）。"""
    sample = mutate(copy.deepcopy(_ard_record()))

    with pytest.raises(ArdContractError):
        GraspoToMsSwiftAdapter().convert_sample(sample)  # type: ignore[arg-type]


def test_invalid_sample_is_not_silently_repaired():
    """拒绝 ≠ 修补：转换失败后原始输入不被就地修改（无副作用）。"""
    record = _ard_record()
    record["targets"][0]["output"]["content"] = ""
    before = copy.deepcopy(record)

    with pytest.raises(ArdContractError):
        GraspoToMsSwiftAdapter().convert_sample(record)

    assert record == before


def test_message_shape_u_and_uau_both_accepted():
    """合法形状 ``U`` 与 ``UAU`` 都通过；``UAU`` 末条仍是 user。"""
    u_only = _ard_record()
    u_only["messages"] = [{"role": "user", "content": "q"}]
    assert len(GraspoToMsSwiftAdapter().convert_sample(u_only)["messages"]) == 1

    uau = _ard_record()
    uau["messages"] = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert len(GraspoToMsSwiftAdapter().convert_sample(uau)["messages"]) == 3


def test_validate_messages_returns_deep_copy():
    """校验返回深拷贝，调用方改动不污染输入。"""
    messages = [{"role": "user", "content": "q"}]
    cleaned = validate_ard_messages(messages)

    cleaned[0]["content"] = "mutated"

    assert messages[0]["content"] == "q"


# ── 批量转换 ───────────────────────────────────────────────────────────────


def test_convert_dataset_reports_index_of_bad_sample():
    """批量转换失败时错误信息带 index（定位到行）。"""
    adapter = GraspoToMsSwiftAdapter()
    good = _ard_record()
    bad = _ard_record()
    bad["targets"][0]["output"]["content"] = ""

    with pytest.raises(ArdContractError, match=r"samples\[1\]"):
        adapter.convert_dataset([good, bad])


def test_convert_dataset_preserves_order_and_count():
    adapter = GraspoToMsSwiftAdapter()
    records = [_ard_record() for _ in range(3)]
    records[0]["id"] = "anchor_first"
    records[2]["id"] = "anchor_third"

    converted = adapter.convert_dataset(records)

    assert [c["metadata"]["id"] for c in converted] == [
        "anchor_first",
        records[1]["id"],
        "anchor_third",
    ]


# ── 方向 2：graspo → ARD（往返）────────────────────────────────────────────


def test_full_round_trip_is_lossless():
    """ARD → graspo → ARD 逐字段还原（共享基座双向可逆）。"""
    original = _ard_record()

    graspo_sample = GraspoToMsSwiftAdapter().convert_sample(original)
    back = MsSwiftToGraspoAdapter().convert_sample(graspo_sample)

    assert back["targets"] == original["targets"]
    assert back["messages"] == original["messages"]
    for key in ("id", "source", "data_source", "schema_version", "input_generator_model", "teacher_id"):
        assert back[key] == original[key], key


def test_reverse_adapter_accepts_ms_swift_response_record():
    """ms-swift 形态（顶层 response）可还原为 ARD targets。"""
    record = {
        "messages": [{"role": "user", "content": "q"}],
        "response": "the answer text",
    }

    ard = MsSwiftToGraspoAdapter().convert_sample(record)

    assert ard["targets"] == [{"id": None, "output": {"content": "the answer text"}}]


def test_reverse_adapter_rejects_empty_response():
    with pytest.raises(ArdContractError, match="non-empty string"):
        MsSwiftToGraspoAdapter().convert_sample(
            {"messages": [{"role": "user", "content": "q"}], "response": ""}
        )


def test_reverse_adapter_rejects_non_sample_types():
    with pytest.raises(ArdContractError, match="Sample or dict"):
        MsSwiftToGraspoAdapter().convert_sample("not a sample")  # type: ignore[arg-type]
