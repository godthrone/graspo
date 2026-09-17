"""L2 契约测试：真实 ARD v3 ``anchor_bank.jsonl`` 样例过适配器（方案 §9.2 / 决策 D4）。

不依赖 GPU / 模型：只做纯数据转换与结构断言。

**真实样例来源（只读）**：环境变量 ``GRASPO_ARD_SAMPLE_ROOT`` 指向的 ARD v3
产出目录（默认占位值 ``~/.cache/graspo/ard-samples``，见下方 ``_ARD_ROOT``）。
该目录须含若干 ``<run>/anchor_bank.jsonl``，本测试按 ``_CANDIDATES`` 中的
data_source 逐档取用。

样例目录可用环境变量 ``GRASPO_ARD_SAMPLE_ROOT`` 覆盖。缺失时 **skip + WARNING**
（不是静默通过——宪法 §3.2 透明退路 + 方案 §9.2 通过判据）。
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from graspo.flow.data_io import load_jsonl
from graspo.flow.msswift.adapter import GraspoToMsSwiftAdapter, MsSwiftToGraspoAdapter
from graspo.ripple.parsing.xml import build_sft_target_text

_ARD_ROOT = Path(
    os.environ.get(
        "GRASPO_ARD_SAMPLE_ROOT",
        "~/.cache/graspo/ard-samples",
    )
).expanduser()

#: 真实样例候选（名字 → 相对 anchor_bank.jsonl 的目录），按 data_source 覆盖。
_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("ard_multi", "e2e-118-multi-40"),
    ("ard_text", "b3-cond2-text"),
    ("ard_text", "b3-multi-image"),
    ("ard_text", "smoke-text-topup"),
)

#: ARD 顶层字段，必须完整透传（方案 §9.2 判据 ④）。
_TOP_LEVEL_FIELDS = (
    "id",
    "source",
    "data_source",
    "schema_version",
    "input_generator_model",
    "teacher_id",
)


def _load_real_records() -> list[tuple[str, dict]]:
    """从真实 ARD 产出读取样例记录；返回 [(来源标签, 记录)]。

    每个可用目录最多取 3 行，避免大文件拖慢测试。
    """
    records: list[tuple[str, dict]] = []
    for label, subdir in _CANDIDATES:
        path = _ARD_ROOT / subdir / "anchor_bank.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                records.append((f"{subdir}:{lineno}", json.loads(line)))
                if lineno >= 3:
                    break
    return records


_REAL_RECORDS = _load_real_records()


@pytest.fixture(scope="module")
def real_records() -> list[tuple[str, dict]]:
    if not _REAL_RECORDS:
        warnings.warn(
            f"ARD real sample root not found or empty: {_ARD_ROOT} — L2 contract test skipped. "
            "Set GRASPO_ARD_SAMPLE_ROOT to the ARD outputs directory to run it.",
            UserWarning,
            stacklevel=1,
        )
        pytest.skip(f"no ARD real samples under {_ARD_ROOT}")
    return _REAL_RECORDS


def test_real_samples_are_ard_v3(real_records):
    """前置断言：样例确实是 ARD v3 形态（防止 fixture 漂移成别的东西）。"""
    for label, record in real_records:
        assert record["schema_version"] == "3.0.0", label
        assert record["source"] == "ard", label
        assert record["data_source"] in {"ard_text", "ard_multi"}, label
        assert record["targets"][0]["output"]["content"], label


def test_real_samples_have_str_content_and_reasoning(real_records):
    """ARD 的契约事实：``content`` 是 str，``reasoning`` 是 str|None。"""
    for label, record in real_records:
        output = record["targets"][0]["output"]
        assert isinstance(output["content"], str), label
        assert output.get("reasoning") is None or isinstance(output["reasoning"], str), label


def test_adapter_accepts_every_real_sample(real_records):
    """每一条真实样例都能过适配器（契约不成立就会在这里炸）。"""
    adapter = GraspoToMsSwiftAdapter()

    converted = [adapter.convert_sample(record) for _, record in real_records]

    assert len(converted) == len(real_records)


def test_adapter_output_is_accepted_by_graspo_pipeline(real_records):
    """转换结果必须能通过 graspo 侧的真实校验与 SFT target 构建。

    这是 L2 的核心断言：ARD 的 str content 会让 ``build_sft_target_text`` 返回
    空串并让 SFT 直接报 ``ValueError("primary target has no content or tool_calls")``；
    适配后的 dict 必须产出非空 target text，且原始答案**可从该 text 逐字符还原**
    （graspo 的 content 契约是"fenced JSON"，所以无损性以"解析回 JSON 后相等"为准，
    而不是"原始子串出现在 text 里"——后者会被 JSON 转义（换行→``\\n``）破坏）。
    """
    adapter = GraspoToMsSwiftAdapter()

    for label, record in real_records:
        converted = adapter.convert_sample(record)
        # graspo 侧真实校验（正常化 + Sample 构造）
        sample = _sample_from_record(converted)
        target_text = build_sft_target_text(sample.targets[0]["output"])
        assert target_text, f"{label}: target text must not be empty"

        ard_answer = record["targets"][0]["output"]["content"]
        assert _decode_fenced_json_text(target_text) == ard_answer, (
            f"{label}: ARD answer must be recoverable character-for-character"
        )


def _decode_fenced_json_text(target_text: str) -> str:
    """从 ``build_sft_target_text`` 产出的 fenced JSON 中取回 ``text`` 值。

    这就是"content 归一无损"在 SFT 侧的准确含义：文本经过 JSON 转义后
    仍能逐字符还原（转义不是信息损失）。
    """
    assert target_text.startswith("```json\n") and target_text.endswith("\n```"), target_text[:80]
    payload = json.loads(target_text[len("```json\n") : -len("\n```")])
    return payload["text"]


def _sample_from_record(record: dict):
    """``graspo.ripple.data.sample_from_record``（延迟导入，避免 torch 导入链）。"""
    from graspo.ripple.data import sample_from_record

    return sample_from_record(record)


@pytest.mark.parametrize("field", _TOP_LEVEL_FIELDS)
def test_top_level_fields_pass_through_into_metadata(real_records, field):
    """顶层路由字段完整透传（判据 ④）。"""
    for label, record in real_records:
        if field not in record:
            continue
        metadata = GraspoToMsSwiftAdapter().convert_sample(record)["metadata"]
        assert metadata[field] == record[field], f"{label}: {field}"


def test_anchor_meta_passes_through(real_records):
    """``anchor_meta``（采样维度）透传，含多模态标记。"""
    for label, record in real_records:
        metadata = GraspoToMsSwiftAdapter().convert_sample(record)["metadata"]
        assert metadata["anchor_meta"] == record["anchor_meta"], label


def test_messages_are_unchanged_by_adapter(real_records):
    """``messages`` 原样保留（含多模态 content 块）——适配器不碰题目侧。"""
    for label, record in real_records:
        converted = GraspoToMsSwiftAdapter().convert_sample(record)
        assert converted["messages"] == record["messages"], label


def test_messages_shape_invariants_hold(real_records):
    """``messages`` 形状不变量（判据 ①）：可选 system@[0]、U/UAU、角色交替、末条 user。"""
    for label, record in real_records:
        messages = GraspoToMsSwiftAdapter().convert_sample(record)["messages"]
        roles = [m["role"] for m in messages]

        # 可选 system 只允许在 index 0（ARD 形状：system? + user + (assistant user)*）
        first_dialogue = next(idx for idx, role in enumerate(roles) if role != "system")
        assert first_dialogue == 0 or all(r == "system" for r in roles[:first_dialogue]), label
        assert roles[first_dialogue] == "user", label
        assert roles[-1] == "user", label
        assert "system" not in roles[first_dialogue:], label
        for prev, current in zip(roles, roles[1:]):
            assert prev != current, f"{label}: {roles}"


def test_content_normalisation_is_lossless(real_records):
    """content 归一无损（判据 ②）：往返后与原 str 逐字符相同。"""
    for label, record in real_records:
        converted = GraspoToMsSwiftAdapter().convert_sample(record)
        back = MsSwiftToGraspoAdapter().convert_sample(converted)

        assert back["targets"][0]["output"]["content"] == record["targets"][0]["output"]["content"], label


def test_reasoning_is_not_lost(real_records):
    """reasoning 不丢（判据 ③）：进了 graspo ``output`` 内部且值相同。"""
    for label, record in real_records:
        graspo_sample = GraspoToMsSwiftAdapter().convert_sample(record)
        output = graspo_sample["targets"][0]["output"]

        assert "reasoning" in output, label
        assert output["reasoning"] == record["targets"][0]["output"].get("reasoning"), label

        # 并且经 graspo 真实校验（normalize_targets）后仍然存在
        sample = _sample_from_record(graspo_sample)
        assert sample.targets[0]["output"]["reasoning"] == output["reasoning"], label


def test_ard_multi_samples_keep_multimodal_blocks(real_records):
    """``data_source=ard_multi`` 的样例：多模态 content 块透传，media 被正确识别。"""
    multi = [(label, r) for label, r in real_records if r["data_source"] == "ard_multi"]
    if not multi:
        pytest.skip("no ard_multi real sample available under the sample root")

    for label, record in multi:
        graspo_sample = GraspoToMsSwiftAdapter().convert_sample(record)
        assert any(isinstance(m.get("content"), list) for m in graspo_sample["messages"]), label

        sample = _sample_from_record(graspo_sample)
        assert sample.media, f"{label}: multimodal sample must expose media"


def test_real_samples_load_through_graspo_jsonl_loader(real_records):
    """端到端形状：转换后写成 JSONL，能被 ``flow.data_io.load_jsonl`` 读回。

    这是"ARD 数据真正进得了 graspo 训练入口"的最小证据（不启动训练）。
    """
    import tempfile

    from graspo.flow.data_io import write_jsonl

    adapter = GraspoToMsSwiftAdapter()
    records = [r for _, r in real_records]
    converted = adapter.convert_dataset(records)
    # write_jsonl 消费 Sample 对象；这里直接走 dict → Sample → jsonl → load_jsonl
    samples = [_sample_from_record(record) for record in converted]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "converted.jsonl"
        write_jsonl(samples, path)
        reloaded = load_jsonl(path)

    assert len(reloaded) == len(converted)
    assert build_sft_target_text(reloaded[0].targets[0]["output"])
