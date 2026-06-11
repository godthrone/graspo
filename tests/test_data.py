from pathlib import Path

import pytest

from graspo.core.data import load_jsonl, write_jsonl
from graspo.core.schema import Sample


def _content_target(content: dict) -> list[dict]:
    return [{"id": "expected", "output": {"content": content}}]


def _tool_target(*calls: dict, target_id: str = "expected") -> list[dict]:
    return [{"id": target_id, "output": {"tool_calls": list(calls)}}]


def test_load_standard_jsonl():
    samples = load_jsonl(Path("data/sample.jsonl"))

    assert len(samples) == 2
    assert samples[0].messages
    assert isinstance(samples[0].targets, list)
    assert samples[0].targets[0]["output"]["content"]["APN"] == "cmnet"


def test_write_and_load_roundtrip(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(
        [
            Sample(
                messages=[{"role": "user", "content": "hello"}], targets=_content_target({"x": 1})
            )
        ],
        path,
    )

    loaded = load_jsonl(path)
    assert loaded[0].messages == [{"role": "user", "content": "hello"}]
    assert loaded[0].targets == _content_target({"x": 1})


def test_load_messages_jsonl(tmp_path):
    path = tmp_path / "messages.jsonl"
    path.write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"q1"},'
        '{"role":"assistant","content":"a1"},{"role":"user","content":"q2"}],'
        '"targets":[{"output":{"content":{"a":1}}}]}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]
    assert sample.messages == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert sample.targets == [{"id": None, "output": {"content": {"a": 1}}}]


def test_load_tools_jsonl():
    sample = load_jsonl(Path("data/sample_tool_call.jsonl"))[0]

    assert sample.expects_tool_calls is True
    assert sample.targets == _tool_target(
        {
            "name": "query_device_status",
            "arguments": {
                "device_id": "OLT-17",
                "panel_time": "2026-06-08T10:30:00+08:00",
            },
        }
    )
    assert "tools" not in sample.metadata


def test_load_tools_jsonl_allows_alternative_targets(tmp_path):
    path = tmp_path / "tools_alternatives.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"move"}],'
        '"tools":[{"type":"function","function":{"name":"move",'
        '"parameters":{"type":"object","properties":{"action":{"type":"string"},'
        '"distance_cm":{"type":"integer"}},"required":["action","distance_cm"]}}}],'
        '"targets":['
        '{"id":"left","output":{"tool_calls":[{"name":"move","arguments":{"action":"left","distance_cm":6}}]}},'
        '{"id":"down","output":{"tool_calls":[{"name":"move","arguments":{"action":"down","distance_cm":4}}]}}'
        "]}\n",
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]

    assert [target["id"] for target in sample.targets] == ["left", "down"]
    assert sample.targets[1]["output"]["tool_calls"][0]["arguments"]["action"] == "down"


def test_load_tools_jsonl_allows_ordered_multi_step_tool_calls(tmp_path):
    path = tmp_path / "tools_multi_step.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"move then inspect"}],'
        '"tools":[{"type":"function","function":{"name":"move","parameters":{"type":"object"}}},'
        '{"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}}],'
        '"targets":[{"output":{"tool_calls":['
        '{"name":"move","arguments":{"action":"left"}},'
        '{"name":"inspect","arguments":{"object":"target"}}'
        "]}}]}\n",
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]

    assert len(sample.targets[0]["output"]["tool_calls"]) == 2


def test_tool_target_unknown_tool_is_rejected(tmp_path):
    path = tmp_path / "bad_tool_name.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],'
        '"tools":[{"type":"function","function":{"name":"known","parameters":{"type":"object"}}}],'
        '"targets":[{"output":{"tool_calls":[{"name":"other","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not declared"):
        load_jsonl(path)


def test_tool_target_enum_value_is_rejected(tmp_path):
    path = tmp_path / "bad_tool_enum.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],'
        '"tools":[{"type":"function","function":{"name":"move","parameters":{"type":"object",'
        '"properties":{"action":{"type":"string","enum":["down"]}},'
        '"required":["action"]}}}],'
        '"targets":[{"output":{"tool_calls":[{"name":"move","arguments":{"action":"up"}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enum"):
        load_jsonl(path)


def test_non_list_tools_is_rejected(tmp_path):
    path = tmp_path / "bad_tools.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"tools":{},'
        '"targets":[{"output":{"content":{}}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tools"):
        load_jsonl(path)


def test_json_file_is_not_a_training_format(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        '[{"messages":[{"role":"user","content":"q"}],"targets":[{"output":{"content":{"a":1}}}]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSONL record"):
        load_jsonl(path)


def test_load_multimodal_messages_jsonl(tmp_path):
    path = tmp_path / "mm.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":['
        '{"type":"text","text":"describe"},'
        '{"type":"image","image":"images/a.png"},'
        '{"type":"video","video":"videos/a.mp4"}'
        ']}],"targets":[{"output":{"content":{"a":1}}}]}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]

    assert sample.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "image": "images/a.png"},
                {"type": "video", "video": "videos/a.mp4"},
            ],
        }
    ]
    assert sample.media == [
        {"type": "image", "path": "images/a.png"},
        {"type": "video", "path": "videos/a.mp4"},
    ]


def test_prompt_only_jsonl_is_rejected(tmp_path):
    path = tmp_path / "prompt_only.jsonl"
    path.write_text('{"prompt":"q","targets":[{"output":{"content":{}}}]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="messages"):
        load_jsonl(path)


def test_top_level_media_fields_are_rejected(tmp_path):
    path = tmp_path / "top_level_media.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"targets":[{"output":{"content":{}}}],'
        '"images":["a.png"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="removed input field"):
        load_jsonl(path)


@pytest.mark.parametrize(
    "targets",
    [
        '"answer"',
        "[]",
        '[{"id":"missing-output"}]',
        '[{"output":{}}]',
        '[{"output":{"content":"answer"}}]',
        '[{"output":{"tool_calls":[]}}]',
    ],
)
def test_invalid_targets_are_rejected(tmp_path, targets):
    path = tmp_path / "bad_targets.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"targets":' + targets + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="targets|output|tool_calls|content"):
        load_jsonl(path)


def test_removed_ground_truth_is_rejected(tmp_path):
    path = tmp_path / "old_gt.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"ground_truth":{"a":1}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ground_truth.*targets"):
        load_jsonl(path)


def test_final_assistant_message_is_rejected(tmp_path):
    path = tmp_path / "leaky.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"answer"}],'
        '"targets":[{"output":{"content":{}}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="final assistant"):
        load_jsonl(path)
