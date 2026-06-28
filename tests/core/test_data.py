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


# ---- assistant tool_calls validation ----


def test_assistant_with_valid_tool_calls_is_accepted(tmp_path):
    path = tmp_path / "valid_tool_call_asst.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"pick up"},'
        '{"role":"assistant","content":"extending","tool_calls":['
        '{"name":"robot_atomic_control","arguments":{"action_type":"伸长手臂","distance_cm":12.5}}'
        "]},"
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"robot_atomic_control","arguments":{"action_type":"降低","distance_cm":5}}]}}]}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]
    # Assistant message is preserved with tool_calls intact.
    asst = sample.messages[1]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"] == [
        {
            "name": "robot_atomic_control",
            "arguments": {"action_type": "伸长手臂", "distance_cm": 12.5},
        }
    ]


def test_assistant_with_raw_xml_in_content_is_rejected(tmp_path):
    path = tmp_path / "raw_xml_asst.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"pick up"},'
        '{"role":"assistant","content":"<tool_call>\\n<function=robot_atomic_control>\\n'
        '<parameter=action_type>\\n伸长手臂\\n</parameter>\\n</function>\\n</tool_call>"},'
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"robot_atomic_control","arguments":{"action_type":"降低","distance_cm":5}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw tool-call text"):
        load_jsonl(path)


def test_assistant_with_raw_function_marker_in_content_is_rejected(tmp_path):
    path = tmp_path / "raw_func_asst.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"q"},'
        '{"role":"assistant","content":"use <function=move> to proceed"},'
        '{"role":"user","content":"ok"}'
        '],"targets":[{"output":{"content":{"a":1}}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw tool-call text"):
        load_jsonl(path)


def test_assistant_tool_calls_non_list_is_rejected(tmp_path):
    path = tmp_path / "bad_tc_type.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"q"},'
        '{"role":"assistant","content":"ok","tool_calls":{"name":"x","arguments":{}}},'
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"x","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tool_calls.*non-empty list"):
        load_jsonl(path)


def test_assistant_tool_calls_empty_list_is_rejected(tmp_path):
    path = tmp_path / "empty_tc.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"q"},'
        '{"role":"assistant","content":"ok","tool_calls":[]},'
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"x","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tool_calls.*non-empty list"):
        load_jsonl(path)


def test_assistant_tool_calls_missing_name_is_rejected(tmp_path):
    path = tmp_path / "no_name_tc.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"q"},'
        '{"role":"assistant","content":"ok","tool_calls":[{"arguments":{}}]},'
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"x","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="\.name must be"):
        load_jsonl(path)


def test_assistant_tool_calls_missing_arguments_is_rejected(tmp_path):
    path = tmp_path / "no_args_tc.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"q"},'
        '{"role":"assistant","content":"ok","tool_calls":[{"name":"move"}]},'
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"move","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="\.arguments must be"):
        load_jsonl(path)


def test_assistant_tool_calls_arguments_with_raw_xml_is_rejected(tmp_path):
    path = tmp_path / "xml_in_args.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"q"},'
        '{"role":"assistant","content":"ok","tool_calls":['
        '{"name":"move","arguments":{"desc":"use <function=move>"}}'
        "]},"
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"move","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw tool-call markers"):
        load_jsonl(path)


def test_assistant_without_tool_calls_plain_text_is_accepted(tmp_path):
    """Assistant with plain text (no tool-call markers, no tool_calls field) is fine."""
    path = tmp_path / "plain_asst.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"system","content":"You are helpful."},'
        '{"role":"user","content":"what is 2+2?"},'
        '{"role":"assistant","content":"Let me think about this."},'
        '{"role":"user","content":"now answer please"}'
        '],"targets":[{"output":{"content":{"answer":4}}}]}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]
    assert sample.messages[2]["role"] == "assistant"
    assert sample.messages[2]["content"] == "Let me think about this."
    assert "tool_calls" not in sample.messages[2]


def test_assistant_multimodal_content_with_xml_is_rejected(tmp_path):
    """Even in list-type (multimodal) content, raw XML in text blocks is caught."""
    path = tmp_path / "mm_xml_asst.jsonl"
    path.write_text(
        '{"messages":['
        '{"role":"user","content":"pick up"},'
        '{"role":"assistant","content":['
        '{"type":"text","text":"here is the call: <tool_call><function=move>"}'
        "]},"
        '{"role":"user","content":"continue"}'
        '],"targets":[{"output":{"tool_calls":[{"name":"move","arguments":{}}]}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw tool-call text"):
        load_jsonl(path)
