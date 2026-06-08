from pathlib import Path

import pytest

from graspo.core.data import load_jsonl, write_jsonl
from graspo.core.schema import Sample


def test_load_standard_jsonl():
    samples = load_jsonl(Path("data/sample.jsonl"))

    assert len(samples) == 2
    assert samples[0].messages
    assert isinstance(samples[0].ground_truth, dict)


def test_write_and_load_roundtrip(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(
        [Sample(messages=[{"role": "user", "content": "hello"}], ground_truth={"x": 1})], path
    )

    loaded = load_jsonl(path)
    assert loaded[0].messages == [{"role": "user", "content": "hello"}]
    assert loaded[0].ground_truth == {"x": 1}


def test_load_messages_jsonl(tmp_path):
    path = tmp_path / "messages.jsonl"
    path.write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"q1"},'
        '{"role":"assistant","content":"a1"},{"role":"user","content":"q2"}],'
        '"ground_truth":{"a":1}}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]
    assert sample.messages == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert sample.ground_truth == {"a": 1}


def test_load_tools_jsonl(tmp_path):
    path = tmp_path / "tools.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"query device status"}],'
        '"tools":[{"type":"function","function":{"name":"query_device_status",'
        '"parameters":{"type":"object","properties":{"device_id":{"type":"string"}}}}}],'
        '"ground_truth":{"name":"query_device_status","arguments":{"device_id":"OLT-17"}}}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]

    assert sample.tools == [
        {
            "type": "function",
            "function": {
                "name": "query_device_status",
                "parameters": {
                    "type": "object",
                    "properties": {"device_id": {"type": "string"}},
                },
            },
        }
    ]
    assert "tools" not in sample.metadata


def test_non_list_tools_is_rejected(tmp_path):
    path = tmp_path / "bad_tools.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"tools":{},"ground_truth":{}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tools"):
        load_jsonl(path)


def test_json_file_is_not_a_training_format(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        '[{"messages":[{"role":"user","content":"q"}],"ground_truth":{"a":1}}]',
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
        ']}],"ground_truth":{"a":1}}\n',
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
    path.write_text('{"prompt":"q","ground_truth":{}}\n', encoding="utf-8")

    try:
        load_jsonl(path)
    except ValueError as exc:
        assert "messages" in str(exc)
    else:
        raise AssertionError("prompt-only records must be rejected")


def test_top_level_media_fields_are_rejected(tmp_path):
    path = tmp_path / "top_level_media.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"ground_truth":{},"images":["a.png"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="removed input field"):
        load_jsonl(path)


@pytest.mark.parametrize("ground_truth", ['"answer"', '[{"a":1}]'])
def test_non_object_ground_truth_is_rejected(tmp_path, ground_truth):
    path = tmp_path / "bad_gt.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"}],"ground_truth":' + ground_truth + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSON object"):
        load_jsonl(path)


def test_final_assistant_message_is_rejected(tmp_path):
    path = tmp_path / "leaky.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"answer"}],'
        '"ground_truth":{}}\n',
        encoding="utf-8",
    )

    try:
        load_jsonl(path)
    except ValueError as exc:
        assert "final assistant" in str(exc)
    else:
        raise AssertionError("final assistant messages must be rejected")
