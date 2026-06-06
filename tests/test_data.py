from pathlib import Path

from graspo.core.data import load_json, load_jsonl, write_jsonl
from graspo.core.schema import Sample


def test_load_standard_jsonl():
    samples = load_jsonl(Path("data/sample.jsonl"))

    assert len(samples) == 2
    assert samples[0].prompt
    assert isinstance(samples[0].ground_truth, dict)


def test_write_and_load_roundtrip(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl([Sample(prompt="hello", ground_truth={"x": 1})], path)

    loaded = load_jsonl(path)
    assert loaded[0].prompt == "hello"
    assert loaded[0].ground_truth == {"x": 1}


def test_load_messages_jsonl(tmp_path):
    path = tmp_path / "messages.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"{\\"a\\":1}"}]}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]
    assert sample.prompt == "q"
    assert sample.ground_truth == '{"a":1}'


def test_load_json_list(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"prompt":"q","ground_truth":{"a":1}}]', encoding="utf-8")

    sample = load_json(path)[0]
    assert sample.prompt == "q"
    assert sample.ground_truth == {"a": 1}


def test_load_multimodal_messages_jsonl(tmp_path):
    path = tmp_path / "mm.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":['
        '{"type":"text","text":"describe"},'
        '{"type":"image","image":"images/a.png"},'
        '{"type":"video","video":"videos/a.mp4"}'
        ']},{"role":"assistant","content":"{\\"a\\":1}"}]}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]

    assert sample.prompt == "describe\n<image>\n<video>"
    assert sample.media == [
        {"type": "image", "path": "images/a.png"},
        {"type": "video", "path": "videos/a.mp4"},
    ]


def test_load_top_level_multimodal_fields(tmp_path):
    path = tmp_path / "mm_fields.jsonl"
    path.write_text(
        '{"prompt":"q","ground_truth":{},"images":["a.png","b.png"],"video":"c.mp4"}\n',
        encoding="utf-8",
    )

    sample = load_jsonl(path)[0]

    assert sample.media == [
        {"type": "video", "path": "c.mp4"},
        {"type": "image", "path": "a.png"},
        {"type": "image", "path": "b.png"},
    ]
