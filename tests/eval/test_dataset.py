"""``graspo.eval.dataset`` 的单测：ELAM V5 读取与图像重叠检测。

重叠检测是**已知数据问题的度量工具**：测试集有 169/702 个样本的图像集是某训练
样本的子集（实测确证）。这些测试锁死度量口径，防止它被悄悄改成"看起来更好看"
的口径。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspo.eval.dataset import (
    DatasetError,
    attach_train_overlap,
    file_sha256,
    load_dataset,
    overlap_sample_indices,
    overlap_stats,
    read_image_bytes,
)


def _write_image(path: Path, payload: bytes = b"\xff\xd8\xff\xe0jpeg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sample(messages, targets, tools=None) -> dict:
    payload = {"messages": messages, "targets": targets}
    if tools is not None:
        payload["tools"] = tools
    return payload


def _image_message(image_refs: list[str], text: str = "捡起电池") -> dict:
    content: list[dict] = [{"type": "image", "image": ref} for ref in image_refs]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


_TARGETS = [
    {"output": {"tool_calls": [{"name": "rotate_arm", "arguments": {"action_type": "左"}}]}}
]


def _build_dataset_root(root: Path, *, test_lines: list[dict], train_lines: list[dict]) -> Path:
    """按 v5-balanced 的布局造数据：<root>/data/*.jsonl + <root>/images/。"""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in test_lines) + "\n",
        encoding="utf-8",
    )
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in train_lines) + "\n",
        encoding="utf-8",
    )
    return data_dir


def test_load_dataset_resolves_image_paths_relative_to_data_dir(tmp_path):
    """图像引用形如 ../images/x.jpg，必须相对数据文件父目录解析（与 v3 一致）。"""
    images = tmp_path / "images"
    _write_image(images / "a.jpg")
    _write_image(images / "b.jpg")
    data_dir = _build_dataset_root(
        tmp_path,
        test_lines=[_sample([_image_message(["../images/a.jpg", "../images/b.jpg"])], _TARGETS)],
        train_lines=[],
    )

    dataset = load_dataset(data_dir / "test.jsonl")
    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.image_paths == [
        str((images / "a.jpg").resolve()),
        str((images / "b.jpg").resolve()),
    ]
    assert sample.image_paths[0].endswith("a.jpg")


def test_load_dataset_reports_sha256_and_counts(tmp_path):
    _write_image(tmp_path / "images" / "a.jpg")
    data_dir = _build_dataset_root(
        tmp_path,
        test_lines=[
            _sample([_image_message(["../images/a.jpg"])], _TARGETS),
            _sample([_image_message(["../images/a.jpg"])], _TARGETS),
        ],
        train_lines=[],
    )
    dataset = load_dataset(data_dir / "test.jsonl")
    assert dataset.sha256 == file_sha256(data_dir / "test.jsonl")
    assert len(dataset.sha256) == 64
    assert len(dataset.samples) == 2
    # 两张样本引用同一张图 → 去重后只有 1 张
    assert dataset.distinct_image_count == 1


def test_sample_keys_are_stable_across_loads(tmp_path):
    """v5 样本无 id 字段；sample_key 必须可复现（同数据两次加载一致）。"""
    _write_image(tmp_path / "images" / "a.jpg")
    data_dir = _build_dataset_root(
        tmp_path,
        test_lines=[_sample([_image_message(["../images/a.jpg"])], _TARGETS)],
        train_lines=[],
    )
    first = load_dataset(data_dir / "test.jsonl").samples[0].sample_key
    second = load_dataset(data_dir / "test.jsonl").samples[0].sample_key
    assert first == second
    assert first.startswith("test.jsonl#")


def test_load_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_load_dataset_rejects_missing_required_fields(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "test.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="missing 'messages' or 'targets'"):
        load_dataset(data_dir / "test.jsonl")


def test_load_dataset_rejects_invalid_json_with_line_number(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "test.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"test\.jsonl:1 invalid JSON"):
        load_dataset(data_dir / "test.jsonl")


def test_load_dataset_rejects_empty_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "test.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="contains no samples"):
        load_dataset(data_dir / "test.jsonl")


def test_read_image_bytes_rejects_missing_image(tmp_path):
    with pytest.raises(DatasetError, match="image missing"):
        read_image_bytes(str(tmp_path / "nope.jpg"))


def test_overlap_detects_covered_sample_subsets(tmp_path):
    """图像集是训练集子集的测试样本必须被识别出来（这是乐观偏差的来源）。"""
    images = tmp_path / "images"
    for name in ("shared_0.jpg", "shared_1.jpg", "unique.jpg", "train_only.jpg"):
        _write_image(images / name)

    data_dir = _build_dataset_root(
        tmp_path,
        test_lines=[
            # 0: 两张图都能在训练集里找到 → 完全覆盖
            _sample(
                [_image_message(["../images/shared_0.jpg", "../images/shared_1.jpg"])],
                _TARGETS,
            ),
            # 1: 含一张训练集没有的图 → 未被覆盖
            _sample(
                [_image_message(["../images/shared_0.jpg", "../images/unique.jpg"])],
                _TARGETS,
            ),
        ],
        train_lines=[
            _sample(
                [_image_message(["../images/shared_0.jpg", "../images/shared_1.jpg"])],
                _TARGETS,
            ),
            _sample([_image_message(["../images/train_only.jpg"])], _TARGETS),
        ],
    )
    test = load_dataset(data_dir / "test.jsonl")
    train = load_dataset(data_dir / "train.jsonl")
    attach_train_overlap(test, train)

    covered = overlap_sample_indices(test)
    assert covered == {0}

    stats = overlap_stats(test)
    assert stats.test_sample_count == 2
    assert stats.train_sample_count == 2
    assert stats.test_image_count_distinct == 3  # shared_0, shared_1, unique
    assert stats.train_image_count_distinct == 3  # shared_0, shared_1, train_only
    assert stats.intersected_image_count == 2  # shared_0, shared_1
    assert stats.test_samples_with_image_set_covered_by_train == 1
    assert stats.covered_sample_percent_of_test == pytest.approx(50.0)


def test_overlap_stats_without_train_attachment_reports_emptiness(tmp_path):
    """未挂训练集时不假装"算过、重叠为零"——返回空集，且是显式可解释的行为。"""
    _write_image(tmp_path / "images" / "a.jpg")
    data_dir = _build_dataset_root(
        tmp_path,
        test_lines=[_sample([_image_message(["../images/a.jpg"])], _TARGETS)],
        train_lines=[],
    )
    dataset = load_dataset(data_dir / "test.jsonl")
    assert overlap_sample_indices(dataset) == set()
    stats = overlap_stats(dataset)
    assert stats.intersected_image_count == 0
    assert stats.train_sample_count == 0
