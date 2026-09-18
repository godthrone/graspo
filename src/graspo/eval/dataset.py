"""ELAM V5 数据集读取与图像重叠检测。

**职责**：把 ELAM V5 的 JSONL 样本读成结构化对象；解析图像相对路径；
计算数据集内容哈希；检测"测试样本图像集是训练样本子集"的数据泄漏。

**本文件不负责**：编码图像为 base64（`vllm_client.py`）、判定对错
（`criteria.py`）、聚合（`evaluate.py`）。这里只做数据搬运与统计。

**数据实物形态（实测确证）**

- ``<root>/data/train.jsonl``（6378 行）、``<root>/data/test.jsonl``（702 行）。
- 每行：``{"messages": [...], "tools": [...], "targets": [...]}``，**无 ``id`` 字段**。
- ``messages[].content`` 是 list，元素为 ``{"type": "image", "image": "../images/x.jpg"}``
  或 ``{"type": "text", "text": "..."}``；纯文本消息的 ``content`` 是 str。
- 图像路径**相对于数据文件父目录**（``data/``）解析——``../images/`` 因此指向
  ``<root>/images/``。这一点与 v3 脚本 ``DATA_DIR = Path(DATA_PATH).parent`` 一致。

**图像重叠（必须显式标注的已知数据问题，不得静默忽略）**

实测：样本 id 交集为 0，但去重后训练集与测试集有 **752 张**相同图像，占测试图像
的 **38.5%**；**169/702** 个测试样本的图像集是某个训练样本图像集的**子集**。
这会让测试准确率带上乐观偏差（模型可能记住了这些图）。

本模块提供两个东西让这个偏差**可见、可切换**：

1. :func:`overlap_stats` —— 量化重叠（供写进报告）。
2. :func:`overlap_sample_indices` —— 给出"图像集被训练集完全覆盖"的测试样本下标，
   供评测时按开关剔除后重算第二个准确率（``EvalSummary.accuracy_percent_excluding_overlap``）。

口径默认是**不剔除**（与历史可比）；剔除版只是并排给出的敏感性分析。
谁都不许悄悄用剔除版替换主口径。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

#: 一次哈希读取的块大小（64 KiB），避免把 1.9G 图像目录读进内存。
_HASH_CHUNK = 64 * 1024


class DatasetError(RuntimeError):
    """数据集读取/校验失败。调用方应终止，不要用空数据继续跑。"""


class EvalSample(BaseModel):
    """一条评测样本：原始字段 + 解析出的图像绝对路径。

    ``messages``/``tools``/``targets`` 保持原始形态，因为请求体构造需要逐字段
    还原（不得重新发明格式——还原错了模型行为就变了）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    sample_key: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    targets: list[dict[str, Any]]
    image_paths: list[str]


class OverlapStats(BaseModel):
    """训练/测试图像重叠的量化结果。"""

    model_config = ConfigDict(extra="forbid")

    train_sample_count: int
    test_sample_count: int
    train_image_count_distinct: int
    test_image_count_distinct: int
    #: 两边都出现的去重图像数。
    intersected_image_count: int
    #: 相交图像占测试去重图像的百分比。
    intersected_image_percent_of_test: float
    #: 图像集**完全**被训练集覆盖的测试样本数（真正会带来记忆偏差的那批）。
    test_samples_with_image_set_covered_by_train: int
    #: 上述样本占测试样本的百分比。
    covered_sample_percent_of_test: float


@dataclass(slots=True)
class Dataset:
    """一份已加载的 JSONL 数据集（含路径解析与哈希）。"""

    path: Path
    samples: list[EvalSample]
    sha256: str
    _train_image_identities: set[str] = field(default_factory=set)
    _train_sample_count: int = 0

    @property
    def image_directory(self) -> Path:
        """图像根目录：数据文件父目录的父目录（v5 布局为 ``<root>/images``）。"""
        return self.path.parent.parent

    @property
    def distinct_image_count(self) -> int:
        return len({identity for sample in self.samples for identity in _identities(sample)})


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256（十六进制小写）。数据版本锁定的唯一依据。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: str | Path) -> Dataset:
    """读取 JSONL 数据集，逐行校验必需字段。

    Args:
        path: JSONL 文件路径（如 ``.../v5-balanced/data/test.jsonl``）。

    Returns:
        ``Dataset``。

    Raises:
        DatasetError: 文件不存在、某行 JSON 非法、或缺少 ``messages``/``targets``。
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise DatasetError(f"dataset not found: {data_path}")

    messages_root = data_path.parent
    samples: list[EvalSample] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{data_path}:{line_number} invalid JSON: {exc}") from None
            if "messages" not in raw or "targets" not in raw:
                raise DatasetError(
                    f"{data_path}:{line_number} missing 'messages' or 'targets' "
                    f"(keys: {sorted(raw)})"
                )
            index = len(samples)
            samples.append(
                EvalSample(
                    index=index,
                    # v5 样本无 id 字段 → 用稳定键（文件名 + 行号）代替，
                    # 保证同一份数据两次运行的 sample_key 一致（可复现）。
                    sample_key=f"{data_path.name}#{index}",
                    messages=list(raw["messages"]),
                    tools=raw.get("tools"),
                    targets=list(raw["targets"]),
                    image_paths=_resolve_image_paths(raw["messages"], messages_root),
                )
            )
    if not samples:
        raise DatasetError(f"{data_path} contains no samples")
    return Dataset(path=data_path, samples=samples, sha256=file_sha256(data_path))


def _resolve_image_paths(messages: list[dict[str, Any]], root: Path) -> list[str]:
    """按出现顺序抽出消息里的图像路径，并相对 ``root`` 解析为绝对路径。"""
    resolved: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "image":
                continue
            raw_path = str(item.get("image") or "")
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (root / raw_path).resolve()
            resolved.append(str(candidate))
    return resolved


def _identities(sample: EvalSample) -> set[str]:
    """样本的图像身份集合（用文件名标识一张图；v5 图像名唯一）。"""
    return {Path(image_path).name for image_path in sample.image_paths}


def attach_train_overlap(dataset: Dataset, train_dataset: Dataset) -> Dataset:
    """把训练集图像身份集合挂到测试集数据集上，供重叠检测使用。

    单独一个函数而不是在 :func:`load_dataset` 里做，是因为"测试集评测"通常
    不需要训练集；只有做重叠分析时才加载训练集（加载 train.jsonl 有成本）。
    """
    dataset._train_image_identities = _identity_union(train_dataset)
    dataset._train_sample_count = len(train_dataset.samples)
    return dataset


def overlap_sample_indices(dataset: Dataset) -> set[int]:
    """返回图像集**完全**被训练集覆盖的测试样本下标。

    需要先调用 :func:`attach_train_overlap`；未挂训练集时返回空集（并因此
    使"剔除重叠"口径与主口径相同——这是显式的、可解释的行为，不是静默降级）。
    """
    train_images = dataset._train_image_identities
    if not train_images:
        return set()
    covered: set[int] = set()
    for sample in dataset.samples:
        identities = _identities(sample)
        if identities and identities <= train_images:
            covered.add(sample.index)
    return covered


def overlap_stats(dataset: Dataset) -> OverlapStats:
    """量化训练/测试图像重叠。需要先 :func:`attach_train_overlap`。"""
    test_images = {identity for sample in dataset.samples for identity in _identities(sample)}
    train_images = dataset._train_image_identities
    intersected = test_images & train_images
    covered = overlap_sample_indices(dataset)
    test_count = len(dataset.samples)
    return OverlapStats(
        train_sample_count=dataset._train_sample_count,
        test_sample_count=test_count,
        train_image_count_distinct=len(train_images),
        test_image_count_distinct=len(test_images),
        intersected_image_count=len(intersected),
        intersected_image_percent_of_test=round(100.0 * len(intersected) / len(test_images), 4)
        if test_images
        else 0.0,
        test_samples_with_image_set_covered_by_train=len(covered),
        covered_sample_percent_of_test=round(100.0 * len(covered) / test_count, 4)
        if test_count
        else 0.0,
    )


def _identity_union(dataset: Dataset) -> set[str]:
    return {identity for sample in dataset.samples for identity in _identities(sample)}


def read_image_bytes(image_path: str) -> tuple[str, bytes]:
    """读单张图像，返回 ``(扩展名, bytes)``。

    Raises:
        DatasetError: 文件不存在（不是静默跳过——缺图说明数据不完整，
            跳过会让分母悄悄变小，与历史数字不可比）。
    """
    candidate = Path(image_path)
    if not candidate.is_file():
        raise DatasetError(f"image missing: {candidate}")
    return candidate.suffix.lower().lstrip(".") or "jpeg", candidate.read_bytes()


def iter_sample_image_bytes(sample: EvalSample) -> Iterator[tuple[str, bytes]]:
    """逐张产出样本图像的 ``(扩展名, bytes)``。"""
    for image_path in sample.image_paths:
        yield read_image_bytes(image_path)
