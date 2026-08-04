"""多模态 rows 的纯数据操作：构建、读取、写入。

本模块是 ripple 算法层的纯函数集——零设施依赖，输入输出都是普通
dict/list，可在单线程本地测试。职责边界见 ``__init__.py``。

核心契约（防呆设计）：
- ``attach_rows`` 是 **唯一** 的写入方：多模态生成完成后必须调用它把
  rows 挂到 metadata 上，否则后续训练拿不到图像。
- ``rows_from_metadata`` 是读取方：只读不写。读取不到时返回 ``[]``
  由调用方决定是告警还是报错（``contract.py`` 负责抛错）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

# metadata 中承载多模态行的键名。全项目唯一真相源：
# 写入方只有 attach_rows，读取方有 rows_from_metadata / contract.py。
MULTIMODAL_ROWS_KEY = "_multimodal_rows"


# ---------------------------------------------------------------------------
# rows 构建（纯函数，从 core/data.py 提取）
# ---------------------------------------------------------------------------


def resolve_messages_media_paths(
    messages: list[dict[str, Any]],
    data_dir: str | Path | None,
) -> None:
    """将 messages 中的相对图像/视频路径就地解析为绝对路径。

    与 SFT 侧 resolve_messages_media_paths 对齐：processor 无法识别
    ``../images/...`` 格式的相对路径。
    """
    if data_dir is None:
        return
    base = Path(data_dir)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("image", "image_url"):
                for key in ("image", "path", "url"):
                    path = block.get(key)
                    if isinstance(path, str) and not path.startswith(
                        ("http://", "https://", "/", "data:")
                    ):
                        block[key] = str((base / path).resolve())
            elif block.get("type") in ("video", "video_url"):
                for key in ("video", "path", "url"):
                    path = block.get(key)
                    if isinstance(path, str) and not path.startswith(
                        ("http://", "https://", "/", "data:")
                    ):
                        block[key] = str((base / path).resolve())


def media_counts(media: list[dict[str, Any]]) -> dict[str, int]:
    """统计多模态媒体类型计数（纯数据转换）。"""
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown") if isinstance(item, dict) else "unknown"
        counts[media_type] = counts.get(media_type, 0) + 1
    return counts


def multimodal_row_from_sample(
    sample: Any, *, data_dir: str | Path | None = None
) -> dict[str, Any]:
    """从 Sample 构建多模态行数据，解析媒体相对路径为绝对路径。"""
    messages = [dict(message) for message in sample.messages]
    if data_dir is not None:
        resolve_messages_media_paths(messages, data_dir)
    row: dict[str, Any] = {
        "messages": messages,
        "media": media_counts(sample.media or []),
    }
    tools = getattr(sample, "tools", None)
    if tools is not None:
        row["tools"] = [dict(tool) for tool in tools]
    return row


# ---------------------------------------------------------------------------
# metadata 读写（唯一真相源）
# ---------------------------------------------------------------------------


def attach_rows(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把多模态 rows 挂到 metadata 上（**唯一的写入方**）。

    :param metadata: 生成/训练使用的 metadata dict（会被就地修改并返回）
    :param rows: 多模态行列表，每行对应一条 completion/experience
    :return: 同一个 metadata（就地修改，方便链式调用）
    """
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
    if not isinstance(rows, list):
        raise TypeError(f"rows must be a list, got {type(rows).__name__}")
    # 深拷贝：attach 后调用方修改 rows 不得污染已入库的 metadata（单一真相源）
    metadata[MULTIMODAL_ROWS_KEY] = [copy.deepcopy(row) for row in rows]
    return metadata


def rows_from_metadata(metadata: Any | None, *, expected_rows: int) -> list[dict[str, Any]]:
    """从 metadata 读取多模态 rows（**只读**，缺键返回空列表）。

    :param metadata: 生成 metadata（dict）、experience metadata 列表（list[dict]）或 None
    :param expected_rows: 期望的 rows 数量（用于校验和单行扩展）
    :return: rows 列表；metadata 为空或缺少该键时返回 ``[]``

    注意：本函数不抛"缺图"错误——那由 ``contract.py`` 的防线负责。
    这里只做结构解析，返回空列表让上层决定语义。
    """
    if metadata is None:
        return []
    if isinstance(metadata, list):
        rows: list[dict[str, Any]] = []
        for item in metadata:
            if isinstance(item, dict):
                rows.extend(rows_from_metadata(item, expected_rows=1))
        if not rows:
            return []
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} multimodal metadata rows, got {len(rows)}"
            )
        return rows
    if not isinstance(metadata, dict):
        return []
    multimodal_rows = metadata.get(MULTIMODAL_ROWS_KEY)
    if multimodal_rows is None:
        return []
    if not isinstance(multimodal_rows, list):
        raise RuntimeError(f"metadata[{MULTIMODAL_ROWS_KEY!r}] must be a list")
    if len(multimodal_rows) == 1 and expected_rows > 1:
        return [dict(multimodal_rows[0]) for _ in range(expected_rows)]
    if len(multimodal_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} multimodal metadata rows, got {len(multimodal_rows)}"
        )
    return [dict(row) for row in multimodal_rows]
