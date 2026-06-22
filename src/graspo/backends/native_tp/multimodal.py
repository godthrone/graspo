from __future__ import annotations

from typing import Any

import torch


def _multimodal_row_from_sample(sample: Any) -> dict[str, Any]:
    row = {
        "messages": [dict(message) for message in sample.messages],
        "media": _media_counts(sample.media or []),
    }
    tools = getattr(sample, "tools", None)
    if tools is not None:
        row["tools"] = [dict(tool) for tool in tools]
    return row


def _messages_from_multimodal_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("multimodal row must contain non-empty messages")
    return [dict(message) for message in messages if isinstance(message, dict)]


def _processor_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = [{"type": "text", "text": content}]
        normalized.append(item)
    return normalized


def _tools_from_multimodal_row(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = row.get("tools")
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("multimodal row tools must be a list")
    return [dict(tool) for tool in tools if isinstance(tool, dict)]


def _normalize_tool_batches(
    tool_batches: list[list[dict[str, Any]] | None] | None,
    expected_len: int,
) -> list[list[dict[str, Any]] | None]:
    if tool_batches is None:
        return [None] * expected_len
    if len(tool_batches) != expected_len:
        raise ValueError(
            f"tool_batches length must match message_batches "
            f"({len(tool_batches)} != {expected_len})"
        )
    return tool_batches


def _tools_for_chat_template(
    tool_batches: list[list[dict[str, Any]] | None],
) -> list[dict[str, Any]] | list[list[dict[str, Any]] | None] | None:
    if not any(tools is not None for tools in tool_batches):
        return None
    first = tool_batches[0]
    if all(tools == first for tools in tool_batches):
        return first
    return tool_batches


def _multimodal_rows_from_metadata(
    metadata: Any | None, *, expected_rows: int
) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        rows: list[dict[str, Any]] = []
        for item in metadata:
            if isinstance(item, dict):
                rows.extend(_multimodal_rows_from_metadata(item, expected_rows=1))
        if not rows:
            return []
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} multimodal metadata rows, got {len(rows)}"
            )
        return rows
    if not isinstance(metadata, dict):
        return []
    multimodal_rows = metadata.get("_multimodal_rows")
    if multimodal_rows is None:
        return []
    if not isinstance(multimodal_rows, list):
        raise RuntimeError("metadata['_multimodal_rows'] must be a list")
    if len(multimodal_rows) == 1 and expected_rows > 1:
        return [dict(multimodal_rows[0]) for _ in range(expected_rows)]
    if len(multimodal_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} multimodal metadata rows, got {len(multimodal_rows)}")
    return [dict(row) for row in multimodal_rows]


def _media_counts(media: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown") if isinstance(item, dict) else "unknown"
        counts[media_type] = counts.get(media_type, 0) + 1
    return counts


def _slice_multimodal_inputs(
    inputs: dict[str, torch.Tensor],
    start: int,
    stop: int,
    *,
    images_per_row: int = 0,
    patches_per_row: int = 0,
    videos_per_row: int = 0,
    video_patches_per_row: int = 0,
) -> dict[str, torch.Tensor]:
    """Slice multimodal_inputs dict to rows [start:stop] batch dimension.

    Assumes all batch rows have identical image/video counts, which holds during
    rollout where the same sample is repeated rollout_group_size times.
    Simple row-sliced keys (input_ids, attention_mask, mm_token_type_ids) are NOT
    included -- only the image/video tensors whose batch dimension differs from
    the input_ids batch dimension are sliced here.

    The caller must separately slice input_ids and attention_mask with [start:stop].
    """
    sliced: dict[str, torch.Tensor] = {}
    if "image_grid_thw" in inputs and images_per_row > 0:
        sliced["image_grid_thw"] = inputs["image_grid_thw"][
            start * images_per_row : stop * images_per_row
        ]
    if "pixel_values" in inputs and patches_per_row > 0:
        sliced["pixel_values"] = inputs["pixel_values"][
            start * patches_per_row : stop * patches_per_row
        ]
    if "video_grid_thw" in inputs and videos_per_row > 0:
        sliced["video_grid_thw"] = inputs["video_grid_thw"][
            start * videos_per_row : stop * videos_per_row
        ]
    if "pixel_values_videos" in inputs and video_patches_per_row > 0:
        sliced["pixel_values_videos"] = inputs["pixel_values_videos"][
            start * video_patches_per_row : stop * video_patches_per_row
        ]
    if "mm_token_type_ids" in inputs:
        sliced["mm_token_type_ids"] = inputs["mm_token_type_ids"][start:stop]
    return sliced


def _slice_multimodal_inputs_offset(
    inputs: dict[str, torch.Tensor],
    row_start: int,
    row_stop: int,
    *,
    image_offsets: torch.Tensor | None = None,
    patch_offsets: torch.Tensor | None = None,
    video_offsets: torch.Tensor | None = None,
    video_patch_offsets: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Slice multimodal_inputs dict to rows [row_start:row_stop) using per-row
    cumulative offset tables.

    Prefer _slice_multimodal_inputs (equal-stride) when all rows have identical
    image counts.  Use this offset-based variant when batching heterogeneous
    samples that have different numbers of images per row.
    """
    sliced: dict[str, torch.Tensor] = {}
    if "image_grid_thw" in inputs and image_offsets is not None:
        i_start = int(image_offsets[row_start].item())
        i_stop = int(image_offsets[row_stop].item())
        if i_stop > i_start:
            sliced["image_grid_thw"] = inputs["image_grid_thw"][i_start:i_stop]
    if "pixel_values" in inputs and patch_offsets is not None:
        p_start = int(patch_offsets[row_start].item())
        p_stop = int(patch_offsets[row_stop].item())
        if p_stop > p_start:
            sliced["pixel_values"] = inputs["pixel_values"][p_start:p_stop]
    if "video_grid_thw" in inputs and video_offsets is not None:
        v_start = int(video_offsets[row_start].item())
        v_stop = int(video_offsets[row_stop].item())
        if v_stop > v_start:
            sliced["video_grid_thw"] = inputs["video_grid_thw"][v_start:v_stop]
    if "pixel_values_videos" in inputs and video_patch_offsets is not None:
        vp_start = int(video_patch_offsets[row_start].item())
        vp_stop = int(video_patch_offsets[row_stop].item())
        if vp_stop > vp_start:
            sliced["pixel_values_videos"] = inputs["pixel_values_videos"][vp_start:vp_stop]
    if "mm_token_type_ids" in inputs:
        sliced["mm_token_type_ids"] = inputs["mm_token_type_ids"][row_start:row_stop]
    return sliced


def _compute_multimodal_offset_tables(
    *,
    per_sample_image_counts: list[int],
    rollout_group_size: int,
    image_grid_thw: torch.Tensor | None,
    pixel_values: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-row cumulative offset tables for heterogeneous multimodal rows.

    Given per-sample image counts (one per original sample, NOT per row),
    this expands each sample's count across rollout_group_size identical rows
    and returns cumulative offset tensors for direct use in
    _slice_multimodal_inputs_offset.

    The encoded image_grid_thw and pixel_values tensors are consumed
    sequentially in document order (row by row, left to right).
    Since all rows for a given sample are identical copies with the same
    image count, the per-sample counts suffice to construct the full table.
    """
    total_rows = len(per_sample_image_counts) * rollout_group_size
    image_offsets = torch.zeros(total_rows + 1, dtype=torch.int64)
    patch_offsets = torch.zeros(total_rows + 1, dtype=torch.int64)
    video_offsets = torch.zeros(total_rows + 1, dtype=torch.int64)
    video_patch_offsets = torch.zeros(total_rows + 1, dtype=torch.int64)

    # For per-sample video counts, default to 0 since currently blocked.
    per_sample_video_counts = [0] * len(per_sample_image_counts)

    # Build per-row image/patch counts.
    # image_grid_thw has shape (total_images, 3).  Patches per image = H * W.
    grid_rows = int(image_grid_thw.shape[0]) if image_grid_thw is not None else 0
    pv_rows = int(pixel_values.shape[0]) if pixel_values is not None else 0

    img_cursor = 0
    patch_cursor = 0
    vid_cursor = 0
    vid_patch_cursor = 0

    for sample_idx, (n_img, n_vid) in enumerate(
        zip(per_sample_image_counts, per_sample_video_counts)
    ):
        # Compute patch count for this sample from image_grid_thw
        sample_patches = 0
        assert image_grid_thw is not None
        for j in range(n_img):
            if img_cursor + j < grid_rows:
                h = int(image_grid_thw[img_cursor + j, 1].item())
                w = int(image_grid_thw[img_cursor + j, 2].item())
                sample_patches += h * w
        # Repeat for all rollout_group_size identical rows of this sample
        for _r in range(rollout_group_size):
            img_cursor += n_img
            patch_cursor += sample_patches
            vid_cursor += n_vid
            vid_patch_cursor += 0  # video patches (not yet supported)
            row = sample_idx * rollout_group_size + _r + 1
            image_offsets[row] = img_cursor
            patch_offsets[row] = patch_cursor
            video_offsets[row] = vid_cursor
            video_patch_offsets[row] = vid_patch_cursor

    # Verify total counts match encoded tensors
    if grid_rows > 0:
        assert img_cursor == grid_rows, (
            f"Offset table image count {img_cursor} != actual {grid_rows}"
        )
    if pv_rows > 0:
        assert patch_cursor == pv_rows, (
            f"Offset table patch count {patch_cursor} != actual {pv_rows}"
        )

    return image_offsets, patch_offsets, video_offsets, video_patch_offsets
