"""多模态防呆契约：缺图即抛异常（防线，非退路）。

边界校验：数据在跨边界时必须校验。多模态数据丢失是**静默
吞功能**（坏的退路）——丢了多模态训练就是纯浪费算力。本模块把
"丢图"从无声的 bug 变成启动/训练时的硬失败。

两道防线，分别对应 RL 与 SFT 两条训练路径：

1. ``assert_rl_training_has_multimodal``（RL 路径）
   sequences 含 ``image_token_id`` 但 metadata 中无 rows → RuntimeError。
   在 ``sequence_log_probs`` / ``train_batch`` 的 forward 前调用。

2. ``assert_sft_batch_has_multimodal``（SFT 路径）
   样本含 media 但 batch 中无 ``multimodal_inputs`` → RuntimeError。
   在 SFT collate 后、forward 前调用。

本模块零设施依赖——image_token_id 是 int 配置值，sequences 可以是
tensor 或普通序列（鸭子类型判断 允许的运行时类型判断）。
"""

from __future__ import annotations

from typing import Any

from graspo.ripple.multimodal.rows import MULTIMODAL_ROWS_KEY, rows_from_metadata


def contains_image_tokens(sequences: Any, image_token_id: int) -> bool:
    """判断 sequences 中是否含图像占位 token（纯逻辑，支持 tensor/列表）。

    :param sequences: 单条 token 序列（``torch.Tensor`` 或可迭代的 int）
    :param image_token_id: 图像占位 token id（配置值）
    :return: 至少含一个图像 token 返回 True；sequences 为空/None 返回 False
    """
    if sequences is None:
        return False
    # tensor 类：.item() 存在即视为张量标量接口（运行时能力检测）
    tensor_scan_result: bool | None = None
    if hasattr(sequences, "item") and hasattr(sequences, "__eq__"):
        try:
            eq = sequences == image_token_id
            if hasattr(eq, "any"):
                tensor_scan_result = bool(eq.any().item())
            else:
                tensor_scan_result = bool(eq.item())
        except (RuntimeError, TypeError):
            # 张量类型/形状不兼容：保持 None，显式落入逐元素扫描备用路径
            tensor_scan_result = None
    if tensor_scan_result is not None:
        return tensor_scan_result
    try:
        return image_token_id in sequences
    except TypeError:
        return any(token == image_token_id for token in sequences)


def assert_rl_training_has_multimodal(
    metadata: Any | None,
    sequences: Any,
    image_token_id: int,
    *,
    expected_rows: int,
    context: str = "RL training forward",
) -> None:
    """RL 防线：训练数据含图但 metadata 无 rows → RuntimeError。

    :param metadata: 本次 forward 使用的 metadata（dict / list[dict] / None）
    :param sequences: 本次 forward 的 token 序列（含图像占位 token 即视为"有图"）
    :param image_token_id: 图像占位 token id；None 表示模型不支持多模态
    :param expected_rows: 本次 batch 期望的多模态 rows 数
    :param context: 错误消息中的场景描述（便于定位调用点）

    判定规则：
    - 模型不支持多模态（image_token_id 为 None）→ 不检查（纯文本模型）
    - sequences 不含图像 token → 本次前向是纯文本，放行
    - sequences 含图像 token 且 metadata 有 rows → 放行
    - sequences 含图像 token 但 metadata 无 rows → **RuntimeError（防线）**

    这正是 v13 崩溃的静默路径：``<image>`` 占位 token 存在，但 metadata
    从未携带 ``_multimodal_rows``，模型把图像占位符当普通 token 嵌入，
    训练 19.5 小时无任何告警。本函数把该路径变成硬失败。
    """
    if image_token_id is None:
        return
    if not contains_image_tokens(sequences, image_token_id):
        return
    rows = rows_from_metadata(metadata, expected_rows=expected_rows)
    if rows:
        return
    raise RuntimeError(
        f"{context}: sequences contain image token {image_token_id} "
        f"but metadata has no {MULTIMODAL_ROWS_KEY!r} rows. "
        "Multimodal inputs would be silently dropped during training — "
        "this is a data-flow defect (see ripple/multimodal/contract.py). "
        "Fix the generation-side attach (attach_rows) before training."
    )


def assert_sft_batch_has_multimodal(
    sample_media_present: bool,
    multimodal_inputs: Any,
    *,
    context: str = "SFT training forward",
) -> None:
    """SFT 防线：样本含媒体但 batch 无 multimodal_inputs → RuntimeError。

    :param sample_media_present: 本 batch 的样本是否含图像/视频（
        ``any(sample.media)`` 或 ``deferred_multimodal is not None``）
    :param multimodal_inputs: collate 后 batch 中的多模态输入（None 即丢失）
    :param context: 错误消息中的场景描述

    判定规则：
    - 样本不含媒体 → 放行（纯文本 SFT）
    - 样本含媒体且 multimodal_inputs 非 None → 放行
    - 样本含媒体但 multimodal_inputs 为 None → **RuntimeError（防线）**

    SFT 路径当前走 ``MultimodalDeferred`` → collate 编码 → batch dict，
    不经过 metadata 通道，因此不触发 RL 防线；但"数据有图却走了纯文本
    collate"的场景仍可能静默发生（如混合批漏分派），需要独立防线。
    """
    if not sample_media_present:
        return
    if multimodal_inputs is not None:
        return
    raise RuntimeError(
        f"{context}: samples contain media but batch has no multimodal_inputs. "
        "Multimodal inputs would be silently dropped during SFT training — "
        "this is a data-flow defect (see ripple/multimodal/contract.py)."
    )
