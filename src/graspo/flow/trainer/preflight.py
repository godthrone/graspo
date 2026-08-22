"""多模态训练启动预检（防线）。

在训练启动时（数据含图时）验证视觉链路完整性，把 v13 式的"跑 19.5 小时
才发现视觉 LoRA 从未训练"提前到启动 30 秒内拦截：

1. 数据含图但模型不支持视觉 → 报错（数据/模型不匹配）
2. encode → attach → resolve 契约链路必须完整（resolve 非 None）
3. visual LoRA 参数必须已注册（vision_common target 生效）
4. fake 1-step 前向必须让 visual LoRA 产生非零梯度

预检只验证**消费侧**链路与视觉可训练性；**生成侧**断链
（generation metadata 未 attach rows）由 ``ripple/multimodal/contract``
在训练 forward 时拦截（阶段 3 接线）。
"""

import logging
from typing import Any

from graspo.ripple.multimodal.rows import (
    MULTIMODAL_ROWS_KEY,
    attach_rows,
    multimodal_row_from_sample,
)

_log = logging.getLogger("graspo.preflight")


def assert_data_vision_compatible(
    samples: list[Any],
    *,
    model_supports_vision: bool,
    model_name: str,
) -> None:
    """纯逻辑校验：数据含图时模型必须支持视觉（可单测，不触 GPU）。

    :param samples: 训练样本列表
    :param model_supports_vision: 模型是否具备视觉塔（config.has_vision_config）
    :param model_name: 模型名（错误消息用）
    :raises RuntimeError: 数据含图但模型不支持视觉
    """
    has_media = any(bool(getattr(sample, "media", None)) for sample in samples)
    if has_media and not model_supports_vision:
        raise RuntimeError(
            f"training data contains media but model {model_name!r} does not "
            "support vision. Either switch to a multimodal model or filter out "
            "media samples."
        )


def run_multimodal_preflight(
    runtime: Any,
    samples: list[Any],
    *,
    data_dir: str,
    image_token_id: int | None,
    model_name: str,
    pp_size: int = 1,
) -> None:
    """设施层预检：encode → attach → resolve → fake forward 梯度检查。

    :param runtime: GraspoFlowRuntime（已 setup，adapter 可用）
    :param samples: 训练样本列表
    :param data_dir: 训练数据目录（解析相对媒体路径）
    :param image_token_id: 图像占位 token id（None = 模型不支持视觉）
    :param model_name: 模型名（错误消息用）
    :param pp_size: 流水线并行度。多模态 + PP>1 的 rollout 生成已实现（R1），
        不再拦截；保留参数仅为调用方签名兼容，不做守卫。
    :raises RuntimeError: 任一环节不满足防线要求

    本函数需要 GPU/分布式上下文，只能真机执行；逻辑校验部分
    （assert_data_vision_compatible）可独立单测。
    """
    has_media = any(bool(getattr(sample, "media", None)) for sample in samples)
    if not has_media or image_token_id is None:
        # 数据纯文本 → 无需预检；模型纯文本 → 无需预检。
        # 两者都不触 adapter/GPU（纯文本训练零开销）。
        return
    adapter = runtime._require_adapter()  # noqa: SLF001 同包编排
    model = getattr(adapter, "model", None)
    has_vision = model is not None and bool(getattr(model, "visual", None))
    if model is None:
        raise RuntimeError("preflight requires a loaded model")
    assert_data_vision_compatible(samples, model_supports_vision=has_vision, model_name=model_name)
    if not has_vision:
        return  # 模型无视觉塔（与数据不匹配已在 assert 中拦截）

    # 取第一个含图样本
    media_sample = next((s for s in samples if bool(getattr(s, "media", None))), None)
    if media_sample is None:
        return

    row = multimodal_row_from_sample(media_sample, data_dir=data_dir)
    metadata = attach_rows({}, [row])

    # 2) 契约链路：attach 后 resolve 必须非 None
    resolved = adapter._multimodal_inputs_from_metadata(  # noqa: SLF001
        metadata, batch_size=1
    )
    if resolved is None:
        raise RuntimeError(
            "multimodal preflight failed: encode → attach → resolve chain is broken. "
            f"metadata={MULTIMODAL_ROWS_KEY!r} present but resolve returned None. "
            "Refusing to start training — multimodal inputs would be silently dropped."
        )

    # 3) visual LoRA 参数必须已注册且 requires_grad
    visual_params = [
        (name, param)
        for name, param in model.named_parameters()
        if "visual" in name and param.requires_grad
    ]
    if not visual_params:
        raise RuntimeError(
            "multimodal preflight failed: no trainable visual parameters found. "
            "vision_common LoRA target may not be registered. Refusing to start training."
        )

    # 4) fake 1-step 前向：visual LoRA 必须产生非零梯度。
    #    复用训练的同一路径：encode（取 input_ids/attention_mask）+ train
    #    模式 sequence_log_probs（带 multimodal_inputs）+ backward。
    _assert_visual_gradients_nonzero(adapter, model, row, visual_params)
    _log.info(
        "multimodal preflight passed: chain OK, %d trainable visual params, "
        "visual LoRA gradients non-zero",
        len(visual_params),
    )


def _assert_visual_gradients_nonzero(
    adapter: Any,
    model: Any,
    row: dict[str, Any],
    visual_params: list[tuple[str, Any]],
) -> None:
    """fake 1-step 前向并断言 visual LoRA 梯度非零。

    与训练 forward 使用同一入口（``_encode_multimodal_rows`` +
    ``model.sequence_log_probs``），确保预检验证的路径就是训练实际
    走的路径。梯度检查后 zero_grad 并恢复 eval 模式。
    """
    import torch  # noqa: PLC0415 设施函数内导入

    encoded = adapter._encode_multimodal_rows(  # noqa: SLF001
        [row],
        add_generation_prompt=False,
        chat_template_kwargs=adapter.config.model.chat_template_kwargs,
    )
    input_ids = encoded["input_ids"].to(adapter.device)
    attention_mask = encoded["attention_mask"].to(adapter.device).bool()
    multimodal_inputs = adapter._multimodal_inputs_to_device(encoded)  # noqa: SLF001

    was_training = model.training
    model.train()
    try:
        log_probs = model.sequence_log_probs(
            input_ids,
            attention_mask,
            multimodal_inputs=multimodal_inputs or None,
        )
        loss = log_probs.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(
                "multimodal preflight failed: fake forward loss is not finite. "
                "Refusing to start training."
            )
        loss.backward()
        # 注意：必须在 zero_grad 之前检查梯度——zero_grad(set_to_none=True)
        # 会把 grad 置为 None，提前清理会导致下面的检查永远全零。
        # TP 分片下 lora_a（输入投影，SUM 同步）在本 rank 梯度可能为零，
        # 属正常现象；判定标准是"至少一个视觉 LoRA 参数收到非零梯度"。
        nonzero = [
            name
            for name, param in visual_params
            if param.grad is not None and bool(param.grad.abs().sum() > 0)
        ]
        if not nonzero:
            raise RuntimeError(
                "multimodal preflight failed: visual LoRA gradients are all zero "
                "in fake 1-step forward. Vision tower is not receiving gradients — "
                "the exact failure mode of a known visual gradient bug. Refusing to start training."
            )
        _log.debug(
            "preflight fake-forward visual gradient signal: %d/%d params non-zero "
            "(TP 下 lora_a 分片梯度为零属正常)",
            len(nonzero),
            len(visual_params),
        )
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
