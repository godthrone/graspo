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

from graspo.core.lora import LORA_TARGET_PRESETS
from graspo.ripple.multimodal.rows import (
    MULTIMODAL_ROWS_KEY,
    attach_rows,
    multimodal_row_from_sample,
)

_log = logging.getLogger("graspo.preflight")


def assert_lora_vision_targets_trainable(
    *,
    lora_target_modules: list[str] | None,
    lora_target_preset: str | None,
    image_token_id: int | None,
    model_name: str,
) -> None:
    """纯逻辑校验：多模态训练时 LoRA 目标必须真的包含视觉塔（可单测，不触 GPU）。

    **为什么需要它（实测 T028，2026-09-19）**：``lora.target_preset`` 默认
    ``language_safe``（``core/lora.py`` 只含 ``language.*`` 模式），于是
    ``build_qwen35_visual_tower`` 先把视觉塔全部参数 ``requires_grad=False``
    再做 LoRA 替换时**没有任何 visual target 命中** ⇒ 视觉塔零可训参数。
    原防线（``run_multimodal_preflight`` 第 3 步）能拦，但它要等模型加载完、
    起好分布式之后才执行（T028 实测在加载后才炸）；本函数把同一判据提前到
    **配置期**判定，不需要 GPU、不需要权重，报错直接给出该改哪个键。

    判据与运行期防线**同源**：预设/精确名是否解析出视觉目标，用的就是
    ``core/lora.py::LORA_TARGET_PRESETS`` 与 ``_match_lora_pattern``——不另立
    一套"这里认得、那边不认得"的规则（宪法 §1.4）。

    :param lora_target_modules: 显式目标名列表（``lora.target_modules``），
        非 None 时优先于预设（与 ``resolve_lora_target_modules`` 同语义）
    :param lora_target_preset: ``lora.target_preset``（None ⇒ 走默认预设）
    :param image_token_id: 模型视觉占位 token id；None = 模型无视觉塔（不判）
    :param model_name: 模型名（错误消息用）
    :raises ValueError: 模型有视觉塔、且 LoRA 目标里没有一个视觉模块
    """
    if image_token_id is None:
        return  # 模型没有视觉塔：是否可训视觉不适用
    requested = tuple(lora_target_modules) if lora_target_modules else (lora_target_preset,)
    patterns: list[str] = []
    for item in requested:
        preset = LORA_TARGET_PRESETS.get(str(item)) if item is not None else None
        if preset is not None:
            patterns.extend(preset)
        elif item is not None:
            patterns.append(str(item))
    if not patterns:
        patterns.extend(LORA_TARGET_PRESETS["language_safe"])  # 与 core/lora.py 默认一致
    visual_patterns = [pattern for pattern in patterns if pattern.startswith("visual.")]
    if visual_patterns:
        return
    suggestions = sorted(name for name in LORA_TARGET_PRESETS if name.startswith("vision"))
    raise ValueError(
        f"multimodal training with model {model_name!r} but the LoRA targets select no "
        f"visual module: lora.target_modules={lora_target_modules!r}, "
        f"lora.target_preset={lora_target_preset!r}, resolved patterns={patterns!r}. "
        "The visual tower would be frozen (all its params are set requires_grad=False "
        "before LoRA replacement), then the run dies at multimodal preflight with "
        "`RuntimeError: multimodal preflight failed: no trainable visual parameters found`. "
        "Refusing to start training. Fix: set lora.target_preset to one of "
        f"{suggestions}, or set lora.target_modules to explicit visual.* module names. "
        "A preset that only names language modules (e.g. 'language_safe' / "
        "'language_all_linear', the default) can never make the visual tower trainable."
    )


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
    if model is None:
        raise RuntimeError("preflight requires a loaded model")
    # 视觉支持是模型级属性：以 config 的 image_token_id 判定（PP 下 visual 塔仅存在于
    # stage 0，不能用 model.visual 逐 rank 判定，否则 stage>0 会误判为无视觉）。
    has_vision = image_token_id is not None
    assert_data_vision_compatible(samples, model_supports_vision=has_vision, model_name=model_name)
    if not has_vision:
        return  # 模型无视觉塔（与数据不匹配已在 assert 中拦截）
    if int(pp_size) > 1:
        # PP>1：resolve / visual-LoRA 梯度重检查需整条流水线参与，无法在此独立执行；
        # 与 SFT 路径一致（SFT 本就不跑此预检），跳过并记 WARNING（透明退路，不改变训练结果）。
        _log.warning(
            "multimodal visual-chain preflight (resolve + visual-LoRA gradient) skipped "
            "under pp_size=%d; the visual chain is validated by the training forward itself",
            int(pp_size),
        )
        return

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
