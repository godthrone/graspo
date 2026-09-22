"""多模态训练启动预检（防线）。

在训练启动时（数据含图时）验证视觉链路完整性，把 v13 式的"跑 19.5 小时
才发现视觉 LoRA 从未训练"提前到启动 30 秒内拦截：

1. 数据含图但模型不支持视觉 → 报错（数据/模型不匹配）
2. encode → attach → resolve 契约链路必须完整（resolve 非 None）
3. 视觉塔必须有可训参数（载体按模式分派：LoRA ⇒ ``visual.*`` target 命中，
   全参 ⇒ 塔内至少一个 ``requires_grad=True`` 的参数）
4. fake 1-step 前向必须让视觉参数产生非零梯度

预检只验证**消费侧**链路与视觉可训练性；**生成侧**断链
（generation metadata 未 attach rows）由 ``ripple/multimodal/contract``
在训练 forward 时拦截（阶段 3 接线）。
"""

import logging
from typing import Any

from graspo.core.lora import LORA_TARGET_PRESETS
from graspo.core.schema import TunerType, resolve_tuner_type
from graspo.ripple.multimodal.rows import (
    MULTIMODAL_ROWS_KEY,
    attach_rows,
    multimodal_row_from_sample,
)

_log = logging.getLogger("graspo.preflight")


class _ModelConfigView:
    """模型配置的最小只读视图（鸭子类型），供纯逻辑预检跨后端复用。

    只暴露预检用到的字段，避免把具体后端配置类型耦合进纯逻辑层。
    """

    def __init__(self, config: Any) -> None:
        self.has_vision_config = bool(getattr(config, "has_vision_config", False))
        self.image_token_id = getattr(config, "image_token_id", None)


def assert_vision_tower_trainable(
    model: Any,
    config: Any,
    *,
    owns_embeddings: bool,
    model_name: str,
    tuner_type: TunerType | None = None,
) -> None:
    """纯逻辑防线：模型声明了视觉，本 rank 又持有 embedding 层，则视觉塔必须真的建出来**且可训**。

    这是缺陷②（``adapter.py`` 的 fail-open 豁免）的**唯一真相源判据**，
    同时被 native adapter 的加载期校验与 SFT 训练入口复用（宪法 §1.4）。

    **为什么需要它（真因）**：`resolve_lora_target_modules` 只在
    ``native_qwen_lora_available_targets`` 暴露的 target 里解析，而该函数
    **仅在 ``has_vision_config=True`` 时才暴露 ``visual.*``**（``lora_helpers.py:44``）
    ⇒ ``resolved`` 里出现 ``visual.*`` **蕴含"模型声明了视觉"**。

    而视觉塔属性（``model.py:96-111``）为 ``None`` 当且仅当
    ``not include_embeddings or not has_vision_config``，其中
    ``include_embeddings = (pp_rank == 0)``（``placement_plan.py:135``）。于是：

    - **模型无视觉**（``has_vision_config=False``）：``visual.*`` 不可能出现在
      ``resolved`` 里 ⇒ 本函数直接放行（**纯文本模型零开销、不误伤**）；
    - **模型有视觉 + 本 rank 不是 embedding rank**（PP>1）：该 rank 上视觉塔
      按设计不存在，工具层亦无法持有（不合法情形）⇒ 放行；
    - **模型有视觉 + 本 rank 持有 embedding 层**：视觉塔**必须在**。缺失即
      "视觉塔被静默跳过" ⇒ ``RuntimeError``，**fail-closed**。

    **"可训"判据按训练模式分派（宪法 §2.3，两种模式各自 fail-closed）**：
    视觉塔"存在"是模型结构事实，与模式无关，故上面的判据不分支；但"可训"
    的**载体**不同：

    - ``tuner_type="lora"``（含缺省 ``None`` ⇒ ``lora``，见
      :func:`graspo.core.schema.resolve_tuner_type`）：可训载体是 LoRA 矩阵 ⇒
      判据 = 注册了 ``visual.*`` 的 enabled LoRA target（**逐字保持原语义**）；
    - ``tuner_type="full"``：不构造任何 LoRA 矩阵（``model.py`` 传
      ``lora_r=0`` ⇒ ``LoRALinear.lora_enabled=False``，
      ``lora_linear.py:123``），``enabled_lora_target_names()`` **结构性恒空** ⇒
      沿用 LoRA 判据必然误报。全参的可训载体是**参数自身**（
      ``model_builders.py:173-177`` 在全参下统一解开视觉塔 ``requires_grad``）
      ⇒ 判据 = 视觉塔至少有**一个** ``requires_grad=True`` 的参数。
      这不是放松校验，而是把"视觉塔必须真的在训练"这一**同一风险点**换成与
      该模式等价的判据（穷举式：塔内零个可训参数 = 视觉塔被冻结，照旧 fail-closed）。

    :param model: 已构建的模型实例（只读 ``visual`` 属性、``visual.parameters()``
        与 ``enabled_lora_target_names()``）
    :param config: 模型配置（含 ``has_vision_config`` / ``image_token_id``）
    :param owns_embeddings: 本 rank 是否持有 embedding 层（PP 的 stage 0）
    :param model_name: 模型名（错误消息用）
    :param tuner_type: ``config.effective_tuner_type``。缺省 ``None`` ⇒ ``lora``
        （缺省调用行为逐字不变）。
    :raises RuntimeError: 声明了视觉、持有 embedding、却不存在视觉塔；或视觉塔
        在该模式下没有任何可训参数（LoRA 目标未覆盖视觉塔 / 全参下塔被冻结）
    """
    view = _ModelConfigView(config)
    if not view.has_vision_config:
        return  # 模型无视觉塔：视觉可训练性不适用（纯文本模型不得被误伤）
    if not owns_embeddings:
        return  # 视觉塔按 PP 设计只存在于 embedding stage；非该 stage 无塔属正常
    if getattr(model, "visual", None) is None:
        raise RuntimeError(
            f"model {model_name!r} declares vision (has_vision_config=True) and this rank "
            "owns the embedding stage, but the visual tower was not built. Refusing to "
            "start: vision would be silently skipped on this rank."
        )
    if resolve_tuner_type(tuner_type) != "lora":
        # 全参：可训载体是参数自身，不是 LoRA 矩阵（见 docstring 的"按模式分派"段）。
        # 视觉塔存在但零可训参数 = 该塔在本 run 里被冻结 ⇒ 与 LoRA 侧同一风险，
        # fail-closed 拦下（防呆强度不倒退）。
        tunable_visual_params = [
            param for param in model.visual.parameters() if bool(param.requires_grad)
        ]
        if not tunable_visual_params:
            raise RuntimeError(
                f"model {model_name!r} declares vision and owns the embedding stage, but the "
                "visual tower has no trainable parameter under tuner_type='full' (all of its "
                "params have requires_grad=False). Refusing to start: the visual tower would "
                "be frozen while the run trains. Fix: the full-parameter entry must unfreeze "
                "the visual tower (see build_qwen35_visual_tower) instead of relying on LoRA "
                "targets, which do not exist in full mode."
            )
        return
    enabled_target_names = getattr(model, "enabled_lora_target_names", None)
    if enabled_target_names is None:
        return  # 非原生 LoRA 模型（无该接口）：不适用
    visual_enabled = [name for name in enabled_target_names() if str(name).startswith("visual.")]
    if not visual_enabled:
        raise RuntimeError(
            f"model {model_name!r} declares vision and owns the embedding stage, but no "
            "trainable visual LoRA target is registered (enabled visual targets: []). "
            "Refusing to start: the visual tower would be frozen while the run trains. "
            "Fix: set lora.target_preset to a vision preset (e.g. 'vision_common') or set "
            "lora.target_modules to explicit visual.* module names."
        )


def assert_lora_vision_targets_trainable(
    *,
    lora_target_modules: list[str] | None,
    lora_target_preset: str | None,
    image_token_id: int | None,
    model_name: str,
    has_vision_config: bool | None = None,
    tuner_type: TunerType | None = None,
) -> None:
    """纯逻辑校验：**LoRA 模式下**多模态训练时 LoRA 目标必须真的包含视觉塔（可单测，不触 GPU）。

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

    **适用范围按训练模式收窄（宪法 §2.3，修 2026-09-22 的 T017/T018/T035/T036）**：
    本判据的前提是"视觉塔的可训载体 = LoRA 矩阵"。``tuner_type="full"`` 下
    ``build_native_qwen_model`` 传 ``lora_r=0`` ⇒ 不构造 LoRA 矩阵
    （``lora_linear.py:123`` ``lora_enabled = lora_enabled and r > 0``），视觉塔是
    由 ``model_builders.py:173-177`` **直接解开 ``requires_grad``** 的：
    "LoRA 目标是否覆盖视觉塔"在全参下**无对象可判**，照判必误报（全参档
    按生成器设计不写 ``lora`` 段，``target_preset`` 只是 schema 默认值
    ``core/schema.py:240``）。因此非 ``lora`` 模式在此直接返回——**这不是放松，
    是同一风险点换判据**：全参的等价判据在
    :func:`assert_vision_tower_trainable`（``tuner_type="full"`` 分支：
    塔内至少一个 ``requires_grad=True`` 的参数）；GRASPO 路径另有
    :func:`run_multimodal_preflight` 的"可训视觉参数存在 + fake 前向梯度非零"
    运行期防线兜底。**"声明了视觉塔却没有可用视觉占位 token"这条与模式无关的
    配置不一致校验保留在最前面**，不被模式分支跳过。

    :param lora_target_modules: 显式目标名列表（``lora.target_modules``），
        非 None 时优先于预设（与 ``resolve_lora_target_modules`` 同语义）
    :param lora_target_preset: ``lora.target_preset``（None ⇒ 走默认预设）
    :param image_token_id: 模型视觉占位 token id；None = 模型无视觉塔（不判）
    :param model_name: 模型名（错误消息用）
    :param has_vision_config: 模型是否**声明**视觉塔（``config.has_vision_config``）。
        默认 ``None`` = 未提供，此时保持既有语义（只按 ``image_token_id`` 判定，
        缺省调用行为逐字不变）。显式传入 ``True`` 而 ``image_token_id`` 为 None，
        是"模型有视觉塔但占位 token 缺失"的配置不一致 ⇒ **fail-closed 报错**
        （此前的 fail-open 会让视觉塔被静默跳过、无声退回语言-only）。
    :param tuner_type: ``config.effective_tuner_type``。缺省 ``None`` ⇒ ``lora``
        （``resolve_tuner_type`` 归一，缺省调用行为逐字不变）；非 ``lora`` ⇒
        本判据不适用，直接返回。
    :raises ValueError: 模型有视觉塔、且 **LoRA 模式**下 LoRA 目标里没有一个视觉
        模块；或声明了视觉塔却没有可用的视觉占位 token（配置不一致）
    """
    if has_vision_config is True and image_token_id is None:
        raise ValueError(
            f"model {model_name!r} declares a vision tower (has_vision_config=True) but has no "
            "usable image token id (config.image_token_id is missing/None). Vision inputs would "
            "be silently dropped and training would fall back to language-only. Refusing to "
            "start training."
        )
    if image_token_id is None:
        return  # 模型没有视觉塔：是否可训视觉不适用
    if resolve_tuner_type(tuner_type) != "lora":
        # 全参模式：不构造 LoRA 矩阵 ⇒ "LoRA 目标覆盖视觉塔"无对象可判。
        # 同一风险点的覆盖者见 docstring（assert_vision_tower_trainable 的 full 分支
        # + run_multimodal_preflight 的运行期梯度防线）。
        return
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
