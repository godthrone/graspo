"""全参多模态训练：数据编码不得产出 inference tensor（作用域补丁，不改 ms-swift 源码）。

**职责边界**（宪法 §1.1）

- **本模块**：``tuner_type: full`` **且**模型带视觉塔时，把 ms-swift ``Template.encode``
  的**装饰器语义**从 ``torch.inference_mode()`` 换成 ``torch.no_grad()``——只换上下文，
  不动编码逻辑；退出作用域即还原。
- **不负责**：编码内容（tokenize / processor）、教师侧任何东西、LoRA 路径
  （LoRA 档**不装这个包装**，见下）、参数映射、数据集准备。

**为什么需要它**（M039/M040 真机实证，2026-10-28）

``swift/template/base.py:652`` 用 ``@torch.inference_mode()`` 装饰 ``Template.encode``，
数据集预处理（``swift/dataset/utils.py:122``）就在这个上下文里跑，于是产出的
``pixel_values`` / ``image_grid_thw`` 是 **inference tensor**，并被带进训练 batch。
全参（``--freeze_vit false``，graspo 对 full 的显式语义，见 ``core/schema.py``）下
视觉塔 ``patch_embed`` 的 conv3d 需要 grad ⇒ 它要把这个输入**存下来供 backward 用**：

.. code-block:: text

    swift/rlhf_trainers/gkd_trainer.py:210  compute_loss → model(**forward_inputs)
    swift/template/base.py:1679            pre_forward_hook → _post_encode
    swift/template/base.py:2345            mixed_embeds = visual(pixel_values_mixed, ...)
    torch/nn/modules/conv.py:718           F.conv3d(...)
    RuntimeError: Inference tensors cannot be saved for backward.

LoRA 档不崩的原因**不是**走了别的代码路径（同一条），而是视觉塔参数
``requires_grad=False`` ⇒ autograd 不记账 ⇒ 不需要保存输入。所以本模块只在
``tuner_type == "full"`` 时安装，LoRA 档连包装都不挂（子代理工作包要求：不让诊断路径
污染 54 档的公共路径）。

**为什么改上下文而不是 detach**（实测，别再踩）

``detach()`` **不解除** inference 属性：``torch.randn(...)`` 在 ``inference_mode`` 下产生后，
``t.detach().is_inference()`` 仍是 ``True``，送进可训练 conv3d 依旧抛同一个
``RuntimeError``（``torch.no_grad() + detach()`` 同样无效）。inference 属性只有两条
出路：①回到**产生它的上下文**（本模块做法，数值逐位不变）；②``clone()`` 换一块普通
内存（PyTorch 错误信息里推荐的做法，但白拷一份像素、且是"看起来修了"的临时手段）。

**上游修好后怎么办**（升级韧性，不静默）

- ``Template.encode`` 若已**没有装饰器**（``__wrapped__`` 缺失）⇒ 说明这段 inference 上下文
  没了（上游已修或换了实现），本模块**直通**并在日志里记一句，不报错。
- 装饰器链若**换了形态**（剥一层后仍产出 inference tensor）⇒ 抛 ``RuntimeError`` 并指明
  哪几个键仍是 inference tensor。**不允许**静默通过——"看起来修了其实没修"比崩更贵。
- 上游符号缺失（``swift`` 未安装 / ``Template.encode`` 改名）⇒ 抛 ``RuntimeError`` 说清缺什么。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

#: 上游符号缺失时的可操作提示（单一真相源）。
MS_SWIFT_TEMPLATE_MISSING = (
    "graspo: cannot patch the ms-swift data-encoding context because "
    "`swift.template.base.Template.encode` is unavailable. "
    "This patch is required for `tuner_type: full` on multimodal models — without it the "
    "vision tower's patch_embed conv3d raises "
    "'RuntimeError: Inference tensors cannot be saved for backward' on the first step. "
    "Reinstalling/upgrading ms-swift may have renamed or removed the symbol; check "
    "swift/template/base.py for `def encode`."
)

#: 装饰器链变了（剥一层后仍是 inference tensor）时的提示。
ENCODE_STILL_INFERENCE = (
    "graspo: patched ms-swift `Template.encode` to run under torch.no_grad(), but the encoded "
    "batch still contains inference tensors ({keys}). The upstream inference_mode decorator is "
    "no longer the outermost one (swift/template/base.py:652), so this patch cannot reach it — "
    "full-parameter multimodal training would crash at the vision tower's patch_embed conv3d "
    "('Inference tensors cannot be saved for backward'). Refusing to continue silently."
)

#: 判别"是否带视觉塔"所需的上游信息缺失时的提示。
TEMPLATE_META_MISSING = (
    "graspo: cannot decide whether ms-swift `Template.encode` needs the graph-safe rewrite: the "
    "template instance has no `model_meta` (set by `Template.init_processor`, "
    "swift/template/base.py:239). Without it we cannot tell whether the model has a vision tower, "
    "so we fail closed instead of guessing."
)

#: 打补丁后的观测计数（测试与排障用；不进业务逻辑）。
_STATS: dict[str, Any] = {
    "installed": False,
    "rewritten_calls": 0,
    "passthrough_calls": 0,
    "degraded_reason": None,
}


def encode_patch_stats() -> dict[str, Any]:
    """返回当前作用域内的补丁观测值（拷贝，调用方可随意持有）。

    ``installed`` = 是否真的挂上了包装（LoRA 档 / 上游已修 ⇒ ``False``）；
    ``rewritten_calls`` = 真的被改写成 ``no_grad`` 的 encode 次数；
    ``passthrough_calls`` = 进了包装但按原样走上游的次数（例如文本塔模板）；
    ``degraded_reason`` = 未安装/退化的原因（``None`` 表示按预期安装）。
    """
    return dict(_STATS)


def reset_encode_patch_stats() -> None:
    """把观测计数清回初始值（测试用）。"""
    _STATS.update(installed=False, rewritten_calls=0, passthrough_calls=0, degraded_reason=None)


def _require_swift_template() -> Any:
    """解析 ``swift.template.base.Template``。

    Raises:
        RuntimeError: ms-swift 未安装或符号缺失 —— 附可操作指引。
    """
    try:
        from swift.template.base import Template
    except ImportError as exc:
        raise RuntimeError(MS_SWIFT_TEMPLATE_MISSING) from exc
    if getattr(Template, "encode", None) is None:
        raise RuntimeError(MS_SWIFT_TEMPLATE_MISSING)
    return Template


def needs_graph_safe_encode(template: Any) -> bool:
    """这个模板实例的编码产物是否会喂给**可训练**的视觉塔。

    判据 = ``model_meta.is_multimodal`` 且 ``model_meta.model_arch.vision_tower`` 非空。
    两个属性都来自 ``processor.model_meta``（``Template.init_processor`` 落盘），
    所以只在 ``encode`` 已被调用过的实例上存在。

    Raises:
        RuntimeError: ``model_meta`` 缺失 —— 无法判定，fail-closed。
    """
    model_meta = getattr(template, "model_meta", None)
    if model_meta is None:
        raise RuntimeError(TEMPLATE_META_MISSING)
    if not getattr(model_meta, "is_multimodal", False):
        return False
    model_arch = getattr(model_meta, "model_arch", None)
    if model_arch is None:
        raise RuntimeError(TEMPLATE_META_MISSING)
    return bool(getattr(model_arch, "vision_tower", None))


def _assert_no_inference_tensors(encoded: Any) -> None:
    """编码产物里不允许残留 inference tensor（否则补丁没真正生效）。"""
    if not isinstance(encoded, dict):
        return
    offending = sorted(
        key
        for key, value in encoded.items()
        if hasattr(value, "is_inference") and bool(value.is_inference())
    )
    if offending:
        raise RuntimeError(ENCODE_STILL_INFERENCE.format(keys=", ".join(offending)))


def _install(template_cls: Any) -> contextlib.AbstractContextManager[None]:
    """把 ``template_cls.encode`` 换成 ``no_grad`` 版本；返回还原用上下文。"""
    import torch

    original = template_cls.encode
    inner = getattr(original, "__wrapped__", None)
    if inner is None:
        # 上游已去掉装饰器 ⇒ 编码本来就在普通上下文里跑，直通即可（升级韧性）。
        _STATS["installed"] = False
        _STATS["degraded_reason"] = "upstream encode carries no decorator (already fixed)"
        logger.info(
            "graspo: ms-swift `Template.encode` is not decorator-wrapped — "
            "the inference_mode data-encoding patch is unnecessary, skipping it."
        )
        return contextlib.nullcontext()

    def encode_without_inference_mode(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not needs_graph_safe_encode(self):
            # 模型没有视觉塔（或纯文本模板）：没有消费者需要保存这个输入，保持上游行为。
            _STATS["passthrough_calls"] += 1
            return original(self, *args, **kwargs)
        with torch.no_grad():
            encoded = inner(self, *args, **kwargs)
        _assert_no_inference_tensors(encoded)
        _STATS["rewritten_calls"] += 1
        return encoded

    for attr in ("__module__", "__name__", "__qualname__", "__doc__"):
        with contextlib.suppress(AttributeError, TypeError):
            setattr(encode_without_inference_mode, attr, getattr(inner, attr))
    encode_without_inference_mode.__wrapped__ = inner  # 保留上游链，便于其它补丁/排障

    template_cls.encode = encode_without_inference_mode
    _STATS["installed"] = True
    _STATS["degraded_reason"] = None
    logger.info(
        "graspo: ms-swift data encoding switched from torch.inference_mode to torch.no_grad "
        "(tuner_type=full + vision tower): inference tensors may not be saved for backward."
    )

    @contextlib.contextmanager
    def _restore() -> Iterator[None]:
        try:
            yield
        finally:
            template_cls.encode = original
            _STATS["installed"] = False

    return _restore()


@contextlib.contextmanager
def graph_safe_encode(*, tuner_type: Any, template_cls: Any = None) -> Iterator[None]:
    """作用域内让 ms-swift 的数据编码不产出 inference tensor。

    Args:
        tuner_type: ``graspo`` 的有效参数化模式（``config.effective_tuner_type``）。
            只有 ``"full"`` 会安装补丁；其余（LoRA）**原样直通**——LoRA 档视觉塔
            参数被冻结，autograd 不记账，本来就不会触发该缺陷。
        template_cls: 被测/被补的模板类，默认惰性解析 ``swift.template.base.Template``。

    Yields:
        None。退出即还原（异常安全）。
    """
    if tuner_type != "full":
        _STATS["installed"] = False
        _STATS["degraded_reason"] = f"tuner_type={tuner_type!r} does not train the vision tower"
        yield
        return

    cls = _require_swift_template() if template_cls is None else template_cls
    if getattr(cls, "encode", None) is None:
        raise RuntimeError(MS_SWIFT_TEMPLATE_MISSING)
    with _install(cls):
        yield
