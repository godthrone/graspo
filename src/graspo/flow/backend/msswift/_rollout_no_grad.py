"""ms-swift on-policy rollout 的 ``inference_mode`` → ``no_grad`` 上下文修正（集成层补丁）。

**职责边界**（宪法 §1.1）

- **本模块**：把 ms-swift ``TransformersEngine._infer`` 上的 ``@torch.inference_mode()``
  在**作用域内**换成 ``torch.no_grad()``，使 rollout 期间被 FSDP2 物化出来的**参数**是
  普通张量；只换"在什么上下文里跑"，不碰张量、不碰 ``requires_grad``、不碰生成参数。
- **不负责**：教师侧任何东西、采样分布/温度/种子的取值（播种是 ``_rollout_seed.py``
  的职责）、参数映射、数据集准备。

**真因（D46 探针，2026-10-29，A800 真实 OPD 全链路 ``--smoke``，逐字读数）**

崩溃现场在视觉塔 ``patch_embed`` 的 ``torch/nn/modules/conv.py`` conv3d。探针在
``Conv3d._conv_forward`` 上打出三项，两次调用形成对照：

- ``#1`` = rollout（``is_inference_mode_enabled()=True``、``is_grad_enabled()=False``）：
  ``weight.is_inference=True``、``input.is_inference=True``；
- ``#2`` = 训练 forward（``is_inference_mode_enabled()=False``、``is_grad_enabled()=True``）：
  **``weight.is_inference=True``、``input.is_inference=False``** ⇒ 抛
  ``RuntimeError: Inference tensors cannot be saved for backward``。

⇒ 被污染的是**参数对象本身**（``weight.requires_grad=True``、``weight.is_leaf=True``），
不是输入。``input.is_inference=False`` 同时否掉了"数据集侧 ``Template.encode`` 泄漏"
那条假设：``swift/template/base.py:2105-2110 _data_collator_mm_data`` 的
``torch.concat`` 已把 encode 产出的 inference 张量归一化成普通张量。

**为什么"参数会变成 inference 张量"**：``_infer`` 的 ``inference_mode`` 罩住了整段
rollout，其中包含 FSDP2 的 all-gather 物化（``torch/.../_fsdp_collectives.py``
``foreach_all_gather_copy_out``）。在 ``inference_mode`` 下 ``torch.empty`` 出的
参数张量带 inference 标记，而 FSDP2 会**缓存**物化后的参数对象；rollout 结束后训练
forward 复用的正是这批缓存对象 ⇒ 反向要保存 weight 时发现它是 inference 张量。
``no_grad`` 只关 autograd 记录、**不产生** inference 标记，因此修法只需换上下文。

**作用面（最小充分）**：仅当 ``tuner_type == full`` ∧ 模型是多模态 ∧ rollout 走
``TransformersEngine``（``use_vllm=false``）时才替换并在执行期生效：

- ``tuner_type != full``（LoRA）⇒ **不安装**（视觉塔被冻结，weight 不进 backward，
  本缺陷不成立；不安装才能让 LoRA 档行为与修前逐字相同）；
- 文本模型 ⇒ 安装但不改写该次调用（``passthrough_calls``），保持上游 inference_mode
  的内存收益；
- 多模态判据取**上游自己的** ``self.model_meta.is_multimodal``（§1.4：不另造探测）。

**fail-closed**（宪法 §2.3）：上游符号缺失或已改成 ``no_grad`` ⇒ **不安装**并记
``degraded_reason``（无害直通）；一旦安装，每次改写前在 ``no_grad`` 里断言
``torch.is_inference_mode_enabled() is False``，否则抛 :data:`ROLLOUT_NO_GRAD_NOT_EFFECTIVE`
点名——⛔ 不允许"看起来修了"。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

#: ms-swift 的引擎模块/类/方法（上游 file:line 见模块 docstring）。
_ENGINE_MODULE = "swift.infer_engine.transformers_engine"
_ENGINE_CLASS = "TransformersEngine"
_ENGINE_METHOD = "_infer"

#: 安装了却没生效时的报错文案（fail-closed，必须点名到可操作）。
ROLLOUT_NO_GRAD_NOT_EFFECTIVE = (
    "graspo rollout no_grad patch is installed but had no effect: "
    "torch.is_inference_mode_enabled() is still True inside "
    "TransformersEngine._infer ⇒ FSDP2 仍会在 inference_mode 下把视觉塔参数物化成 "
    "inference 张量，反向保存 weight 时必炸（RuntimeError: Inference tensors cannot "
    "be saved for backward）。请检查是否有外层代码又套了 inference_mode。"
)


def _torch() -> Any:
    import torch  # 局部导入：本模块被 ChatOps/CLI 侧 import 时不强制拉起 torch

    return torch


def _decorator_context(func: Any) -> Any:
    """取装饰器包装函数持有的上下文管理器实例（用来识别它包的是哪一层）。

    ``@torch.inference_mode()`` / ``@torch.no_grad()`` 在函数上返回的是普通 function，
    但闭包里持有 ``torch.autograd.grad_mode`` 的上下文管理器实例——这是唯一可靠的
    判别依据（不带括号的类式用法 ``torch.inference_mode`` 直接就是该类实例）。

    Args:
        func: 待识别的类属性。

    Returns:
        上下文管理器实例；不是装饰器产物时返回 ``None``。
    """
    grad_mode = _torch().autograd.grad_mode
    if isinstance(func, grad_mode._DecoratorContextManager):
        return func  # 不带括号的类式用法
    for cell in getattr(func, "__closure__", None) or ():
        try:
            content = cell.cell_contents
        except ValueError:  # 空 cell
            continue
        owner = getattr(content, "__self__", None)
        if isinstance(owner, grad_mode._DecoratorContextManager):
            return owner
    return None


def _resolve_engine_class() -> Any:
    """解析 ms-swift 的 ``TransformersEngine``（未安装 ⇒ ``None``，不抛）。"""
    import importlib

    try:
        module = importlib.import_module(_ENGINE_MODULE)
    except ImportError:
        return None
    return getattr(module, _ENGINE_CLASS, None)


def _owner_class(cls: Any, name: str) -> Any:
    """找到 ``name`` 真正定义在 MRO 的哪一层（改属性必须改定义处）。"""
    for base in getattr(cls, "__mro__", ()):
        if name in base.__dict__:
            return base
    return cls


def rollout_no_grad_required(config: Any) -> tuple[bool, str | None]:
    """判断本条配置是否落在"全参 + 走 TransformersEngine rollout"的作用面内。

    多模态这一维**不在这里判**：它要等引擎实例的 ``model_meta`` 才拿得到，因此在
    :func:`rollout_no_grad` 的包装函数里按次判定（文本模型直通）。

    Args:
        config: ``GraspoConfig``。

    Returns:
        ``(是否需要, 不需要时的原因)``。原因为 ``None`` 表示需要安装。
    """
    tuner_type = getattr(config, "effective_tuner_type", None)
    if tuner_type != "full":
        return False, (
            f"tuner_type={tuner_type!r}: 只有全参训练会让视觉塔权重进 autograd 图"
            f"（LoRA 冻结视觉塔，本缺陷不成立）"
        )
    use_vllm = getattr(getattr(config, "msswift", None), "use_vllm", False)
    if use_vllm:
        return False, (
            "use_vllm=true: rollout 走 vLLM，TransformersEngine._infer 不参与"
            "（本补丁只覆盖 transformers 引擎路径）"
        )
    return True, None


@contextlib.contextmanager
def rollout_no_grad(config: Any) -> Iterator[dict]:
    """在作用域内把 ``TransformersEngine._infer`` 的 ``inference_mode`` 换成 ``no_grad``。

    Args:
        config: ``GraspoConfig``（作用面判据见 :func:`rollout_no_grad_required`）。

    Yields:
        可变账本：``{"installed": bool, "degraded_reason": str | None,
        "rewritten_calls": int, "passthrough_calls": int, "required": bool,
        "installed_at": str | None}``。调用方**必须**像 ``rollout_seeding`` 那样把账本
        带进日志：``installed=False`` 且 ``degraded_reason`` 非空时，本轮"已修"是假的。

    Note:
        - 幂等：已经装过就复用既有包装，不叠第二层。
        - 退出时**无条件**恢复原属性（异常路径也不留补丁）。
        - ``degraded_reason`` 只在"上游变了/装不上"时出现；安装成功但执行期发现
          ``inference_mode`` 仍在 ⇒ **抛错**，不降级。
    """
    ledger: dict[str, Any] = {
        "installed": False,
        "degraded_reason": None,
        "rewritten_calls": 0,
        "passthrough_calls": 0,
        "required": False,
        "installed_at": None,
    }
    required, reason = rollout_no_grad_required(config)
    ledger["required"] = required
    if not required:
        ledger["degraded_reason"] = reason
        logger.info("graspo rollout no_grad: 未安装（%s）", reason)
        yield ledger
        return

    engine_cls = _resolve_engine_class()
    if engine_cls is None:
        ledger["degraded_reason"] = f"ms-swift 不可用：找不到 {_ENGINE_MODULE}.{_ENGINE_CLASS}"
        logger.info("graspo rollout no_grad: 未安装（%s）", ledger["degraded_reason"])
        yield ledger
        return

    owner = _owner_class(engine_cls, _ENGINE_METHOD)
    if _ENGINE_METHOD not in owner.__dict__:
        ledger["degraded_reason"] = (
            f"上游漂移：{_ENGINE_CLASS} 上找不到 {_ENGINE_METHOD}（符号缺失，不能装）"
        )
        logger.info("graspo rollout no_grad: 未安装（%s）", ledger["degraded_reason"])
        yield ledger
        return

    original = owner.__dict__[_ENGINE_METHOD]
    if getattr(original, "_graspo_rollout_no_grad", False):
        ledger["installed"] = True
        ledger["installed_at"] = f"{owner.__module__}.{owner.__name__}.{_ENGINE_METHOD}"
        yield ledger
        return

    inner = getattr(original, "__wrapped__", None)
    context_manager = _decorator_context(original)
    grad_mode = _torch().autograd.grad_mode
    if inner is None or not isinstance(context_manager, grad_mode.inference_mode):
        ledger["degraded_reason"] = (
            f"上游已变或不可识别：{_ENGINE_METHOD} 当前的装饰器是 "
            f"{type(context_manager).__name__ if context_manager is not None else '无'}，"
            f"不再需要（或不再允许）本补丁"
        )
        logger.info("graspo rollout no_grad: 未安装（%s）", ledger["degraded_reason"])
        yield ledger
        return

    def _graspo_infer_no_grad(self: Any, *args: Any, **kwargs: Any) -> Any:
        """改写后的 ``_infer``：多模态在全参下改走 ``no_grad``，其余逐字直通。"""
        meta = getattr(self, "model_meta", None)
        if not getattr(meta, "is_multimodal", False):
            ledger["passthrough_calls"] += 1
            return original(self, *args, **kwargs)
        ledger["rewritten_calls"] += 1
        with _torch().no_grad():
            if _torch().is_inference_mode_enabled():
                raise RuntimeError(ROLLOUT_NO_GRAD_NOT_EFFECTIVE)
            return inner(self, *args, **kwargs)

    _graspo_infer_no_grad.__name__ = _ENGINE_METHOD
    _graspo_infer_no_grad.__doc__ = getattr(original, "__doc__", None)
    _graspo_infer_no_grad._graspo_rollout_no_grad = True  # type: ignore[attr-defined]
    _graspo_infer_no_grad._graspo_original = original  # type: ignore[attr-defined]

    owner._infer = _graspo_infer_no_grad  # type: ignore[attr-defined]
    ledger["installed"] = True
    ledger["installed_at"] = f"{owner.__module__}.{owner.__name__}.{_ENGINE_METHOD}"
    logger.info(
        "graspo rollout no_grad: 已安装于 %s（原装饰器 inference_mode → 执行期改用 no_grad；"
        "多模态判据取 model_meta.is_multimodal）",
        ledger["installed_at"],
    )
    try:
        yield ledger
    finally:
        owner._infer = original  # type: ignore[attr-defined]

    if ledger["rewritten_calls"] > 0:
        logger.info(
            "graspo rollout no_grad: 本次运行有 %d 次 rollout 在 no_grad 下执行"
            "（直通 %d 次：文本模型或非多模态批次）。",
            ledger["rewritten_calls"],
            ledger["passthrough_calls"],
        )
    elif ledger["passthrough_calls"] > 0:
        logger.info(
            "graspo rollout no_grad: 已安装但 %d 次 rollout 全部直通（非多模态）——"
            "本条通道无需改写。",
            ledger["passthrough_calls"],
        )
    else:
        logger.warning(
            "graspo rollout no_grad: 已安装但一次 rollout 都没发生（rewritten=0, "
            "passthrough=0）——本次运行的「已修」未被执行期证实，结论需另取证。"
        )


def reset_rollout_no_grad_ledger(ledger: dict) -> None:
    """把账本清零（供测试与多次 run 复用同一账本）。"""
    for key, value in (
        ("installed", False),
        ("degraded_reason", None),
        ("rewritten_calls", 0),
        ("passthrough_calls", 0),
        ("required", False),
        ("installed_at", None),
    ):
        ledger[key] = value
