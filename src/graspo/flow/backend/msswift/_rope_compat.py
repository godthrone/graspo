"""RoPE 参数在 transformers 5.x 下的键名适配（计算层，纯函数 + 一个作用域补丁）。

**职责边界**

- **本模块**：把 ms-swift 写进模型 config 的**旧格式** ``rope_scaling``
  （``{'type': 'yarn', 'factor': ...}``）适配成 transformers 5.x 读取的**新格式**
  ``rope_parameters``（``{'rope_type': 'yarn', ...}``），并在 ms-swift 的模型加载
  边界上**作用域内**应用这个适配。
- **不负责**：改 ms-swift 源码、决定用户该用哪种 rope（那是配置层）、
  长文显存/上限（那是实测层）。

**为什么必须在 graspo 侧做这次适配**（E2b 实测，2026-09-16）

ms-swift 4.5.3 的 ``swift/model/register.py:232`` 用
``HfConfigFactory.set_config_attr(config, 'rope_scaling', self.rope_scaling)`` 把参数
写进已构造好的 config 对象。transformers 5.12 的 ``PretrainedConfig`` 里
``rope_scaling`` 是 ``rope_parameters`` 的**属性别名**（``configuration_utils.py:489-494``），
所以这行实际写的是 ``config.rope_parameters = {'type': 'yarn', ...}``——
**旧键** ``type``。

而 transformers 5.x 的模型代码读的是新键：

.. code-block:: python

    # transformers/models/qwen3/modeling_qwen3.py:96
    self.rope_type = self.config.rope_parameters["rope_type"]

把 ``type`` 归一化成 ``rope_type`` 的那步是
``PretrainedConfig.standardize_rope_params()``（``modeling_rope_utils.py:747``），
它只在两条路径上被调用：① config 构造时（``convert_rope_params_to_dict``）；②
``rope_init_fn`` 内部。**ms-swift 的 setattr 两条都不走**——config 早就构造完了，
而 ``Qwen3RotaryEmbedding.__init__`` 在调 ``rope_init_fn`` **之前**就读了
``rope_parameters["rope_type"]``。于是必然 ``KeyError: 'rope_type'``。

**与 ms-swift 自身代码的关系**：ms-swift 是**知道**新键的——同文件的
``_postprocess_config`` 会为了 transformers 5 从 ``rope_parameters`` 里补
``rope_theta`` / ``partial_rotary_factor``，``infer_engine/vllm_engine.py:280`` 也显式
做了 ``type → rope_type`` 的搬运。只有"给 HF 模型加载用的那份 config"漏了这一步，
属 ms-swift 4.5.3 与 transformers 5.12 之间的版本缝隙。按宪法 §1.2（不改上游、
只在扩展点适配）由 graspo 在**边界上**补齐，不 fork ms-swift。

**适配的两种实现**（先官方后兜底，都在同一个函数里，顺序显式）：

1. ``config.standardize_rope_params()`` —— transformers 自己的归一化函数，
   内部就是 ``rope_parameters.setdefault('rope_type', rope_parameters.get('type', 'default'))``。
2. 兜底：若第 1 步没生效（例如未来版本改了方法名或语义），显式写入
   ``rope_type``。两条路都走完仍缺 ``rope_type`` 才算失败——**不静默通过**（§13.1）。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

#: 旧键 → 新键（transformers 5.x 的 rope 参数命名）。
_ROPE_KEY_ALIASES: tuple[tuple[str, str], ...] = (("type", "rope_type"),)

#: ``rope_parameters`` 可能挂在这些嵌套子 config 上（多模态：LLM 部分在 ``llm_config`` 等）。
_ROPE_CONFIG_PATHS: tuple[str, ...] = (
    "",
    "language_config",
    "llm_config",
    "text_config",
    "thinker_config",
    "vision_config",
)


def suggested_rope_parameters(value: Any) -> dict[str, Any] | None:
    """把用户写的 ``msswift.rope_scaling`` 归一成 transformers 5.x 的 ``rope_parameters``。

    Args:
        value: 配置里的 ``rope_scaling``（``str`` 如 ``"yarn"``，或 ``dict``）。

    Returns:
        ``{"rope_type": "yarn", ...}``；``None`` 表示"未提供"（不透传，§2.2 None 语义）。

    Note:
        只做**键名**搬运，不发明取值：``yarn`` / ``dynamic`` / ``linear`` / ``default``
        都是 transformers ``ROPE_INIT_FUNCTIONS`` 认识的类型。ms-swift 的
        ``_init_rope_scaling`` 会在此基础上补 ``factor`` /
        ``original_max_position_embeddings``（它读的是新键 ``rope_type``，与本次适配同向）。
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # ms-swift 也接受 JSON 字符串形态（`--rope_scaling '{"type": "yarn"}'`）。
        if text.startswith("{"):
            import json

            try:
                parsed = json.loads(text)
            except ValueError:
                return {"rope_type": text}
            return _rename(parsed) if isinstance(parsed, dict) else {"rope_type": text}
        return {"rope_type": text}
    if isinstance(value, dict):
        return _rename(value)
    return {"rope_type": str(value)}


def _rename(mapping: dict[str, Any]) -> dict[str, Any]:
    """旧键 → 新键（不改传入的 dict，避免污染配置层对象）。"""
    result = dict(mapping)
    for old, new in _ROPE_KEY_ALIASES:
        if old in result and new not in result:
            result[new] = result.pop(old)
    result.setdefault("rope_type", "default")
    return result


#: 进程内缓存：模型目录 → 出厂 ``rope_theta``（同一进程内每个模型只读一次盘）。
_ROPE_THETA_CACHE: dict[str, float | None] = {}


def _resolve_rope_theta(config: Any) -> float:
    """找回被 ms-swift 覆写 dict 时丢掉的 ``rope_theta``。

    优先级（**显式**，不静默降级到错值）：

    1. ``config.rope_theta`` 属性 —— 部分 transformers 版本把出厂值留在属性上。
    2. 模型目录的 ``config.json`` —— 权威来源（transformers 5.12 的 Qwen3 config
       把 rope 参数收进了 ``rope_parameters``，删掉了顶层 ``rope_theta`` 属性，
       实测 ``getattr(cfg, 'rope_theta')`` 不存在）。
    3. ``config.default_theta`` —— transformers 的基础默认值（Qwen3 上是 10000.0，
       **不是**该模型的 1000000），只有前两条都不可用时才用，并记 WARNING
       （§3.2 透明退路：结果可能不同，必须让用户知道）。

    Note:
        第 2 条读的是**只读**的模型目录；无盘可读时退回第 3 条并留痕，不抛异常——
        这里的目标是"让 rope 能工作"，不是"没有盘就停摆"。
    """
    attribute = getattr(config, "rope_theta", None)
    if isinstance(attribute, (int, float)) and attribute > 0:
        return float(attribute)

    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path:
        cached = _ROPE_THETA_CACHE.get(str(name_or_path), "MISS")
        if cached != "MISS":
            return cached if cached is not None else _default_theta(config)
        value: float | None = None
        try:
            import json
            import os

            path = os.path.join(str(name_or_path), "config.json")
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            candidate = raw.get("rope_theta")
            if candidate is None and isinstance(raw.get("rope_scaling"), dict):
                candidate = raw["rope_scaling"].get("rope_theta")
            if isinstance(candidate, (int, float)) and candidate > 0:
                value = float(candidate)
        except (OSError, ValueError):
            value = None
        _ROPE_THETA_CACHE[str(name_or_path)] = value
        if value is not None:
            return value

    return _default_theta(config)


def _default_theta(config: Any) -> float:
    """第 3 条兜底：transformers 的基础默认值（带 WARNING，不静默）。"""
    fallback = getattr(config, "default_theta", None)
    fallback = 10000.0 if not isinstance(fallback, (int, float)) else float(fallback)
    logger.warning(
        "graspo rope compatibility: could not recover the model's own rope_theta (no "
        "rope_theta attribute and no readable config.json); falling back to the transformers "
        "default %s. If the model uses a custom RoPE base, pass it explicitly or the long-context "
        "rope scaling will be computed on the wrong base.",
        fallback,
    )
    return fallback


def normalize_rope_parameters(config: Any) -> dict[str, Any] | None:
    """就地把 config（含嵌套子 config）的 rope 参数归一为 transformers 5.x 形态。

    Args:
        config: HuggingFace ``PretrainedConfig``（或等价对象）。

    Returns:
        归一化后的 ``rope_parameters``；``None`` 表示该 config 上本来就没有 rope 参数
        （此时不改动，也不报错——不是所有模型都有 rope）。

    Raises:
        ValueError: 明明有 rope 参数、但归一化后仍缺 ``rope_type``
            （拒绝静默通过：那种状态下模型加载必然失败，只是报错点会很远）。
    """
    touched: dict[str, Any] | None = None
    for path in _ROPE_CONFIG_PATHS:
        target = config if not path else getattr(config, path, None)
        if target is None:
            continue
        parameters = getattr(target, "rope_parameters", None)
        if parameters is None:
            continue
        # ① ``rope_theta`` 不能在 ms-swift 覆写 dict 时丢失。
        #
        # ms-swift 用一整个新 dict 覆盖 ``rope_parameters``，把出厂 config 里的
        # ``rope_theta`` 洗掉了；而 transformers 的 ``_compute_yarn_parameters`` 需要
        # ``rope_parameters["rope_theta"]`` 当底（实测：不做这一步会得到
        # ``TypeError: unsupported operand type(s) for ** or pow(): 'NoneType' and 'Tensor'``，
        # 这是 rope 适配修好 KeyError 之后的**下一个**坑，E2b 的报错止于第一个坑所以没暴露）。
        if isinstance(parameters, dict) and not parameters.get("rope_theta"):
            parameters["rope_theta"] = _resolve_rope_theta(target)
        # ② 官方归一化（transformers 5.x 自己的函数，不复制其语义）
        standardize = getattr(target, "standardize_rope_params", None)
        if callable(standardize):
            standardize()
        # ③ 兜底：显式写 rope_type（版本漂移时仍能工作）
        parameters = getattr(target, "rope_parameters", None)
        if isinstance(parameters, dict):
            for old, new in _ROPE_KEY_ALIASES:
                if old in parameters and new not in parameters:
                    parameters[new] = parameters.pop(old)
            if "rope_type" not in parameters:
                raise ValueError(
                    "graspo rope compatibility: rope_parameters is present but 'rope_type' is "
                    f"still missing after normalization (keys={sorted(parameters)}); refusing to "
                    "continue because transformers>=5 reads rope_parameters['rope_type'] and "
                    "would fail later with an unrelated KeyError."
                )
            touched = parameters
    return touched


def _classmethod_function(descriptor: Any, accessed: Any = None) -> Any:
    """从"访问后"的 ``classmethod`` 取出**未绑定**的底层函数。

    `PreTrainedModel.from_pretrained` 是 ``@classmethod``；通过类访问会得到 bound
    method（`__self__` 是类本身），它**没有**可读的 `__func__`。ms-swift 自己的
    `patch_automodel`（``swift/model/patcher.py:380``）在**装饰器内**做
    `PreTrainedModel.from_pretrained.__func__`，那时类属性还是 descriptor，所以能拿到。

    本函数用 ``inspect.getattr_static`` 绕过描述符协议拿 descriptor，兼容三种形态：

    1. ``classmethod`` descriptor → ``.__func__``
    2. bound method（``__self__`` 是类）→ ``.__func__``
    3. 普通函数 → 原样返回

    **为什么必须走这一步**：第一版包装器写了
    ``original.__func__ if isinstance(original, classmethod) else original``，
    运行时 ``isinstance`` 恒为 False，于是绑定后的函数被当成未绑定函数调用，
    `cls` 被塞进了 `pretrained_model_name_or_path` 位置 —— 真实报错是
    ``HFValidationError: Repo id ... '<class ...Qwen3ForCausalLM>'``（E3 实测）。
    """
    if isinstance(descriptor, classmethod):
        return descriptor.__func__
    if isinstance(descriptor, staticmethod):
        return descriptor.__func__
    if isinstance(accessed, classmethod):
        return accessed.__func__
    return getattr(accessed, "__func__", accessed)


@contextlib.contextmanager
def rope_parameters_compatible(rope_scaling: Any) -> Iterator[None]:
    """在 ms-swift 的模型加载边界上**作用域内**补 transformers 5.x 的 rope 键名。

    Args:
        rope_scaling: graspo 配置里的 ``msswift.rope_scaling``。为 ``None`` 时本上下文
            **完全不装补丁**（不改变未使用该参数时的行为）。

    Why patch ``PreTrainedModel.from_pretrained``：

        ms-swift 在 ``ModelLoader._postprocess_config``（``swift/model/register.py:222``）
        里把 rope 参数写进 config，那一步发生在 ``from_pretrained`` **内部**。要保证
        "写完之后、模型 ``__init__`` 之前"这个时机，唯一不依赖 ms-swift 内部函数名的
        钩子就是 ``from_pretrained`` 自身：config 在进入本上下文时已被 ms-swift 写好
        （实测：本包装器里读到的 ``rope_parameters`` 已含 ``factor`` /
        ``original_max_position_embeddings``，正是 ms-swift 的产物）。

        补丁是**包装**（调用原函数）而不是替换，且退出即还原（§2.2 显式即防呆）；
        不改 ms-swift 源码（§1.2）。

        **必须保持 `classmethod` 语义**：`PreTrainedModel.from_pretrained` 是
        `@classmethod`，`cls` 由描述符协议注入。第一版包装器漏了 `@classmethod`，
        于是被调用时 `cls` 绑定成了第一个位置参数（模型类本身），
        实际报错是 `HFValidationError: Repo id ... '<class transformers...Qwen3ForCausalLM>'`
        ——**错误信息与真实原因相隔极远**（E3 实测踩过）。这里的写法与 ms-swift
        自己的 `patch_automodel`（`swift/model/patcher.py:380-401`）逐字同构。
    """
    if rope_scaling is None:
        yield
        return

    from transformers import PreTrainedModel

    # 用 `inspect.getattr_static` 拿**类上的 descriptor**（绕过描述符协议）：
    #   - 还原时用 `descriptor`（保持 classmethod 语义，getattr_static 能逐字取回）；
    #   - 调用时用 `descriptor.__func__`（未绑定函数，`cls` 才由我们显式传入）。
    # 直接读 `PreTrainedModel.from_pretrained` 得到的是 bound method
    # （`__self__` = PreTrainedModel），它**没有**可读的 `__func__`——第一版据此写的
    # `isinstance(original, classmethod)` 恒为 False（E3 实测踩过的坑）。
    original = inspect.getattr_static(PreTrainedModel, "from_pretrained")
    unbound = _classmethod_function(original, PreTrainedModel.from_pretrained)
    applied: list[dict[str, Any]] = []

    @classmethod
    def _graspo_from_pretrained(cls: Any, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.get("config")
        if config is not None:
            normalized = normalize_rope_parameters(config)
            if normalized is not None:
                applied.append(normalized)
        return unbound(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = _graspo_from_pretrained
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = original

    if applied:
        logger.warning(
            "graspo rope compatibility: normalized rope_parameters for transformers>=5 "
            "(%d config(s)): %s",
            len(applied),
            applied[0],
        )
