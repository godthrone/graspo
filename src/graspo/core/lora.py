"""LoRA target module resolution —— 纯计算层，零设施依赖。

该模块可在 CPU 上独立测试，不依赖 GPU、分布式或任何模型实现。

**目标写法的唯一真相源（宪法 §1.4）**：``lora.target_modules`` 是**一条真实 HF
模块名正则**，native 与 ms-swift 两个后端都按这**同一条**表达式确定更新哪些权重
（ms-swift 走官方 ``--target_regex``，见 ``msswift/_config_mapping.py``；native 在
建模时用它对每个 ``LoRALinear.hf_module_path`` 做 ``re.fullmatch``）。
正则只有本模块一处定义（:data:`LORA_TARGET_MODULES_REGEX`），⛔ 不得在别处复制字面量。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

#: ★★ 两个后端的 LoRA 目标 —— **唯一真相源**（一条真实 HF 模块名正则）。
#:
#: 覆盖语言侧（``model.language_model.layers.N.<self_attn|linear_attn|mlp>.*``）与
#: 视觉侧（``model.visual.{merger,blocks.N}.*``）；两侧都是**真实 HF 模块名**
#: （与 ``transformers/models/qwen3_5/modeling_qwen3_5.py:1244-1245`` 的
#: ``model.visual`` / ``model.language_model`` 两棵子树逐字对应），因此同一串可直接
#: 喂给 ms-swift 的 ``--target_regex``（PEFT 按 ``re.fullmatch`` 逐模块名匹配），
#: native 亦按 ``LoRALinear.hf_module_path`` 逐模块匹配 ⇒ 两侧目标集合同源。
#:
#: ★ 语言侧枚举里**不含** ``in_proj_a`` / ``in_proj_b``：保留这两项会让两个后端的
#: **权重矩阵集合**不等（仅正则命中 in_proj_a 24 + in_proj_b 24 = 48 ⇒ 358 vs 310）。
#: 去掉后两侧同为 310（语言 200 + 视觉 110）。
LORA_TARGET_MODULES_REGEX: str = (
    r"^(?:model\.)?language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_z|out_proj"
    r"|in_proj_qkv|gate_proj|up_proj|down_proj)$"
    r"|^(?:model\.)?visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc1|mlp\.linear_fc2)"
    r"|merger\.linear_fc1|merger\.linear_fc2)$"
)

#: 正则元字符集：模块名 / 预设名里都不会出现。用于把"形如正则"与"裸名字"分开
#: （fail-closed：裸名字 = 旧 preset 名或漏写锚点的模块名，一律拒绝，不给别名）。
_REGEX_METACHARACTERS = frozenset("^$()[]{}|*+?\\")

#: **视觉命名空间**的代表性真实模块名（fail-closed 探针）。
#:
#: 配置期判"这条正则能不能训到视觉塔"时模型还没加载，只能拿真实命名空间的代表名
#: 试匹配（与 ``eval/merged_export.py`` 的 ``MULTIMODAL_NAMESPACE_PROBE_NAMES``
#: 同一手法、同一理由）。名字取自 Qwen3.5 真实 HF 命名空间：视觉塔挂在
#: ``model.visual``，merger 2 个 + 每个 block 4 个 ``nn.Linear``
#: （``merger.linear_fc{1,2}``、``attn.qkv``、``attn.proj``、``mlp.linear_fc{1,2}``）。
VISUAL_MODULE_PROBES: tuple[str, ...] = (
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
)

#: ⚠ **遗留表**（legacy）：native 侧的旧 preset 名 → glob 模式。
#:
#: ``resolve_lora_target_modules`` **已不再消费它**（preset 名一律 fail-closed 拒绝，
#: 见 :func:`compile_lora_target_regex`）。保留的唯一原因：**上一世代（已冻结、本轮
#: 不动）**的矩阵 yaml ``samples/configs/matrix54/`` 与其生成器
#: ``tests/e2e/generate_matrix.py`` 仍按 ``lora.target_preset`` 产出/校验，相关既有测试
#: 从本表取"具名预设"真值（``tests/e2e/test_generate_matrix.py``）——它是那份冻结产物的
#: 契约，**不是**新路径的别名。新配置一律用 :data:`LORA_TARGET_MODULES_REGEX`。
LORA_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "language_safe": (
        "language.*.q_proj",
        "language.*.v_proj",
    ),
    "language_all_linear": (
        "language.*.q_proj",
        "language.*.k_proj",
        "language.*.v_proj",
        "language.*.o_proj",
        "language.*.in_proj_z",
        "language.*.out_proj",
        "language.*.gate_proj",
        "language.*.up_proj",
        "language.*.down_proj",
    ),
    "vision_merger": (
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
    ),
    "vision_common": (
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
        "visual.blocks.*.attn.*",
        "visual.blocks.*.mlp.linear_fc*",
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedLoRATargets:
    requested: tuple[str, ...]
    resolved: tuple[str, ...]

    @property
    def signature(self) -> dict[str, object]:
        return {
            "requested": list(self.requested),
            "resolved": list(self.resolved),
        }


def looks_like_lora_target_regex(target_modules: str) -> bool:
    """该串是否"形如正则"（含模块名 / 预设名里不会出现的正则元字符）。"""
    return isinstance(target_modules, str) and any(
        ch in _REGEX_METACHARACTERS for ch in target_modules
    )


def compile_lora_target_regex(target_modules: str) -> re.Pattern[str]:
    """把 ``lora.target_modules`` 编译成匹配器；**任何不认识的形式一律报错**。

    fail-closed 的三种拒绝（⛔ 不留别名、不做隐式兼容）：

    1. 非 ``str``（list/tuple/set/...）——含旧写法的 preset 名**列表**
       （``[language_all_linear, vision_common]``）；
    2. 形如"裸名字"（不含任何正则元字符）的 ``str``——含旧 preset 名
       （``language_all_linear`` / ``vision_common``）与漏写锚点的单个模块名；
    3. 正则不可编译。

    :raises ValueError: 上述任一情形
    """
    if not isinstance(target_modules, str):
        raise ValueError(
            "lora.target_modules must be a single real HF module-name regex string; got "
            f"{type(target_modules).__name__} {target_modules!r}. The old form (a list of "
            "preset names / module names, e.g. [language_all_linear, vision_common]) is no "
            "longer accepted: both backends now share one regex "
            f"(graspo.core.lora.LORA_TARGET_MODULES_REGEX = {LORA_TARGET_MODULES_REGEX!r})."
        )
    if not looks_like_lora_target_regex(target_modules):
        raise ValueError(
            "lora.target_modules must be a real HF module-name regex (with anchors / "
            f"alternation), got the bare name {target_modules!r}. Named presets "
            f"({sorted(LORA_TARGET_PRESETS)}) are no longer accepted — use the shared regex "
            f"{LORA_TARGET_MODULES_REGEX!r} or write an explicit regex."
        )
    try:
        return re.compile(target_modules)
    except re.error as exc:
        raise ValueError(
            f"lora.target_modules is not a compilable regex ({exc}): {target_modules!r}"
        ) from exc


def resolve_lora_target_modules(
    target_modules: str | None,
    *,
    available: Iterable[str],
) -> ResolvedLoRATargets:
    """把 ``lora.target_modules``（一条正则）解析成**真实 HF 模块名**集合。

    匹配域 = :func:`graspo.flow.backend.native.lora.lora_helpers.native_qwen_lora_available_targets`
    给出的真实 HF 模块名（即每个 ``LoRALinear.hf_module_path`` 的取值域），与
    ms-swift / PEFT 侧的匹配域**逐字相同** ⇒ 同一条正则 ⇒ 同一目标集合。

    :param target_modules: ``lora.target_modules``（一条 HF 模块名正则）。``None``
        = 未提供 ⇒ **fail-closed 报错**（旧 ``lora.target_preset`` 回退已废除，
        不再有"不写就用某个预设"的隐式默认）。
    :param available: 本模型可被 LoRA 的真实 HF 模块名全集。
    :raises ValueError: 目标写法不认识（见 :func:`compile_lora_target_regex`），或一条
        都没命中（拼错 / 只覆盖另一族命名空间时必报，而不是静默建出零 LoRA 的模型）
    """
    available_targets = tuple(sorted(set(str(item) for item in available)))
    if target_modules is None:
        raise ValueError(
            "lora.target_modules is required (a single real HF module-name regex; both "
            "backends share it). lora.target_preset no longer selects LoRA targets — it is "
            "kept only so pre-existing config files keep parsing. Set lora.target_modules "
            f"to the shared regex {LORA_TARGET_MODULES_REGEX!r} or an explicit regex."
        )
    pattern = compile_lora_target_regex(target_modules)
    matched = tuple(sorted(name for name in available_targets if pattern.fullmatch(name)))
    if not matched:
        sample = ", ".join(available_targets[:5]) if available_targets else "(none)"
        raise ValueError(
            f"lora.target_modules={target_modules!r} matched none of the model's LoRA-able "
            f"modules ({len(available_targets)} available, e.g. {sample}). Refusing to build "
            "a model whose LoRA target set is empty."
        )
    return ResolvedLoRATargets(requested=(target_modules,), resolved=matched)


def lora_target_modules_cover_visual(target_modules: str | None) -> bool:
    """这条正则能不能训到**视觉塔**（配置期判据，模型尚未加载）。

    用 :data:`VISUAL_MODULE_PROBES` 的真实视觉模块名做 ``re.fullmatch`` 实测——
    与 PEFT / native 的匹配语义一致。非法 / 不可编译 / 空值一律判否（fail-closed）。
    """
    if not looks_like_lora_target_regex(target_modules):  # type: ignore[arg-type]
        return False
    pattern = compile_lora_target_regex(target_modules)  # type: ignore[arg-type]
    return any(pattern.fullmatch(name) is not None for name in VISUAL_MODULE_PROBES)
