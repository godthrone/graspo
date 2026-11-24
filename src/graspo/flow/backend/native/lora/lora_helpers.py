"""Pure helpers for LoRA target module resolution.

Extracted from ``lora.py`` (Type B helpers):
these functions don't depend on ``LoRALinear``'s state and are independently testable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graspo.flow.backend.native.models.common.native_qwen_config import NativeQwenConfig

#: 视觉塔在 **HF 命名空间**下的前缀（``modeling_qwen3_5.py:1244`` 的 ``model.visual``）。
#: native 与 ms-swift 都用它——native 的视觉 ``LoRALinear`` 记的就是
#: ``lora_linear.py::_replace_visual_lora_modules`` 里的 ``f"model.visual.{module_path}"``。
VISUAL_HF_PREFIX = "model.visual."

#: 真实 HF 模块名里的**解码层号**段（``model.language_model.layers.12.…`` /
#: ``model.layers.12.…``）；视觉塔走 ``model.visual.blocks.N.…``，不含 ``layers.``。
_LAYER_HF_PATH_RE = re.compile(r"\.layers\.(\d+)\.")


def lora_target_layer_index(hf_module_path: str) -> int | None:
    """该目标属于**第几层**解码器；不含层号（embeddings / lm_head / 视觉塔）返回 ``None``。"""
    match = _LAYER_HF_PATH_RE.search(str(hf_module_path))
    return int(match.group(1)) if match is not None else None


def stage_exempt_lora_targets(
    targets: Iterable[str],
    *,
    local_layer_indices: Iterable[int] | None,
    is_pipeline: bool,
) -> frozenset[str]:
    """PP（``pp_size>1``）下把**本 stage 不持有**的那些层的 target 从判据域里剔除。

    **为什么需要（缺陷 N2 真因）**：``native_qwen_lora_available_targets`` 枚举的是
    **整模型**的 LoRA 可命中模块（正则的匹配域，与 ms-swift/PEFT 逐字相同），而
    pipeline 的每个 stage 只构造 ``placement.local_layer_indices`` 里的层
    （``model.py`` 只按这些下标建 ``TensorParallelQwen35DecoderLayer``）。于是
    ``enabled_lora_hf_module_paths()`` 在 stage k 上**结构性**只含本 stage 的层，
    拿它与整模型的 ``resolved`` 相减，必然把**别的 stage 的层**误报成"本机未实现"
    ⇒ ``pp_size>1`` 的档一律在建模型后崩（远端机实测 22 档 native+pp>1）。

    **判据（不得放宽）**：只剔除**有层号且该层号不在本 stage** 的目标。
    - 有层号且在本 stage ⇒ 留在"必须命中"域（真该建的没建，照旧 fail-closed）；
    - 无层号（embeddings / lm_head / ``model.visual.*``）⇒ 一律**不**剔除
      （视觉塔由 ``exempted_visual_targets`` 那套 ``include_embeddings`` 判据管，
      两者各自同域、互不替代）；
    - ``pp_size==1`` ⇒ 恒空集（本 rank 持有全部层，什么都不豁免）。

    因此"空集/漏配"不会因本函数而静默通过：每条 target 恰好归属某一个层，也就恰好
    由某一个 stage 负责建成——豁免只把"**谁**该建"摊销到正确的 stage。

    :param targets: 一次解析得到的真实 HF 模块名集合（``ResolvedLoRATargets.resolved``）。
    :param local_layer_indices: 本 stage 持有的层号（``NativePlacementPlan.local_layer_indices``）。
    :param is_pipeline: 是否 ``pp_size>1``（``NativePlacementPlan.is_pipeline``）。
    """
    if not is_pipeline:
        return frozenset()
    owned = {int(idx) for idx in (local_layer_indices or ())}
    exempted: set[str] = set()
    for target in targets:
        layer_index = lora_target_layer_index(str(target))
        if layer_index is not None and layer_index not in owned:
            exempted.add(str(target))
    return frozenset(exempted)


def native_qwen_lora_available_targets(
    hf_config: NativeQwenConfig,
) -> tuple[str, ...]:
    """native 侧可被 LoRA 的模块 —— **真实 HF 模块名**全集（正则的匹配域）。

    逐个对应建模时真的会构造出来的 ``LoRALinear.hf_module_path``
    （``layers.py`` / ``layers_qwen3.py`` / ``lora_linear.py`` 的注入点），
    覆盖三个后缀族：

    - 全注意力层 ``layers.N.self_attn.{q,k,v,o}_proj``（HF 侧也是 ``self_attn``，
      ``modeling_qwen3_5.py:764``；native 内部把这一族的 **target 名**叫
      ``language.full_attn.*``，但 **hf 路径**与 HF 一致用 ``self_attn``）；
    - 线性注意力层 ``layers.N.linear_attn.{in_proj_qkv,in_proj_z,out_proj,in_proj_b,in_proj_a}``
      （``modeling_qwen3_5.py:433-436`` 恰好这 5 个 ``nn.Linear``；其中
      ``in_proj_qkv`` 在 native 侧是**一个**权重矩阵 ⇒ **一个** ``LoRALinear``
      ——整块一份低秩，2026-10-21 起不再切成 q/k/v 三片各配一份）；
    - ``layers.N.mlp.{gate,up,down}_proj``；视觉侧 ``model.visual.merger.linear_fc{1,2}``
      + 每个 block 的 ``attn.{qkv,proj}`` / ``mlp.linear_fc{1,2}``。

    ★ 与旧版的区别（本函数即"两后端同一匹配域"的落点）：返回的是**真实 HF 模块名**
    （ms-swift/PEFT 的世界），不是 native 内部的 ``language.*`` 规范名 —— 于是同一条
    ``LORA_TARGET_MODULES_REGEX`` 在两个后端选出**同一个集合**，不需要任何名字翻译表。

    :raises ValueError: 声明了视觉塔却给不出 ``vision_config.depth``（fail-closed：
    不允许"静默少枚举视觉侧"），或 ``layer_types`` 长度与层数不符（与
    ``qwen35_36/model.py:71-72`` 同一判据，不得猜）。
    """
    family = str(getattr(hf_config, "family", ""))
    if family == "qwen3":
        key_prefix = str(getattr(hf_config, "key_prefix", "model"))
        return tuple(
            f"{key_prefix}.layers.{idx}.{suffix}"
            for idx in range(int(hf_config.num_hidden_layers))
            for suffix in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
    if family == "qwen3_5_text":
        key_prefix = str(getattr(hf_config, "key_prefix", "model.language_model"))
        num_hidden_layers = int(hf_config.num_hidden_layers)
        layer_types = list(getattr(hf_config, "layer_types", []) or [])
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "qwen3_5_text layer_types length must match num_hidden_layers "
                f"(got {len(layer_types)} vs {num_hidden_layers}); cannot enumerate the "
                "LoRA-able HF module names without knowing each layer's attention kind."
            )
        targets: list[str] = []
        for idx, layer_type in enumerate(layer_types):
            layer_prefix = f"{key_prefix}.layers.{idx}"
            if layer_type == "full_attention":
                targets.extend(
                    f"{layer_prefix}.self_attn.{name}"
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                )
            elif layer_type == "linear_attention":
                targets.extend(
                    f"{layer_prefix}.linear_attn.{name}"
                    for name in (
                        "in_proj_qkv",
                        "in_proj_z",
                        "out_proj",
                        "in_proj_b",
                        "in_proj_a",
                    )
                )
            else:
                raise ValueError(f"Unsupported qwen3_5_text layer type: {layer_type!r}")
            targets.extend(
                f"{layer_prefix}.mlp.{name}"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        if bool(getattr(hf_config, "has_vision_config", False)):
            depth = int((getattr(hf_config, "vision_config", {}) or {}).get("depth") or 0)
            if depth <= 0:
                raise ValueError(
                    "model declares a vision tower (has_vision_config=True) but "
                    "vision_config.depth is missing/0; cannot enumerate the visual LoRA-able "
                    "modules. Refusing to guess."
                )
            targets.extend(
                (
                    f"{VISUAL_HF_PREFIX}merger.linear_fc1",
                    f"{VISUAL_HF_PREFIX}merger.linear_fc2",
                )
            )
            for idx in range(depth):
                targets.extend(
                    f"{VISUAL_HF_PREFIX}blocks.{idx}.{name}"
                    for name in (
                        "attn.qkv",
                        "attn.proj",
                        "mlp.linear_fc1",
                        "mlp.linear_fc2",
                    )
                )
        return tuple(targets)
    return ()


def _lora_target_enabled(lora_targets: set[str], hf_module_path: str) -> bool:
    """该模块是否被选为 LoRA 目标（**精确匹配真实 HF 模块名**，⛔ 无后缀别名）。

    ``lora_targets`` 是 ``resolve_lora_target_modules`` 的解析结果（HF 模块名集合）。
    旧实现还接受"叶子名"（``q_proj``）这种别名 —— 已删除：别名会让"同一份配置在两个
    后端选出不同集合"，正是本轮要消灭的东西（宪法 §1.4）。
    """
    return hf_module_path in lora_targets
