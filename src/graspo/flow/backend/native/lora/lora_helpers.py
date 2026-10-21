"""Pure helpers for LoRA target module resolution.

Extracted from ``lora.py`` (Type B helpers):
these functions don't depend on ``LoRALinear``'s state and are independently testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graspo.flow.backend.native.models.common.native_qwen_config import NativeQwenConfig

#: 视觉塔在 **HF 命名空间**下的前缀（``modeling_qwen3_5.py:1244`` 的 ``model.visual``）。
#: native 与 ms-swift 都用它——native 的视觉 ``LoRALinear`` 记的就是
#: ``lora_linear.py::_replace_visual_lora_modules`` 里的 ``f"model.visual.{module_path}"``。
VISUAL_HF_PREFIX = "model.visual."


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
      ``in_proj_qkv`` 在 native 侧被切成 q/k/v **三片**，但**模块名只有一个**）；
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
