"""flow.backend.native.config — ★ native 后端的**默认值唯一真相源**。

宪法依据：
- §1.4 单一真相源：同一默认值只住一处；
- §10.1 推论 1：参与决定产物的参数必须在 config 里声明；
- §3.3/§1.3：**backend 默认值属 flow**，``core/schema.py`` 只写契约（字段名/类型/
  校验），默认值经 ``core.discovery`` 注册表从本模块注入 —— 这样 ``core`` 不再
  硬编码 ``flow`` 的模块路径（X4 边界违规）。

本模块**零重依赖**（不 import torch / transformers），可被无 torch 环境下的
注册表加载器直接导入。
"""

from __future__ import annotations

from typing import Any

#: 默认模型适配器（native）。格式 ``module:Class``（由 ``flow/runtime.py`` 解析）。
NATIVE_DEFAULT_ADAPTER = (
    "graspo.flow.backend.native.models.qwen35_36.model_adapter:Qwen35ModelAdapter"
)

#: native 侧 offload 的接线只落在 qwen35_36 适配器的 mixin 上
#: （``_Qwen35SFTTrainingMethods._build_optimizer``，WP-X2）。这是**接线范围的事实**，
#: 不是设计取舍：别的 model family 要开启，得先有同样的接线。
#: 用模块前缀而不是"逐字等于默认值"，是为了让同模块内的子类/别名路径也能通过。
NATIVE_OFFLOAD_ADAPTER_MODULE = "graspo.flow.backend.native.models.qwen35_36."

#: PP 在途 microbatch 数默认值（native 调度默认；在役值 = 2）。
DEFAULT_PP_MAX_INFLIGHT_MICROBATCHES = 2


def native_defaults() -> dict[str, Any]:
    """native 后端默认值表（供 ``graspo.native_defaults`` 注册表加载器调用）。"""
    return {
        "adapter": NATIVE_DEFAULT_ADAPTER,
        "offload_adapter_module": NATIVE_OFFLOAD_ADAPTER_MODULE,
        "pp_max_inflight_microbatches": DEFAULT_PP_MAX_INFLIGHT_MICROBATCHES,
    }


__all__ = [
    "DEFAULT_PP_MAX_INFLIGHT_MICROBATCHES",
    "NATIVE_DEFAULT_ADAPTER",
    "NATIVE_OFFLOAD_ADAPTER_MODULE",
    "native_defaults",
]
