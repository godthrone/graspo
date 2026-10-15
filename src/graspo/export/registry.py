"""形态 → 导出器 / 显式"暂不支持"的登记与**唯一分派点**（§9.3 / §1.2）。

**登记机制只有一套**（W1）：沿用本项目既有的 entry_points 机制
（``core/discovery.py``；先例 ``flow/registry.py``、``pyproject.toml``）：

- 生产真相源 = ``pyproject.toml`` 的 ``[project.entry-points."graspo.exporters"]``；
- 开发回退 = 本模块的 :func:`register_dev_fallbacks`（未 ``pip install`` 时生效）；
- ⛔ 不存在 ``_EXPORTERS`` 字典 / 注册装饰器 / "导入即注册" 列表。

**本文件不负责**：不含任何形态知识（形态的检测逻辑在各自的导出器里，本表只登记路径）。
"""

from __future__ import annotations

from typing import Any

from graspo.core.discovery import _discover, register_dev_fallbacks
from graspo.export.errors import ExportError
from graspo.export.schema import CheckpointForm, ExportTarget, UnsupportedForm

#: entry_point 组名（登记与分派的单一真相源）。
GROUP = "graspo.exporters"

#: 开发模式回退表：form 值 → ``模块:属性``。新增一类形态 = 本表 1 行 + pyproject 1 行
#: + :class:`CheckpointForm` 1 个成员（分派逻辑 0 行）。
register_dev_fallbacks(
    GROUP,
    {
        CheckpointForm.HF_BASE_MODEL.value: (
            "graspo.export.exporters.base_model_exporter:BaseModelExporter"
        ),
        CheckpointForm.PEFT_ADAPTER.value: (
            "graspo.export.exporters.peft_adapter:PEFT_ADAPTER_UNSUPPORTED"
        ),
        CheckpointForm.NATIVE_LORA_RANK.value: (
            "graspo.export.exporters.native_lora:NativeLoraRankExporter"
        ),
        CheckpointForm.NATIVE_FULL_PARAM_RANK.value: (
            "graspo.export.exporters.native_full_param:NativeFullParamRankExporter"
        ),
    },
)


def _registry() -> dict[str, Any]:
    """返回 ``{form 值: lazy loader}``（每次现读，便于测试用 monkeypatch 构造阳性对照）。"""
    return _discover(GROUP)


def registered_form_names() -> list[str]:
    """已登记的全部形态名（用于 fail-closed 错误消息与覆盖度不变量）。"""
    return sorted(_registry())


def get_disposition(form: CheckpointForm | str) -> Any:
    """**唯一分派点**：取某形态的处置（导出器类或 :class:`UnsupportedForm`）。

    未登记 ⇒ fail-closed 并列出全部已登记形态（§9.3）。
    """
    key = form.value if isinstance(form, CheckpointForm) else str(form)
    loader = _registry().get(key)
    if loader is None:
        registered = ", ".join(registered_form_names()) or "(none)"
        raise ExportError(
            f"no export disposition registered for form {key!r}; registered forms: "
            f"{registered} — refusing to continue on an unrecognised form (register it in "
            f"pyproject.toml [{GROUP}] and graspo.export.registry dev fallbacks)"
        )
    return loader()


def iter_dispositions() -> list[tuple[CheckpointForm, Any]]:
    """按 :class:`CheckpointForm` 的声明顺序返回 ``(form, disposition)``（供形态检测遍历）。"""
    registry = _registry()
    result: list[tuple[CheckpointForm, Any]] = []
    for form in CheckpointForm:
        loader = registry.get(form.value)
        if loader is not None:
            result.append((form, loader()))
    return result


def resolve(form: CheckpointForm, target: ExportTarget) -> Any:
    """``(形态, 目标)`` 二元组解析。

    形态未登记 / 该形态是 ``UnsupportedForm`` / 该形态不支持该目标 ⇒ 均 fail-closed，
    错误消息给出**具体理由**与可选目标（§9.3）。
    """
    disposition = get_disposition(form)
    if isinstance(disposition, UnsupportedForm):
        raise ExportError(
            f"checkpoint form {form.value!r} is explicitly unsupported: {disposition.reason}"
        )
    supported = frozenset(getattr(disposition, "supported_targets", frozenset()))
    if target not in supported:
        options = ", ".join(sorted(item.value for item in supported)) or "(none)"
        raise ExportError(
            f"checkpoint form {form.value!r} cannot be exported as {target.value!r}; "
            f"supported targets: {options}"
        )
    return disposition


__all__ = [
    "GROUP",
    "get_disposition",
    "iter_dispositions",
    "registered_form_names",
    "resolve",
]
