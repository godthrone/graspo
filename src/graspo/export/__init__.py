"""导出模块的公开面（唯一稳定接口，§3.1）。

**面 1（面向用户）**：``graspo export --config <yaml>``（CLI，参数只有 ``--config``）。
**面 2（面向内部调用者）**：本模块的 :func:`export_checkpoint` 与 :func:`classify_checkpoint`。

调用方（``cli`` / ``flow`` 训练侧 / ``eval``）只依赖本文件；新增一类形态 = 登记一处，
⛔ 不改调用方。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# 导入 registry 即完成"开发模式回退"登记（幂等；生产走 entry_points）。
from graspo.export import registry as registry
from graspo.export.classify import classify_checkpoint
from graspo.export.errors import ExportError, OutputDirNotEmptyError
from graspo.export.paths import prepare_output_dir
from graspo.export.schema import (
    CheckpointClassification,
    CheckpointForm,
    ExportResult,
    ExportTarget,
    UnsupportedForm,
)

if TYPE_CHECKING:  # 仅为类型注解；运行时不拉入 core.schema（保持公开面轻量）
    from graspo.core.schema import ExportConfig


def export_checkpoint(
    config: ExportConfig,
    *,
    base_model_path: str | None = None,
) -> ExportResult:
    """**唯一导出入口**。

    流程：校验输入定位 → 形态检测（注册表驱动）→ ``(形态, 目标)`` 解析 → 委托导出器。

    Args:
        config: 导出配置（``core.schema.ExportConfig``）；落入点、目标格式全部由它决定。
        base_model_path: 基座模型路径的**唯一来源**（设计 V10）= 调用方从
            ``config.model.model_path`` 传入；⛔ 不回落读 checkpoint 内嵌 config。

    Returns:
        :class:`~graspo.export.schema.ExportResult`。

    Raises:
        ExportError: 任何一条 fail-closed 边界被触发（未知形态、未登记处置、
            不支持的目标、分片不完整、DP 副本分歧、自校验失败……）。
    """
    checkpoint = str(getattr(config, "checkpoint_path", "") or "")
    if not checkpoint:
        raise ExportError("export.checkpoint_path is required")
    output = str(getattr(config, "export_output", "") or "")
    if not output:
        raise ExportError("export.export_output is required")

    try:
        target = ExportTarget.parse(str(getattr(config, "export_format", "")))
    except ValueError as exc:
        raise ExportError(str(exc)) from None

    classification = classify_checkpoint(checkpoint)
    exporter_cls = registry.resolve(classification.form, target)
    return exporter_cls().export(config, classification, base_model_path=base_model_path)


__all__ = [
    "CheckpointClassification",
    "CheckpointForm",
    "ExportError",
    "ExportResult",
    "ExportTarget",
    "OutputDirNotEmptyError",
    "UnsupportedForm",
    "classify_checkpoint",
    "export_checkpoint",
    "prepare_output_dir",
    "registry",
]
