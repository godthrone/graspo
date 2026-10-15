"""F1：HF 基座模型（已是可服务模型）的导出器。

**职责**：识别"``config.json`` + 权重文件"的可服务目录，并给出**显式回执** ——
对 ``merged-hf`` 而言导出是 **no-op**（产物就是它自己，不是错误、也不假装做了合并）。

**本文件不负责**：不做任何写盘、不加载权重、不 import torch。

**为什么不是"报错"**：把"没有任何事情要做"报成失败会诱使操作者手工搬文件（§2.4 反模式）；
但必须给出**显式回执**，⛔ 不能静默成功（§2.1 契约）。
"""

from __future__ import annotations

from pathlib import Path

from graspo.export.base import Exporter
from graspo.export.errors import ExportError
from graspo.export.schema import (
    CheckpointClassification,
    CheckpointForm,
    ExportResult,
    ExportTarget,
    ShardPlan,
)

#: 权重文件候选（与 ``eval/merged_export.py`` 的 ``WEIGHT_FILE_CANDIDATES`` 同源语义）。
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_INDEX_NAME = "model.safetensors.index.json"


def _weight_files(root: Path) -> list[str]:
    names = sorted(
        item.name
        for item in root.iterdir()
        if item.is_file() and item.suffix in _WEIGHT_SUFFIXES
    )
    if not names and (root / _INDEX_NAME).is_file():
        names = [_INDEX_NAME]
    return names


class BaseModelExporter(Exporter):
    """HF 基座目录 → merged-hf 的显式回执（no-op）。"""

    form = CheckpointForm.HF_BASE_MODEL
    supported_targets = frozenset({ExportTarget.MERGED_HF})

    @classmethod
    def _matches(cls, root: Path) -> tuple[bool, tuple[str, ...]]:
        config = root / "config.json"
        if not config.is_file():
            return False, ()
        weights = _weight_files(root)
        if not weights:
            return False, ()
        evidence = [
            "found config.json",
            f"found weight file(s): {', '.join(weights[:3])}",
        ]
        return True, tuple(evidence)

    def _plan(
        self,
        config: object,
        classification: CheckpointClassification,
        *,
        base_model_path: str | None = None,
    ) -> ShardPlan:
        root = Path(classification.path)
        target = ExportTarget.parse(str(getattr(config, "export_format", "")))
        return ShardPlan(
            form=self.form,
            target=target,
            checkpoint_name=root.name,
            source_dir=str(root),
        )

    def _is_passthrough(self, plan: ShardPlan) -> bool:
        return plan.target is ExportTarget.MERGED_HF

    def _passthrough_result(
        self, plan: ShardPlan, classification: CheckpointClassification
    ) -> ExportResult:
        root = Path(classification.path)
        weights = _weight_files(root)
        if not weights:
            raise ExportError(
                f"{root} was classified as an HF base model but has no weight file; refusing "
                "to report success on a directory that cannot be served"
            )
        return ExportResult(
            destination=str(root),
            target=plan.target,
            manifest_path=None,
            produced_files=weights,
            consumed_shards=[],
        )

    # 以下三个钩子对 passthrough 形态不可达；实现为 fail-closed 而不是静默返回空。
    def _load(self, plan: ShardPlan):
        raise ExportError(f"{type(self).__name__} does not load weights (passthrough form)")

    def _assemble(self, loaded, plan: ShardPlan):
        raise ExportError(f"{type(self).__name__} does not assemble weights (passthrough form)")

    def _write(self, destination: Path, assembled, plan: ShardPlan) -> tuple[str, ...]:
        raise ExportError(f"{type(self).__name__} never writes (passthrough form)")


__all__ = ["BaseModelExporter"]
