"""按目录内容判定 checkpoint 形态 + 命中证据（**注册表驱动，纯文件系统**）。

**职责**：遍历**已登记处置**的检测器，给出形态与逐条证据；零命中/多命中都 fail-closed。

**本文件不负责**：不加载权重、不合并、不 import torch。

**为什么不是 if 链**（§9.3）：形态识别与分派同源 —— 新增一类形态只需登记，检测自动纳入；
顺序敏感的手写判定链（现状 ``eval/merged_export.py:407-456``）会在"多产物同目录"时
按顺序猜，正是上述事故的成因。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graspo.export import registry
from graspo.export.errors import ExportError
from graspo.export.schema import CheckpointClassification, CheckpointForm, UnsupportedForm


def _matcher(disposition: Any):
    if isinstance(disposition, UnsupportedForm):
        return disposition.detector
    return getattr(disposition, "_matches", None)


def classify_checkpoint(path: str | Path) -> CheckpointClassification:
    """识别 ``path`` 的形态。

    判据来自**已登记处置**的检测器（导出器 ``_matches`` / ``UnsupportedForm.detector``）：

    - **零命中** ⇒ fail-closed，错误消息列出目录 entries 与已登记形态（V1）；
    - **多命中** ⇒ fail-closed，列出全部命中形态与各自证据（⛔ 不按顺序猜，修掉 N2）；
    - **恰一命中** ⇒ 返回 :class:`~graspo.export.schema.CheckpointClassification`。
    """
    root = Path(path)
    if not root.is_dir():
        raise ExportError(f"checkpoint directory not found: {root}")

    hits: list[tuple[CheckpointForm, tuple[str, ...]]] = []
    for form, disposition in registry.iter_dispositions():
        matcher = _matcher(disposition)
        if matcher is None:
            continue
        matched, evidence = matcher(root)
        if matched:
            hits.append((form, tuple(str(item) for item in evidence)))

    if not hits:
        entries = sorted(item.name for item in root.iterdir())[:10]
        registered = ", ".join(registry.registered_form_names()) or "(none)"
        raise ExportError(
            f"cannot classify checkpoint at {root}: no registered form matched "
            f"(entries: {entries}); registered forms: {registered}"
        )

    if len(hits) > 1:
        detail = "; ".join(f"{form.value}: {' | '.join(evidence)}" for form, evidence in hits)
        raise ExportError(
            f"checkpoint at {root} matches multiple forms ({len(hits)}): {detail} — "
            "refusing to guess which one (declare the intended form explicitly)"
        )

    form, evidence = hits[0]
    return CheckpointClassification(form=form, path=str(root), evidence=evidence)


__all__ = ["classify_checkpoint"]
