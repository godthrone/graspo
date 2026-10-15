"""输出目录的准备与防呆（**全仓唯一实现**）。

**职责**：目录存在性/非空判定、预授权后的**逐项列出 + 逐项删除**、落点重叠校验。

**本文件不负责**：不决定"要不要导出"、不管格式、不写任何产物文件。

**唯一性**：本函数升格自 ``eval/merged_export.py::prepare_output_directory`` 的**合规**
语义（先看得见、再动手；⛔ 不用通配符、⛔ 不 ``rmtree`` 整个输出根），并取代
``flow/backend/native/lora/lora_io.py::prepare_output_dir`` 的**违规**语义
（它对整个输出目录直接 ``shutil.rmtree``，未列出、未逐项删除 —— 违反 §2.4）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from graspo.export.errors import ExportError, OutputDirNotEmptyError


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_output_dir(output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """准备输出目录。默认拒绝覆盖已有数据（§3.3 预授权退路）。

    语义（唯一合规实现）：

    - 目录不存在 ⇒ 建出来；
    - 目录存在且**为空** ⇒ 直接用；
    - 目录存在且**非空**且未预授权 ⇒ 抛 :class:`OutputDirNotEmptyError`（**不动任何文件**）；
    - 目录存在且非空且已预授权 ⇒ **先逐项列出**（``entries`` 已排序）再**用具体路径
      逐项删除**（子目录按名 ``rmtree`` 该子目录、文件按名 ``unlink``），
      ⛔ 不对输出根使用通配符或整体 ``rmtree``，目录外文件不受影响。

    Args:
        output_dir: 目标目录。
        overwrite: 是否已获预授权覆盖（默认 ``False``）。

    Returns:
        目标目录 :class:`~pathlib.Path`（已存在且为空）。
    """
    root = Path(output_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return root

    if not root.is_dir():
        raise ExportError(f"output path exists but is not a directory: {root}")

    # 先逐项列出：即使不覆盖也要把"将失去什么"看清（§2.4 操作防呆）。
    entries = sorted(root.iterdir())
    if not entries:
        return root

    if not overwrite:
        raise OutputDirNotEmptyError(
            f"Output directory is not empty: {root} ({len(entries)} entries); "
            "refusing to overwrite — set training.overwrite_output_dir: true "
            "(or export.allow_overwrite: true) to accept losing it"
        )

    # 预授权后：按**具体路径**逐项删除（顺序已确定、条目已列出）。
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return root


def assert_destination_allowed(
    destination: str | Path,
    *,
    checkpoint_dir: str | Path,
    base_model_path: str | Path | None = None,
) -> None:
    """落点重叠校验（V12）：产物不得写进自己、也不得与基座互相包含。

    - ``destination == checkpoint_dir`` ⇒ 拒绝（会把输入覆盖掉）；
    - ``destination`` 位于基座目录内、或基座目录位于 ``destination`` 内 ⇒ 拒绝。

    ★ 允许 ``destination`` 位于 ``checkpoint_dir`` **内**：训练结束的自动导出落点就是
    ``<checkpoint_dir>/<target>``（``trainer/checkpoint.py``），这是 config 决定的确定性落点。
    """
    dest = Path(destination).resolve()
    ckpt = Path(checkpoint_dir).resolve()
    if dest == ckpt:
        raise ExportError(
            f"output directory must not be the checkpoint directory itself: {dest}"
        )
    if base_model_path:
        base = Path(base_model_path).resolve()
        if dest == base or _is_relative_to(dest, base):
            raise ExportError(
                f"output directory must not be the base model directory nor inside it: {dest}"
            )
        if _is_relative_to(base, dest):
            raise ExportError(
                f"output directory must not contain the base model directory: {dest}"
            )


__all__ = ["assert_destination_allowed", "prepare_output_dir"]
