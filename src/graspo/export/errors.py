"""导出失败的公开类型契约。

**职责**：让调用方（``cli`` / ``eval`` / ``scripts``）只 import 本模块即可捕获
**全部**导出失败，而不触发任何 ``_load`` / ``_write`` 的懒加载路径（§1.3 计算与设施分离）。

**本文件不负责**：不读文件、不 import torch/peft/transformers，不含任何形态知识。
"""

from __future__ import annotations


class ExportError(RuntimeError):
    """导出失败（fail-closed）：拒绝即拒绝，不给"看起来正常"的产物留通道（§2.3）。"""


class OutputDirNotEmptyError(FileExistsError, ExportError):
    """输出目录非空且未获预授权覆盖（§3.3 预授权退路）。

    **为什么是两个基类**：本异常有两个既有调用方，各自按**不同**类型分派，而
    "非空即拒绝"的判定与"逐项删除"的动作必须只有一份实现（§1.4）：

    - 训练侧 ``prepare_output_dir_once`` 用 ``except FileExistsError`` 把该判决在
      rank 间广播（status=1 → 全体 rank 抛 ``FileExistsError``）；
    - 导出侧调用方只捕获 ``ExportError``（单一导出失败类型）。

    因此它是一个 ``ExportError`` 的**特化**：调用方仍只需捕获 ``ExportError``，
    训练侧的多 rank 广播契约也不变。⛔ 业务代码不得 catch 本类做额外分支。
    """


__all__ = ["ExportError", "OutputDirNotEmptyError"]
