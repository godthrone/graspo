"""``flow.backend.msswift`` 的共享异常类型（§12.2：单类文件的文件名必须编码类名的功能部分）。

**职责**：只定义 msswift 域内的异常类，不做任何业务逻辑。

**为什么合并成一个文件**：这三个异常原先各自住在 ``_ard_contract.py`` /
``_logps_shape_guard.py`` / ``group_sharding.py`` 里，而这三个文件名都不是对应异常类
``snake(类名)`` 的下划线后缀（``ard_contract`` / ``logps_shape_guard`` /
``group_sharding``）⇒ §12.2 FAIL。按既有的 ``graspo/eval/error.py`` 与
``scripts/errors.py`` 模式，把异常抽到本文件；原模块各留一行 import 把类接回
（下游 ``from ..._ard_contract import ArdContractError`` 等引用点零改动）。

**为什么是 ``error.py``**：本文件含 **3 个顶层类** ⇒ 走 checker 的多类文件豁免（§8.4），
文件名不再受"必须是 ``snake(类名)`` 后缀"约束；单数 ``error`` 同时满足未来只留 1 个类的情形。

（仅 import ``typing.Any`` 用于异常载荷的类型标注 ⇒ 不引入导入环。）
"""

from typing import Any


class ArdContractError(ValueError):
    """ARD 样本不符合共享基座契约时抛出（边界校验，宪法 §2.3 防线）。"""


class LogpsShapeGuardError(RuntimeError):
    """per-token logps 的形状在进入切片/gather 前已可判定不自洽（fail-closed）。

    只报告，不修复：持有六个量 + 两个 file:line 位置 + 可复算的预测形状。
    """

    def __init__(self, message: str, *, context: dict[str, Any]) -> None:
        super().__init__(message)
        self.context = context


class ShardingError(RuntimeError):
    """整组分片不满足 G1 不变量（V1–V3）。"""
