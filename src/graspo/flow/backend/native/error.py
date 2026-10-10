"""``flow.backend.native`` 的共享异常类型（§12.2：单类文件的文件名必须编码类名的功能部分）。

**职责**：只定义 ``AssignmentError`` 一个异常类，不做任何分配逻辑。
分配逻辑（``assign_whole_groups`` / ``assert_balanced_assignment`` /
``plan_group_rebalance`` / ``step_group_quota`` / ``whole_group_count``）在
``assignment.py``——它 200 行的主体是纯分配函数，异常只是附带物，因此按 §12.2
把类抽到本文件，文件名 ``error`` 是 ``assignment_error`` 的下划线后缀。
``assignment.py`` 留一行 import 把类接回（下游
``from graspo.flow.backend.native.assignment import AssignmentError`` 零改动）。

**本文件不 import 任何东西**（无依赖 ⇒ 不可能引入导入环）。
"""


class AssignmentError(RuntimeError):
    """分配不满足 β1b 不变量（整组不拆 / 组集合不变 / 均衡）。"""
