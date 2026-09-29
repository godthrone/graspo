"""评测链路的异常类型（§12.2：单类文件的文件名必须编码类名的功能部分）。

**职责**：只定义 `OrchestrationError` 一个异常类，不做任何编排逻辑。
编排逻辑（`load_eval_dataset` / `resolve_eval_target` / `run_evaluation` /
`write_report`）在 `orchestrator.py`——它 471 行的主体是编排函数，异常只是附带物，
因此按 §12.2 把类抽到本文件，文件名 `error` 是 `orchestration_error` 的下划线后缀。

**为什么是单数 `error.py` 而不是 `errors.py`**：本文件只含 1 个顶层类，
checker 要求文件名是 `snake(类名)` 的下划线后缀；`errors` 不是 `orchestration_error`
的后缀 ⇒ 必须用单数才能通过。若日后把同类异常也搬进来（≥2 个顶层类），
则走多类文件豁免，`error.py` 仍然合法。

**本文件不 import 任何东西**（无依赖 ⇒ 不可能引入导入环）。
"""

from __future__ import annotations


class OrchestrationError(RuntimeError):
    """编排失败。调用方应终止并把这条信息原样报给上级。"""
