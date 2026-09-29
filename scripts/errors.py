"""`scripts/` 下脚本的共享异常类型（§12.2：单类文件的文件名必须编码类名的功能部分）。

**★ 不变量（改动本文件前必读）**：本文件**必须始终包含 ≥2 个顶层类**。
checker 的判定链是：0 类 ⇒ 跳过；≥2 类 ⇒ 多类文件豁免（§8.4）；**恰好 1 类** ⇒ 逐名比对，
此时文件名须是 `snake(类名)` 的下划线后缀。`errors` 既不是 `calibration_input_error`
也不是 `usage_error` 的后缀 ⇒ **任一异常类被移走，本文件会静默回退成 FAIL**。
若确需只留 1 个类，请改为把文件改名为 `input_error.py` / `usage_error.py` 之一，
或把该类移回其所属脚本。

**为什么把两个不相关脚本的异常放在一起**：`scripts/` 是 §10.2 定义的"共享临时代码工具"
目录。这两个脚本各自的文件名（`a4_calibrate.py` / `measure_elam_seq_len.py`）由 5+ 处
文档与 1 个契约测试硬编码，改名是净损失；而 checker 允许的单类名集合
（`error.py` / `input_error.py` / `calibration_input_error.py` / `usage_error.py`）
**没有一个能编码"a4 标定"或"测 ELAM 长度"**。因此按 §8.4 的多类豁免把它们合并到本文件，
两个脚本各自 `from errors import <X>Error` 使下游（`rig.CalibrationInputError` 等）零改动。

**本文件不 import 任何东西**（无依赖 ⇒ 脚本直接执行与 importlib 加载都能用）。
"""

from __future__ import annotations


class CalibrationInputError(ValueError):
    """池规格非法（缺字段 / 违反 M1 / 独立性与卡集纪律）。"""


class UsageError(Exception):
    """用法/环境错误 ⇒ 退出码 2（fail-closed，不产出半成品 JSON）。"""
