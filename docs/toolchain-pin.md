# ★ 依赖 pin 与基线口径（2026-10-14 立）

## 基线口径（唯一真相源）
- **运行时 = 镜像**（`GRASPO_IMAGE`），训练/评测/导出**一律在镜像内执行**；
- **宿主只跑纯 Python 工具**（`scripts/`、`tools/matrix/`）。

## 已实测的宿主边界（2026-10-14）
```
$ python3 -c 'import torch'   ⇒ ModuleNotFoundError
$ python3 -c 'import pytest'  ⇒ ModuleNotFoundError
$ python3 -V                  ⇒ Python 3.12.3
$TREE.hostdeps 只是一个 PYTHONPATH 目录（内含 pydantic 等若干 vendored 包）
```
⇒ **需 torch/pytest 的校验（如 `test_merged_export_classify.py` 的转换类用例）
一律在镜像内执行**；本地跑该文件时会 `importorskip` 跳过（保持本地 gate 绿）。

## 待办
- 锁矩阵工具链的**纯 Python 依赖**（`requirements`/`pyproject` 约束 ＋ 校验脚本）；
- ⛔ 不以"某次恰好装到的版本"为基线。

## 基线解释器（2026-10-14 补，实测驱动）
**基线解释器 = `.venv/bin/python`**（工具链自测亦用 `.venv/bin/pytest`）。

核对实测：
```
$ .venv/bin/python tools/check_env.py   ⇒ rc=0（依赖齐备）
$ python3 tools/check_env.py            ⇒ rc=1（系统解释器缺项更多 ⇒ ⛔ 易误判为工具链坏）
```
⇒ **`tools/check_env.py` 自检 `sys.prefix`/`VIRTUAL_ENV`：非 venv 运行会打印提示**。
⇒ 门禁/校验一律用 `.venv/bin/python`；⛔ 不用系统 `python3` 判"工具链缺依赖"。
