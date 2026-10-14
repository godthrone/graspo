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
