"""测试基建：本机无 torch 时的"纯逻辑导入垫片"（pytest 全局 conftest）。

**问题**：`graspo/__init__.py` 与 `graspo/core/__init__.py` 都会
``from graspo.core.schema import ...``，而 schema 又会在校验 reward 时拉
``graspo.ripple`` → ``import torch``。本机（开发机）没有 torch，因此任何
``import graspo.*`` 会在**收集期**就失败——这正是"只 stub
``graspo.ripple.reward.reward``"那版垫片失效的根因：包 ``__init__`` 比被测
子模块先执行，stub 根本轮不到。

**修法**：torch 不可用时，预先把 ``graspo`` 与 ``graspo.core`` 注册成
**命名空间包**（`__path__` 指向真实目录、不执行它们的 ``__init__.py``），
于是 ``graspo.core.gpu_guard`` 这类**纯逻辑子模块**可以正常导入，而
``graspo.core.schema`` 等真正依赖 torch 的模块仍会以清晰的
``ModuleNotFoundError: No module named 'torch'`` 失败。

**边界（重要）**：
- 该垫片**只在 torch 不可用时生效**；容器内（torch 存在）行为完全不变。
- 它不改变任何源码，只影响测试进程的模块导入路径。
- 依赖 torch / transformers / 真模型的测试**不在本机跑**，必须进目标 GPU 服务器上的
  ``graspo-msswift:4.5.3`` 容器。

**测试运行策略（约定，写在此处是因为仓库的
``tests/cli/test_cli.py::test_only_readmes_are_tracked_markdown_docs`` 只允许
固定的 tracked markdown 白名单，新增 ``tests/README.md`` 会破坏它）：**

*本机（无 torch）可跑*：只依赖 stdlib 或**纯计算层**的测试。判据是
"实测在本机跑通"，不是"希望它能跑"：

    # 测试基建自身（本工作包）
    python3 -m pytest \
      tests/core/test_gpu_guard.py tests/core/test_result_judge.py \
      tests/cli/test_gpu_monitor_trusted.py tests/cli/test_gpu_monitor.py \
      tests/e2e/test_generate_matrix.py tests/e2e/test_collect_results.py \
      tests/e2e/test_check_env_versions.py \
      -q -p no:cacheprovider

    # 评测链路（tests/eval/）：**整目录都不依赖 torch**，由 tests/eval/conftest.py
    # 按文件路径加载源码原文。实测（本机无 torch）：
    #   133 collected → 129 passed / 4 failed
    #   4 个失败**全部**是缺少 PyYAML（`import yaml`），与 torch/GPU 无关；
    #   装上项目声明的 `pyyaml==6.0.3` 后即可全部通过。
    python3 -m pytest tests/eval/ -q -p no:cacheprovider

    # 注意：本机若缺 pure-python 依赖（如 pyyaml），对应用例会以
    # `ModuleNotFoundError: No module named 'yaml'` 失败——这是环境缺依赖，
    # **不是**"必须上机"；不得为让它通过而放宽断言或加假 stub。

*必须在目标 GPU 服务器上的 ``graspo-msswift:4.5.3`` 容器内跑*：任何 import
torch / transformers / peft / safetensors / ms-swift，或需要真模型、真数据、GPU
的测试（``tests/flow/``、``tests/ripple/``、``tests/core/test_schema*.py``、
``tests/cli/test_cli.py`` 等）：

    python3 scripts/gpu_lock_guard.py --visible 0 || exit 1
    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
      -v "$REPO":/work -w /work -e PYTHONPATH=/work/src \
      graspo-msswift:4.5.3 python -m pytest tests/ -q -p no:cacheprovider

**唯一"必须上机"的评测事项不是单元测试**：`merge_peft_checkpoint` 的**实际权重合并**
（真 torch/peft/transformers + 真权重）。`tests/eval/` 里没有跑真合并的用例——
`test_merged_export.py` 只覆盖分类/解析/目录准备，以及"重依赖 import 之前先做
文件系统校验"的契约。真合并属人工/上机验证项。

卡计划铁律（单一路径）：宿主侧先用 ``scripts/gpu_lock_guard.py`` 前置断言，
容器只认 ``NVIDIA_VISIBLE_DEVICES``（不再叠加 ``--gpus``，两者可能互相覆盖）；
GPU6/7 是生产卡，永不触碰。新增"纯逻辑"测试时，被测模块不得 import
torch/transformers；否则归入"容器内测试"。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

#: 需要绕过其重量级 ``__init__.py`` 的包 —— (模块名, 相对 src 的目录)
_HEAVY_PACKAGES: tuple[tuple[str, str], ...] = (
    ("graspo", "graspo"),
    ("graspo.core", "graspo/core"),
)


def torch_available() -> bool:
    """torch 是否可导入（不真正导入，避免副作用）。"""
    return importlib.util.find_spec("torch") is not None


def _install_pure_import_shim() -> None:
    if torch_available():
        return
    for name, relative in _HEAVY_PACKAGES:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / relative)]
        module.__package__ = name
        module.__doc__ = (
            f"no-torch test shim for {name!r}: bypasses the package __init__ "
            "so pure-logic submodules import without torch."
        )
        sys.modules[name] = module


_install_pure_import_shim()
