"""``tests/eval`` 的收集期垫片：让评测链路的**纯逻辑**测试在本机无 torch 时可跑。

**为什么要它**：``graspo/__init__.py`` → ``graspo.ripple.algorithm`` → ``torch``，
因此无 torch 的机器上任何 ``import graspo.*`` 都在收集阶段失败。但 ``graspo.eval``
里有一大半是**零设施依赖**的纯计算（口径判定、聚合、Δ 配对、checkpoint 形态识别、
锁卡规则），这些必须在 CPU 上可独立测试（宪法 §1.3 层次边界）。

**垫片策略（非破坏式，宪法 §2 防呆）**：只把相关**包**命名空间的 ``__path__``
指向**真实源码目录**，让包的 ``__init__.py`` 不被执行、而子模块照常可导入。
除此之外什么都不造：

- **不覆盖** ``sys.modules`` 里已存在的条目（别人装好的保持原样，不产生冲突）；
- **不伪造叶子模块**——尤其不能把 ``graspo.ripple.reward.reward`` 换成假实现，
  那会同时①遮蔽真 ``REWARD_REGISTRY``、②让 ``graspo.ripple.*`` 的子模块
  （``monitoring`` / ``buffer`` / ``parsing``）对**别的测试文件**不可见；
- ``__path__`` 一律指向真实目录 ⇒ 后继测试文件的 import 不受影响。

**为什么必须强调"非破坏式"**：``sys.modules`` 是**全局**状态，测试文件在收集期
留下的假条目会污染**后继测试文件**，而且报出的错误（如
``No module named 'graspo.ripple.monitoring'``）与真因完全无关——**把排查者引向
错误方向**。这条路径上踩过真实缺陷，故采用"让它不可能污染"而非"记得还原"。

**范围**：只影响 ``tests/eval/``。不触碰上级 ``tests/conftest.py``（属测试基建
工作包，避免与其冲突）。
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "graspo"

#: 需要"跳过 __init__、但保留真实 __path__"的包命名空间（模块名, 相对 src 的目录）。
_PACKAGE_NAMESPACES: tuple[tuple[str, str], ...] = (
    ("graspo", "."),
    ("graspo.core", "core"),
    ("graspo.eval", "eval"),
    ("graspo.cli", "cli"),
    ("graspo.ripple", "ripple"),
    # ``ripple/reward/__init__.py`` 会 re-export 重量级名字并触发循环导入链，
    # 因此 reward 子包也按命名空间处理（叶子 ``reward.reward`` 本身只依赖 pydantic）。
    ("graspo.ripple.reward", "ripple/reward"),
)


def _ensure_namespace(name: str, source_dir: Path) -> None:
    """确保 ``name`` 是指向**真实源码目录**的包命名空间；已存在则原样保留。

    幂等 + 非破坏：已存在的条目（无论来自上级 ``tests/conftest.py`` 还是别的
    测试文件）一律不动，因此本垫片不会与任何人的垫片打架，也不会污染后继测试。
    """
    if name in sys.modules:
        return
    module = ModuleType(name)
    # 必须装一个**真的 ModuleSpec**：`importlib.util.find_spec("pkg.sub")` 会先看
    # 父包的 `__spec__`，父包 `__spec__ is None` 时直接抛 ValueError。只设
    # `__path__` 不够——那正是"看起来像包、实际不是"的半吊子状态。
    spec = importlib.machinery.ModuleSpec(name, loader=None, origin=None, is_package=True)
    spec.submodule_search_locations = [str(source_dir)]
    module.__spec__ = spec  # type: ignore[attr-defined]
    module.__loader__ = None  # type: ignore[attr-defined]
    module.__path__ = [str(source_dir)]  # type: ignore[attr-defined]
    module.__package__ = name
    module.__doc__ = (
        f"no-torch test namespace for {name!r}: points at the real source tree so that "
        "submodules stay importable for every test file (no fake leaf module is created)."
    )
    sys.modules[name] = module
    if "." in name:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], attribute, module)


def _install_pure_import_shim() -> None:
    """安装非破坏式命名空间垫片。不安装任何假叶子模块。"""
    for name, relative in _PACKAGE_NAMESPACES:
        source_dir = _SRC_ROOT if relative == "." else _SRC_ROOT / relative
        _ensure_namespace(name, source_dir)


if importlib.util.find_spec("torch") is None:  # pragma: no cover - 取决于运行环境
    _install_pure_import_shim()
