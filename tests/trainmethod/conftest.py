"""``tests/trainmethod`` 的收集期垫片：让 CPT / OPD 通道的**纯逻辑**测试在本机无 torch 时可跑。

**为什么要它**：``graspo/__init__.py`` → ``graspo.ripple.algorithm_core`` → ``torch``；
``graspo/flow/__init__.py`` → ``graspo.flow.runtime`` → ``torch``。本机（开发机）没有
torch，因此任何 ``import graspo.*`` / ``import graspo.flow.*`` 都会在**收集期**失败。
但本目录断言的全是**纯计算**：配置模型校验、``(train_method, backend)`` 路由解析、
graspo 配置 → ms-swift 参数向量的映射、数据集行形态——这些必须在 CPU 上独立可测
（宪法 §1.3 层次边界）。

**垫片策略（非破坏式，宪法 §2 防呆，与 ``tests/eval/conftest.py`` 同一手法）**：
只把相关**包**命名空间的 ``__path__``/``__spec__`` 指向**真实源码目录**，让包的
``__init__.py`` 不被执行、而子模块照常可导入。除此之外什么都不造：

- **不覆盖** ``sys.modules`` 里已存在的条目（别人装好的保持原样，不产生冲突）；
- **不伪造叶子模块**——尤其不能把 ``graspo.ripple.reward.reward`` 换成假实现，
  那会同时①遮蔽真 ``REWARD_REGISTRY``、②让 ``graspo.ripple.*`` 的子模块对别的
  测试文件不可见；
- **只在 torch 不可用时安装**——容器内（torch 存在）行为完全不变。

**范围**：只影响 ``tests/trainmethod/``。不触碰上级 ``tests/conftest.py``
（属测试基建工作包，避免与其冲突）。
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "graspo"

#: 需要"跳过 __init__、但保留真实 __path__"的包命名空间（模块名, 相对 src/graspo 的目录）。
#: 覆盖本目录用到的三条导入链：
#:   schema  → graspo.core.schema（它经 graspo.ripple.reward.reward 取奖励注册表）
#:   mapping → graspo.flow.msswift._config_mapping（纯计算）+ _rope_compat（纯计算）
#:   dataset → graspo.flow.msswift.dataset（+ ripple.parsing / ripple.multimodal 的同源函数）
_PACKAGE_NAMESPACES: tuple[tuple[str, str], ...] = (
    ("graspo", "."),
    ("graspo.core", "core"),
    ("graspo.ripple", "ripple"),
    ("graspo.ripple.reward", "ripple/reward"),
    ("graspo.ripple.parsing", "ripple/parsing"),
    ("graspo.ripple.multimodal", "ripple/multimodal"),
    ("graspo.flow", "flow"),
    ("graspo.flow.msswift", "flow/msswift"),
)


def _ensure_namespace(name: str, source_dir: Path) -> None:
    """确保 ``name`` 是指向**真实源码目录**的包命名空间；已存在则原样保留内容。

    **父包属性也要补**：``sys.modules`` 里有 ``graspo.flow`` 不等于 ``graspo`` 模块上
    有 ``flow`` 属性——``monkeypatch.setattr("graspo.flow.x.y", ...)`` 这类按字符串
    解析目标的调用会走 ``getattr(graspo, "flow")``，父属性缺失时直接
    ``AttributeError: module 'graspo' has no attribute 'flow'``（全量跑测试时实测踩到：
    别的垫片先把 ``graspo`` 放进 ``sys.modules``，子包属性却没人补）。
    这里只补**缺失**的属性，已存在的一律不动（非破坏式）。
    """
    if name in sys.modules:
        module = sys.modules[name]
    else:
        module = ModuleType(name)
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
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attribute, None) is None:
            setattr(parent, attribute, module)


def _install_pure_import_shim() -> None:
    for name, relative in _PACKAGE_NAMESPACES:
        source_dir = _SRC_ROOT if relative == "." else _SRC_ROOT / relative
        _ensure_namespace(name, source_dir)


if importlib.util.find_spec("torch") is None:  # pragma: no cover - 取决于运行环境
    _install_pure_import_shim()
