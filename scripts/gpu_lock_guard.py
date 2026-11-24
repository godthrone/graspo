#!/usr/bin/env python3
"""锁卡守卫命令行入口——任何训练/测试入口在启动前调用，fail-closed。

职责：读取 ``NVIDIA_VISIBLE_DEVICES``（或 ``--visible`` 显式取值），调用
``graspo.core.gpu_guard`` 校验，通过则打印确认行并退出 0，拒绝则打印可操作
的错误信息并以退出码 1 终止。**不启动任何进程、不碰 GPU**。

设备来源两种（F-1，见 ``core/gpu_guard`` 模块 docstring）：

- ``--visible`` / 环境变量是**显式宿主卡号** → 静态比对，全部判据 = 三条：
  **① 必须显式设置；② 卡号 ⊆ 允许集合（默认 0–7 = 本机全卡，可用
  ``GRASPO_ALLOWED_GPU_INDICES`` 收窄）；③ 无空元素 / 重复卡**。
  ★ **本层不做任何"禁用卡号"过滤**，**也不判卡数** —— 既不按卡号判"邻居卡"，
  也不判"生产锁卡"，更不判"单作业卡数上限"。
- 环境变量是 runtime 哨兵 ``void`` → **不提供** ``--probe-inventory`` 时拒绝
  （本脚本要的是真实卡号；容器内请用 ``python -m graspo record-gpu-memory --idle-only``
  或训练入口的 ``require_gpu_lock_or_exit``，它们会实测可见卡）。

★★ 2026-10-17 主席裁定（唯一标尺，已脱敏转述）：「**远端机先到先得，随便用。**」
＋「**单个作业不限制卡数。**」⇒ 本守卫**只有一条**卡号判据（**存在性**：卡号 ∈ 0–7）
且**没有卡数上限**：
  · ⛔ 无预留卡、⛔ 无禁用卡、⛔ 无"生产锁卡"、⛔ 无"邻居卡"、⛔ 无单作业卡数上限；
  · 生成/清单侧同此口径（``gpu_assignment.GPU_ALLOWED = (0..7)``；生成器只判
    "卡号 ∈ 0–7 ∧ 卡数 == 该档 cards"，见 ``generate_matrix_manifest_source.card_range_violation``）；
  · ★ 唯一的**硬要求 =「不抢占他人」**，它是**动态事实**判据、住在**设施层**：
    起跑前**逐卡实测空闲** = **剩余显存 ≥ 本档头寸（默认 48 GiB）∧ util == 0**
    （``scripts/gpu_idle_assert.py`` → ``graspo.flow.guard.idle``），被占则**换卡或等**，
    ⛔ **绝不 kill 他人进程 / 绝不停他人容器**。
  · 因此"某张卡此刻被谁占着"**不是**静态禁用问题：本层不猜、不拦，由上面那道实测断言负责。

用法：
    NVIDIA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/gpu_lock_guard.py
    python3 scripts/gpu_lock_guard.py --visible 0,1        # 宿主侧预检
    python3 scripts/gpu_lock_guard.py --quiet              # 只靠退出码判断
    python3 scripts/gpu_lock_guard.py --visible 0,1 --assert-idle   # 再断言实测空闲

本脚本是被 run.sh / run_matrix 反复调用的稳定入口，属宪法 §10.2 的
"制度化接口" ⇒ **已升级登记为 ``graspo gpu-lock-guard`` 稳定子命令**
（``src/graspo/cli/devops.py``，同一份实现、参数透传）。宿主/镜像引导层
（尚未建 venv / 未安装 graspo）仍走本脚本路径（``run.sh:114`` 是稳定契约，
不得迁址）；两者契约见 ``docs/devops-cli.md``。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"


def _install_namespace_shim() -> None:
    """让 ``graspo.*`` 的**纯逻辑子模块**在裸环境（无 torch）可按路径导入。

    ``graspo/__init__.py`` 会拉 ``core.schema`` → ``ripple`` → ``torch``；因此把
    ``graspo`` / ``graspo.core`` / ``graspo.flow`` 注册为命名空间包（``__path__``
    指向真实目录、**不执行它们的 ``__init__``**）。P0 的「gpu_guard 双家消除」把
    设施操作搬到 ``graspo.flow.guard``（见 ``flow/guard/__init__.py`` docstring）
    ⇒ 垫片必须覆盖 ``graspo.flow``；``graspo.flow.guard`` 本身仍走正常导入以执行
    其 ``__init__.py``（它只依赖 ``graspo.core.gpu_guard``，不碰 torch）。
    """
    import types

    for name, relative in (
        ("graspo", "graspo"),
        ("graspo.core", "graspo/core"),
        ("graspo.flow", "graspo/flow"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / relative)]
        module.__package__ = name
        sys.modules[name] = module


def _load_guard_module() -> ModuleType:
    """加载 ``graspo/flow/guard/__init__.py`` 的守卫**设施**实现。

    守卫是任何训练/测试之前的第一道防线，必须在**裸环境**可用——不能因为
    ``graspo/__init__.py`` 会导入 torch/pydantic 就跟着失败。因此优先按文件
    路径独立加载（不经过包 ``__init__``）；文件不存在时才退回常规包导入
    （安装到 site-packages 的场景）。实现仍只有一份（§1.4 单一真相源）。

    ⚠ 这里加载的是 ``flow/guard``（P0 后的唯一设施入口）——它 re-export 了本脚本
    用到的 ``assert_gpu_lock`` / ``assert_gpu_lock_from_env`` / ``format_verdict``
    / ``GpuLockError``。改前指 ``core/gpu_guard.py``（只剩纯策略，无
    ``format_verdict``）⇒ ``run.sh:114`` 会以 ``AttributeError`` 拒绝起跑（静默
    失效，B3 冒烟抓出并修正）。
    """
    _install_namespace_shim()
    source = _SRC / "graspo" / "flow" / "guard" / "__init__.py"
    if source.is_file():
        spec = importlib.util.spec_from_file_location("_graspo_gpu_guard_facility", source)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            # 登记到 sys.modules，保证模块内 dataclass 等反射逻辑可解析 __module__。
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    import graspo.flow.guard as installed  # noqa: PLC0415

    return installed


_guard: Any = _load_guard_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed GPU lock guard: reject unset/all/out-of-range device sets "
            "(card indices must be a subset of the machine's 0-7; no per-job card-count cap)."
        ),
    )
    parser.add_argument(
        "--visible",
        default=None,
        help=(
            "Explicit device list to validate instead of reading the "
            "NVIDIA_VISIBLE_DEVICES environment variable (host-side pre-check)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Print nothing on success.")
    parser.add_argument(
        "--assert-idle",
        action="store_true",
        help=(
            "F-10/B10: after the lock check, also assert every target GPU is available — "
            "free memory >= this run's position (default 48 GiB, GRASPO_GPU_POSITION_MIB) "
            "AND utilization.gpu == 0 over consecutive samples. "
            "Never kills other processes or stops other containers."
        ),
    )
    return parser


def _idle_monitor() -> ModuleType:
    """按文件路径加载空闲断言实现（``cli/gpu_monitor``，只依赖 core.gpu_guard）。

    与 :func:`_load_guard_module` 同因：必须在**裸环境**（无 torch）可用，因此
    预注册 ``graspo`` / ``graspo.core`` 命名空间包，绕过会拉 torch 的包 ``__init__``。
    """
    import types

    src = Path(__file__).resolve().parents[1] / "src"
    for name, relative in (("graspo", "graspo"), ("graspo.core", "graspo/core")):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(src / relative)]
        module.__package__ = name
        sys.modules[name] = module
    source = src / "graspo" / "cli" / "gpu_monitor.py"
    spec = importlib.util.spec_from_file_location("_graspo_gpu_monitor_standalone", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"找不到空闲断言实现：{source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.visible is None:
            devices = _guard.assert_gpu_lock_from_env()
        else:
            devices = _guard.assert_gpu_lock(args.visible)
    except _guard.GpuLockError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not args.quiet:
        print(_guard.format_verdict(devices))

    if args.assert_idle:
        try:
            _idle_monitor().assert_visible_gpus_idle(visible=args.visible)
        except _guard.GpuLockError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"目标卡空闲断言失败（fail-closed）：{exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
