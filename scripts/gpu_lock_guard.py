#!/usr/bin/env python3
"""锁卡守卫命令行入口——任何训练/测试入口在启动前调用，fail-closed。

职责：读取 ``NVIDIA_VISIBLE_DEVICES``（或 ``--visible`` 显式取值），调用
``graspo.core.gpu_guard`` 校验，通过则打印确认行并退出 0，拒绝则打印可操作
的错误信息并以退出码 1 终止。**不启动任何进程、不碰 GPU**。

设备来源两种（F-1，见 ``core/gpu_guard`` 模块 docstring）：

- ``--visible`` / 环境变量是**显式宿主卡号** → 静态比对（含 6/7 或 >4 卡即拒绝）；
- 环境变量是 runtime 哨兵 ``void`` → **不提供** ``--probe-inventory`` 时拒绝
  （本脚本要的是真实卡号；容器内请用 ``python -m graspo record-gpu-memory --idle-only``
  或训练入口的 ``require_gpu_lock_or_exit``，它们会实测可见卡）。

用法：
    NVIDIA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/gpu_lock_guard.py
    python3 scripts/gpu_lock_guard.py --visible 0,1        # 宿主侧预检
    python3 scripts/gpu_lock_guard.py --quiet              # 只靠退出码判断
    python3 scripts/gpu_lock_guard.py --visible 0,1 --assert-idle   # 再断言实测空闲

本脚本是被 run.sh / run_matrix 反复调用的稳定入口，属宪法 §10.2 的
"制度化接口"；若 README / docs 开始引用它，应升级为 CLI 子命令。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_guard_module() -> ModuleType:
    """加载 ``graspo/core/gpu_guard.py`` 的守卫实现。

    守卫是任何训练/测试之前的第一道防线，必须在**裸环境**可用——不能因为
    ``graspo/__init__.py`` 会导入 torch/pydantic 就跟着失败。因此优先按文件
    路径独立加载（不经过包 ``__init__``）；文件不存在时才退回常规包导入
    （安装到 site-packages 的场景）。实现仍只有一份（§1.4 单一真相源）。
    """
    source = Path(__file__).resolve().parents[1] / "src" / "graspo" / "core" / "gpu_guard.py"
    if source.is_file():
        spec = importlib.util.spec_from_file_location("_graspo_gpu_guard_standalone", source)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            # 登记到 sys.modules，保证模块内 dataclass 等反射逻辑可解析 __module__。
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    import graspo.core.gpu_guard as installed  # noqa: PLC0415

    return installed


_guard: Any = _load_guard_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed GPU lock guard: reject unset/all/out-of-range/>4-card device sets."
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
            "F-10: after the lock check, also assert every target GPU is idle "
            "(nvidia-smi memory.used <= 64 MiB and utilization.gpu <= 5%%). "
            "Never kills other processes."
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
