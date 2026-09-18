#!/usr/bin/env python3
"""目标卡"实测空闲"断言命令行入口（F-10）——任何上机作业启动前调用，fail-closed。

职责：逐卡实测 ``nvidia-smi`` 的 ``memory.used`` 与 ``utilization.gpu``，任一张
目标卡 **>64 MiB 或 util>5%** 即拒绝启动（退出码 1），通过则打印确认行并退出 0。
**不启动任何进程、不 kill 任何进程、不抢卡**——宁等不抢。

为什么必须固化：这是本轮上机实战里救过场的一次断言（实测拦下了含被第三方占用的
GPU3 的 4 卡目标集）。放着不看，就会把"卡上有别人的负载"误当成可用空闲卡，
显存峰值数字与训练结果都不可信。

退出码：
    0 = 全部目标卡实测空闲；
    1 = 至少有目标卡被占（或不空闲、读数拿不到）；
    2 = 用法/环境错误（可见集非法、--visible 与 --assert-idle 组合不当等）。

用法：
    NVIDIA_VISIBLE_DEVICES=0,1 python3 scripts/gpu_idle_assert.py
    python3 scripts/gpu_idle_assert.py --visible 0,1,2,3     # 宿主侧预检
    python3 scripts/gpu_idle_assert.py --visible 0 --quiet
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"


def _install_namespace_shim() -> None:
    """让 ``graspo.*`` 的**纯逻辑子模块**在裸环境（无 torch）可按路径导入。

    与 ``scripts/gpu_lock_guard.py`` 同因：``graspo/__init__.py`` 会拉
    ``core.schema`` → ``ripple`` → ``torch``。把 ``graspo`` / ``graspo.core``
    注册为命名空间包（``__path__`` 指向真实目录、不执行它们的 ``__init__``），
    于是本脚本在**任何**环境都能做前置断言。
    """
    import types

    for name, relative in (("graspo", "graspo"), ("graspo.core", "graspo/core")):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / relative)]
        module.__package__ = name
        sys.modules[name] = module


def _load_monitor_module() -> ModuleType:
    """按文件路径加载 ``cli/gpu_monitor.py``（只依赖 ``core.gpu_guard``，无 torch）。"""
    _install_namespace_shim()
    source = _SRC / "graspo" / "cli" / "gpu_monitor.py"
    spec = importlib.util.spec_from_file_location("_graspo_gpu_monitor_standalone", source)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    import graspo.cli.gpu_monitor as installed  # noqa: PLC0415

    return installed


_monitor: Any = _load_monitor_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed idle assertion: every target GPU must show "
            "memory.used <= 64 MiB and utilization.gpu <= 5%."
        )
    )
    parser.add_argument(
        "--visible",
        default=None,
        help=(
            "Explicit target GPU list (host indices) to assert instead of reading "
            "NVIDIA_VISIBLE_DEVICES (host-side pre-check)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Print nothing on success.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from graspo.core.gpu_guard import GpuLockError

    try:
        _monitor.assert_visible_gpus_idle(visible=args.visible)
    except GpuLockError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # 读数拿不到就 fail-closed——不能把"测不到"当成"空闲"。
        print(f"目标卡空闲断言失败（fail-closed）：{exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print("[gpu-idle] OK: 目标卡实测空闲（memory.used ≤64 MiB 且 util ≤5%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
