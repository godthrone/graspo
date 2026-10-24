#!/usr/bin/env python3
"""目标卡"起跑前可用"断言命令行入口（F-10）——任何上机作业启动前调用，fail-closed。

职责（**三判据并取**，2026-10-25 起）：逐卡实测，任一判据不满足即拒绝启动（rc=1），
并在消息里**点名卡号与原因**：

① **显存**：``--query-gpu`` 的 ``memory.used`` ≤ ``--max-used-mib``（默认 2 GiB）
   **且** 剩余（total − used）≥ 本档头寸（``--position-mib`` 训练档默认 48 GiB；
   服务档用 ``--position-ratio 0.85`` ⇒ 头寸 = 0.85 × 该卡总显存，对齐 vLLM 的
   ``gpu_memory_utilization``）；
② **进程**：``--query-compute-apps`` 中该卡**没有任何** compute 进程（有则点名
   pid / 进程名 / 占用；进程名由 ``/proc/<pid>/cmdline`` 或 ``ps`` 补，⛔ 不用 sudo）；
③ **容器**：``docker ps`` → ``docker inspect`` 的 ``HostConfig.DeviceRequests.DeviceIDs``
   与 ``NVIDIA_VISIBLE_DEVICES`` 里**不得**出现该卡（能读到容器名则点名）；
④ **采样稳定**：``--frames`` 帧（默认 3）、间隔 ``--interval``（默认 10 s）内，
   **每帧**都满足 ①②③ 才算可用。

为什么必须三条：228 实测（2026-10-25）里 GPU0/1/4/5 被他人的**常驻推理服务**占着
13–16 GiB、GPU2 被**宿主** compute 进程占 13.3 GiB，而 util 采样**全是 0 %** ⇒
只看「剩余 ≥ 头寸 ∧ util == 0」会对 ``--visible 0,1,2,4,5`` **全部 rc=0 放行**，
起服（vLLM 需 0.85 × 总显存 ≈ 69 632 MiB）与起训必撞 OOM。GPU7 显存 0 MiB、util 0 %，
但被他人容器 ``pf228-transform-worker-1`` 声明预留 ⇒ 也只有读容器面才判得出来。

**不启动任何进程、不 kill 任何进程、不启/不停容器、不抢卡**——宁等不抢。

退出码：
    0 = 全部目标卡三判据全满足（帧帧满足）；
    1 = 至少一张目标卡不可用，或读数拿不到（fail-closed）；
    2 = 用法错误（``--position-mib`` 与 ``--position-ratio`` 同时给出等）。

用法::

    # 训练档（头寸 48 GiB）
    python3 scripts/gpu_idle_assert.py --visible 0,1
    # 服务档（vLLM：头寸 = 0.85 × 总显存）
    python3 scripts/gpu_idle_assert.py --visible 3,7 --position-ratio 0.85
    # 机读（编排脚本用）：逐卡 available/reasons/processes/containers
    python3 scripts/gpu_idle_assert.py --visible 0,1,2,3 --json --frames 3 --interval 10

宪法 §10.2：本脚本被专门测试固化（``tests/scripts/test_gpu_idle_assert.py``）
⇒ 已是**制度化接口**，**已升级登记为 ``graspo gpu-idle-assert`` 稳定子命令**
（``src/graspo/cli/devops.py``，同一份实现、参数透传）。宿主/镜像引导层仍走本
脚本路径（稳定契约，不得迁址）；两者契约见 ``docs/devops-cli.md``。
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

    # ★ P0（adapter 消歧 / gpu_guard 双家消除）后，设施操作住 `graspo.flow.guard`，
    # 因此命名空间垫片也必须覆盖 `graspo.flow` / `graspo.flow.guard`（同样不执行其
    # `__init__`，避免经 `graspo.flow.backend.native.runtime` 拉起 torch）。★ 只注册到 `graspo.flow`
    # 为止：`graspo.flow.guard` 必须走正常导入以执行其 `__init__.py`（它只依赖
    # `graspo.core.gpu_guard`，不碰 torch）。
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
            "Fail-closed pre-run availability assertion (F-10, three criteria AND-ed): "
            "memory.used <= --max-used-mib (default 2 GiB) AND free >= position "
            "(training tier 48 GiB, or --position-ratio 0.85 x total for serving tiers) "
            "AND no compute process on the card AND the card is not declared by any "
            "container, across --frames consecutive samples."
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
    position = parser.add_mutually_exclusive_group()
    position.add_argument(
        "--position-mib",
        type=float,
        default=None,
        help=(
            "Tier memory position in MiB (default: GRASPO_GPU_POSITION_MIB or 48 GiB). "
            "Training tiers use this; serving tiers should use --position-ratio instead."
        ),
    )
    position.add_argument(
        "--position-ratio",
        type=float,
        default=None,
        help=(
            "Serving-tier position as a fraction of the card's total memory "
            "(e.g. 0.85 => 69 632 MiB on an 81 920 MiB card, which is what vLLM needs). "
            "Mutually exclusive with --position-mib."
        ),
    )
    parser.add_argument(
        "--max-used-mib",
        type=float,
        default=None,
        help=(
            "Allowed memory.used ceiling per card in MiB (default 2048 = 2 GiB). "
            "Raise it only when you deliberately accept sharing a card with someone."
        ),
    )
    parser.add_argument(
        "--frames",
        "--samples",
        dest="frames",
        type=int,
        default=None,
        help=(
            "Consecutive samples that must ALL pass (default 3; "
            "--samples kept as a compatibility alias)."
        ),
    )
    parser.add_argument(
        "--interval",
        "--interval-s",
        dest="interval",
        type=float,
        default=None,
        help=(
            "Seconds between consecutive samples (default 10; "
            "--interval-s kept as a compatibility alias)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print a machine-readable per-card verdict "
            "(available/card/reasons[]/processes[]/containers[]) to stdout."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Print nothing on success.")
    return parser


def _human_lines(verdicts: list[Any]) -> list[str]:
    """逐卡人读结论（卡号 + 可用/不可用 + 原因）。"""
    lines: list[str] = []
    for verdict in verdicts:
        if verdict.available:
            lines.append(
                f"[gpu-idle] GPU{verdict.gpu_index} 可用（三判据全满足，{verdict.frames} 帧）"
            )
        else:
            lines.append(f"[gpu-idle] GPU{verdict.gpu_index} 不可用：{'；'.join(verdict.reasons)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    import json as _json

    args = build_parser().parse_args(argv)
    from graspo.core.gpu_guard import GpuLockError

    if args.max_used_mib is not None and args.max_used_mib < 0:
        print("--max-used-mib 不得为负（0 表示「卡上不许有任何占用」）", file=sys.stderr)
        return 2
    if args.frames is not None and args.frames < 1:
        print(
            "--frames 必须 ≥ 1（起跑前判定建议 ≥ 2：单帧会把'邻居恰好不在算'误判为空闲）",
            file=sys.stderr,
        )
        return 2

    try:
        verdicts = _monitor.evaluate_visible_gpus_idle(
            visible=args.visible,
            position_mib=args.position_mib,
            position_ratio=args.position_ratio,
            max_used_mib=args.max_used_mib,
            samples=args.frames,
            interval_s=args.interval,
        )
    except GpuLockError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # 读数拿不到就 fail-closed——不能把"测不到"当成"可用"。
        print(f"目标卡可用断言失败（fail-closed）：{exc}", file=sys.stderr)
        return 1

    failed = [verdict for verdict in verdicts if not verdict.available]
    human = _human_lines(verdicts)
    if args.json:
        payload = {
            "schema": "gpu-idle-assert/v1",
            "visible": args.visible,
            "targets": [
                int(v.gpu_index) if str(v.gpu_index).isdigit() else str(v.gpu_index)
                for v in verdicts
            ],
            "position_mib": None if args.position_ratio is not None else verdicts[0].position_mib,
            "position_ratio": args.position_ratio,
            "max_used_mib": verdicts[0].max_used_mib,
            "frames": args.frames,
            "interval_s": args.interval,
            "ok": not failed,
            "available": [v.as_dict() for v in verdicts if v.available],
            "unavailable": [v.as_dict() for v in verdicts if not v.available],
            "cards": [v.as_dict() for v in verdicts],
        }
        print(_json.dumps(payload, ensure_ascii=False, indent=2))
        for line in human:
            print(line, file=sys.stderr)
    elif not args.quiet:
        # 可用 ⇒ stdout；不可用 ⇒ stderr（调用方按 stderr 读"为什么拒"，与旧版一致）。
        for verdict, line in zip(verdicts, human, strict=True):
            print(line, file=sys.stdout if verdict.available else sys.stderr)
        if not failed:
            print(
                f"[gpu-idle] OK: {'/'.join(str(v.gpu_index) for v in verdicts)} 三判据全满足"
                "（已用显存 ≤ 上限 ∧ 剩余 ≥ 头寸 ∧ 无 compute 进程 ∧ 未被容器声明；帧帧满足）"
            )
    if failed:
        if not args.json:
            print(f"[gpu-idle] 拒绝：{len(failed)} 张卡不可用", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
