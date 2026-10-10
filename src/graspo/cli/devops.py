"""稳定 CLI 入口：把 ``scripts/`` 下被制度化的宿主/运维工具登记为 ``graspo`` 子命令（§10.2）。

宪法 §10.2 判据：脚本一旦「被测试固化 / 被 README 或 docs 引用 / 被工作流反复
使用」，即成为**制度化接口**，必须升级为 CLI 子命令或删除。本模块把这批工具
登记为对外**稳定子命令**（参数与输出契约随本项目同步，不随开发者临时改动）：

=========================  ==============================  =============================
子命令                     实现（唯一真相源）               职责
=========================  ==============================  =============================
``gpu-lock-guard``         ``scripts/gpu_lock_guard.py``    宿主/容器锁卡守卫（fail-closed）
``gpu-idle-assert``        ``scripts/gpu_idle_assert.py``   目标卡「实测空闲」断言（fail-closed）
``check-env-versions``     ``scripts/check_env_versions.py`` 声明版本 vs 镜像版本门禁
``collect-results``        ``scripts/collect_results.py``   矩阵链 A1–A6 台账收集器
=========================  ==============================  =============================

**为什么保留 ``scripts/<name>.py`` 原址、不做包内搬移**（§1.4 单一真相源）：这四个
脚本同时是**宿主引导层**入口——``run.sh`` 的锁卡守卫与 ``docker/Dockerfile*`` 的
构建期 version gate 在「尚未建 venv / 尚未安装 graspo」的时机调用它们，只能走
``python3 scripts/<name>.py``（见 ``run.sh:114``、``tests/conftest.py:53``、
``docker/Dockerfile:92``）。仓内 ``scripts/`` 路径因此是**稳定契约，不得迁址**。
本模块只在这份**同一实现**之上再登记一份 ``graspo`` 侧入口：

- 不复制参数表（参数**原样透传**给脚本自己的 argparse，``--help`` 亦透传）⇒
  不存在「两套参数」的第二真相源；
- 不改变任何既有调用点（``run.sh`` / ``tests/conftest.py`` / docker / pyproject）
  ⇒ 无静默失效面。

分工（§10.2「根目录启动脚本不属于 ``scripts/``」的同构结论）：宿主/镜像引导层走
``scripts/<name>.py``（已安装前可用）；交互式运维与已安装环境走
``graspo <name>``（稳定 CLI）。`eval_serve_vllm.sh` 属**运维/Docker 层**入口，
不登记为 python 子命令，见 ``docs/devops-cli.md``。
"""

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

#: 稳定子命令名 → 被登记的脚本文件名（唯一映射点，§1.4；顺序 = 文档顺序）。
DEVOPS_ENTRIES: dict[str, str] = {
    "gpu-lock-guard": "gpu_lock_guard.py",
    "gpu-idle-assert": "gpu_idle_assert.py",
    "check-env-versions": "check_env_versions.py",
    "collect-results": "collect_results.py",
}


def repo_root() -> Path:
    """仓根：本文件 = ``<root>/src/graspo/cli/devops.py`` ⇒ ``parents[3]``。"""
    return Path(__file__).resolve().parents[3]


def scripts_dir() -> Path:
    """被登记脚本所在目录（与 ``eval_commands.py`` 同一仓根推导口径）。"""
    return repo_root() / "scripts"


def run_script_entry(script_name: str, argv: Sequence[str]) -> int:
    """以 ``__main__`` 身份执行 ``scripts/<script_name>``，参数原样透传。

    Args:
        script_name: ``scripts/`` 下的文件名（来自 :data:`DEVOPS_ENTRIES`）。
        argv: 透传给被登记脚本的参数（含 ``--help``）；不含脚本名。

    Returns:
        被登记脚本的退出码。脚本抛 ``SystemExit`` 时归一为 ``int``
        （``None`` ⇒ 0；非 int 的 payload ⇒ 1），与直接执行脚本同语义。
        脚本不存在 ⇒ 2（fail-closed，绝不静默成功）。
    """
    script = scripts_dir() / script_name
    if not script.is_file():
        print(f"找不到被登记的脚本：{script}", file=sys.stderr)
        return 2
    saved_argv = sys.argv
    # 脚本用自己的 ``sys.argv``（``check_env_versions`` / ``collect_results`` 直接读它）
    # ⇒ 必须把 argv[0] 设成脚本路径，argv[1:] 为透传参数。
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:  # 脚本的 `raise SystemExit(main())`
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    finally:
        sys.argv = saved_argv
    return 0


def build_devops_parser(subparsers: argparse._SubParsersAction) -> None:
    """登记稳定子命令（供 ``graspo --help`` 列出；参数透传由 ``main`` 分派）。

    这里**不**声明任何参数：参数契约的唯一声明处是被登记脚本的 argparse。
    ``add_help=False`` 是刻意的——``graspo gpu-lock-guard --help`` 必须落到
    脚本自己的 ``--help``（真实参数表），而不是本壳的占位帮助。
    """
    for name, script_name in DEVOPS_ENTRIES.items():
        parser = subparsers.add_parser(
            name,
            help=f"Stable CLI entry for scripts/{script_name} (§10.2).",
            add_help=False,
        )
        parser.set_defaults(func=lambda args, _script=script_name: run_script_entry(_script, []))


def dispatch_argv(argv: Sequence[str]) -> int | None:
    """若 ``argv`` 指向被登记的稳定子命令则执行并把参数透传，否则返回 ``None``。

    由 ``cli.app.main`` 在 argparse **之前**调用：被登记脚本的参数表里既有
    ``--manifest`` 这类选项，argparse 的 ``REMAINDER`` 无法把「首个 token 即选项」
    的序列原样收下（会把未知选项判成用法错误）⇒ 透传必须在这里完成。
    """
    if argv and argv[0] in DEVOPS_ENTRIES:
        return run_script_entry(DEVOPS_ENTRIES[argv[0]], list(argv[1:]))
    return None
