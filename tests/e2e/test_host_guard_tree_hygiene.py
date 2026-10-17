"""宿主侧 python 调用不得把 ``__pycache__`` 写进装配树（能失败的对照）。

**缺陷（已具名登记、本次修复）**：宿主侧跑 ``tests/e2e/run_matrix54.sh``（以及
``tests/e2e/launch_matrix54.sh``）时，它用宿主解释器 ``$PYBIN="${PYTHON:-python3}"``
起 ``scripts/gpu_lock_guard.py`` / ``scripts/gpu_idle_assert.py``，调用点**没带**
``PYTHONDONTWRITEBYTECODE``。守卫按 ``Path(__file__).resolve().parents[1] / "src"``
导入 ``graspo.flow.guard`` / ``graspo.cli.gpu_monitor`` ⇒ CPython 在**树内**
``src/graspo/**/__pycache__/`` 留下 ``.pyc``。

**为什么必须修**：228 上跑能力矩阵时，代码不是镜像，而是把「装配树」只读挂载进容器；
装配树的纪律是「与基准 commit **逐字节等价**」，靠 ``git ls-tree -r <commit>`` 逐文件
比对（873 文件）来证明。树内多出任何 ``__pycache__`` / ``*.pyc`` 都会让该审计失效。

**本测试是能失败的对照**：

1. :func:`test_every_host_python_call_point_disables_bytecode` —— ``tests/e2e/*.sh``
   里**每一个**宿主侧 python 调用点都必须带 ``PYTHONDONTWRITEBYTECODE=1``。清单是显式
   登记的（``_HOST_CALL_SPECS``），并配一条保守的**完整性扫描**：脚本里任何没登记形态
   的宿主侧 python 行也会被要求带该变量 —— 防「只改 369/378 两行」式的漏改。
2. :func:`test_host_guard_leaves_no_bytecode_in_tree` —— 造一棵**干净**临时装配树，
   **按 shell 调用点里写明的形态**（卫生变量从脚本行里抽出来，测试不自备）跑宿主侧
   守卫，断言树内 ``__pycache__`` + ``*.pyc`` 计数 **== 0**。把 shell 里的
   ``PYTHONDONTWRITEBYTECODE=1`` 撤掉，本测试即红（实测：守卫在树内写下 11 个路径）。

红线：假 ``nvidia-smi`` + 临时副本；无真 GPU、无 docker、无真实容器、
⛔ 不碰任何既有装配树（只在 ``tmp_path`` 里造一次性副本）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_E2E = _ROOT / "tests" / "e2e"

#: 宿主侧 python 调用点必须带的环境变量 —— 只影响字节码落盘，不改任何
#: 命令行 / 退出码 / fail-closed 语义。
_HYGIENE_TOKEN = "PYTHONDONTWRITEBYTECODE=1"
_HYGIENE_ASSIGN_RE = re.compile(r"PYTHONDONTWRITEBYTECODE=([01])")

#: ``tests/e2e/*.sh`` 里宿主侧 python 的**解释器 token**（容器内一律走 ``$CONTAINER_PY``）。
_HOST_PYTHON_TOKENS = ('"$PYBIN"', '"$PYBIN_OR_PYTHON"', "python3")

#: 宿主侧 python 调用点清单（文件名 → 必须逐字出现的调用形态）。
#: 新增宿主侧 python 调用点时必须登记到这里；下面的完整性扫描也会挡住漏登记。
_HOST_CALL_SPECS: dict[str, tuple[str, ...]] = {
    "run_matrix54.sh": (
        # 档解析（stdin ``-``）：本次一并卫生化（-- 它现在只 import stdlib，写不出树内缓存，
        # 但统一调用形态后，"宿主侧 python 一律带卫生变量"这条不变量可由 grep 一次证明）。
        'TIER_JSON=$(PYTHONDONTWRITEBYTECODE=1 "$PYBIN" - "$MANIFEST" "$TIER"',
        '< <(PYTHONDONTWRITEBYTECODE=1 "$PYBIN" - "$TIER_JSON"',
        # 宿主侧守卫（缺陷本体所在调用点）
        'PYTHONDONTWRITEBYTECODE=1 "$PYBIN" "$ROOT_DIR/scripts/gpu_lock_guard.py"',
        'PYTHONDONTWRITEBYTECODE=1 "$PYBIN" "$ROOT_DIR/scripts/gpu_idle_assert.py"',
    ),
    "launch_matrix54.sh": (
        'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PYTHONPATH" python3 "$TREE/scripts/gpu_lock_guard.py"',
        'PYTHONDONTWRITEBYTECODE=1 "$PYBIN_OR_PYTHON" "$TREE/scripts/gpu_idle_assert.py"',
        'PYTHONDONTWRITEBYTECODE=1 python3 "$STAGE/sched_window_guard.py"',
    ),
    "run_matrix54_source.sh": (
        'PYTHONDONTWRITEBYTECODE=1 "$PYBIN" - "$MANIFEST" "$TIER"',
    ),
    # 旧矩阵 runner：全脚本无宿主侧 python 调用（登记为"已清点、确认为空"）。
    "run_matrix.sh": (),
}

#: 假 ``nvidia-smi``：只回空闲读数（``index, memory.used, memory.total, utilization.gpu``）。
#: 行为由环境驱动，绝不触碰真 GPU。
_FAKE_SMI = """#!/bin/bash
args="$*"
sel="none"
case "$args" in *" -i "*) sel="${args##* -i }";; esac
IFS=',' read -ra ids <<< "$sel"
for id in "${ids[@]}"; do
  printf '%s\\n' "$id, 1, 81920, 0"
done
exit 0
"""


def _host_python_call_lines() -> list[tuple[str, int, str]]:
    """``tests/e2e/*.sh`` 里的宿主侧 python 调用行 ``(文件名, 行号, 去缩进行)``。

    保守扫描：宁可多报（多报只会让"必须带卫生变量"的范围更严），也不漏报。
    排除：注释、``$CONTAINER_PY``（容器内）、``command -v`` 解释器探测、提示文本里的
    ``$PYBIN``、以及纯赋值（``PYBIN=...`` —— 它不是调用）。
    """
    found: list[tuple[str, int, str]] = []
    for script in sorted(_E2E.glob("*.sh")):
        for lineno, raw in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "CONTAINER_PY" in line:
                continue
            if not any(token in line for token in _HOST_PYTHON_TOKENS):
                continue
            if "command -v" in line:
                continue
            if re.match(r"^(?:echo|printf|say|die)\b", line):
                continue
            # 纯赋值（如 ``PYBIN="${PYTHON:-python3}"``）不是调用；但
            # ``VAR=1 python …``（调用前的环境前缀）与 ``VAR=$(python …)``（命令替换）
            # 都是调用，不能误排除 —— 因此只排除"整行就是一个赋值"的形态。
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", line):
                continue
            found.append((script.name, lineno, line))
    return found


def _assertions_from_call_points() -> dict[str, str]:
    """把调用点行里写明的字节码卫生变量抽出来（测试**不自备**该变量）。

    于是本测试验证的是「脚本声明的调用形态」：撤掉 shell 里的
    ``PYTHONDONTWRITEBYTECODE=1`` ⇒ 环境里没有它 ⇒ 守卫真的会把 ``.pyc`` 写进树 ⇒ 红。
    """
    env: dict[str, str] = {}
    for _name, _lineno, line in _host_python_call_lines():
        match = _HYGIENE_ASSIGN_RE.search(line)
        if match:
            env["PYTHONDONTWRITEBYTECODE"] = match.group(1)
    return env


def _clean_assembled_tree(dst: Path) -> Path:
    """造一棵干净装配树副本：只拷守卫要用的 ``scripts/`` + ``src/``，排除既有字节码。

    ⛔ 不碰仓库里既有的装配树/工作树 —— 只写到调用方给的 ``tmp_path``。
    """
    dst.mkdir(parents=True, exist_ok=True)
    for top in ("scripts", "src"):
        shutil.copytree(
            _ROOT / top,
            dst / top,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return dst


def _bytecode_paths(tree: Path) -> list[str]:
    """树内 ``__pycache__`` 目录与 ``*.pyc`` 文件（相对路径，排序）。"""
    hits = [
        path
        for path in tree.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    return sorted(str(path.relative_to(tree)) for path in hits)


# ── 1) 静态清点：每个宿主侧 python 调用点都禁用字节码写入 ────────────────────────
def test_every_host_python_call_point_disables_bytecode():
    for name, needles in _HOST_CALL_SPECS.items():
        text = (_E2E / name).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, (
                f"{name}: 宿主侧调用点形态已变或漏改（缺 {_HYGIENE_TOKEN}）：{needle}"
            )

    # 完整性：脚本里任何宿主侧 python 行都必须带卫生变量（防漏登记 / 防只改两行）。
    missing = [
        f"{name}:{lineno}: {line}"
        for name, lineno, line in _host_python_call_lines()
        if _HYGIENE_ASSIGN_RE.search(line) is None
    ]
    assert missing == [], "以下宿主侧 python 调用点没有禁用字节码写入：\n" + "\n".join(missing)


def test_the_inventory_still_matches_the_scripts():
    """清单的显式调用形态必须能在脚本里逐条找到（防清单腐烂成空话）。"""
    registered = sum(len(needles) for needles in _HOST_CALL_SPECS.values())
    assert registered > 0
    for name, needles in _HOST_CALL_SPECS.items():
        assert (_E2E / name).is_file(), f"登记的脚本不存在：{name}"
        for needle in needles:
            assert needle in (_E2E / name).read_text(encoding="utf-8")


# ── 2) 行为对照：跑过宿主侧守卫后，树内字节码计数 == 0 ──────────────────────────
def test_host_guard_leaves_no_bytecode_in_tree(tmp_path: Path):
    tree = _clean_assembled_tree(tmp_path / "assembled-tree")
    assert _bytecode_paths(tree) == [], "临时副本起点必须干净（0 个字节码路径）"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_smi = bin_dir / "nvidia-smi"
    fake_smi.write_text(_FAKE_SMI, encoding="utf-8")
    fake_smi.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GRASPO_GPU_IDLE_SAMPLES": "2",
        "GRASPO_GPU_IDLE_INTERVAL_S": "0",
        # ★ 卫生变量**取自 shell 调用点**，不是测试自己塞的。
        **_assertions_from_call_points(),
    }
    env.pop("NVIDIA_VISIBLE_DEVICES", None)

    # 复刻 run_matrix54.sh:369 / :378 两条宿主侧守卫调用（同一 argv 形态）。
    for script in ("scripts/gpu_lock_guard.py", "scripts/gpu_idle_assert.py"):
        completed = subprocess.run(
            [sys.executable, str(tree / script), "--visible", "0"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{script} 退出码 {completed.returncode}（守卫语义被改动？）\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )

    leftovers = _bytecode_paths(tree)
    assert len(leftovers) == 0, (
        "跑过宿主侧守卫后，装配树内 __pycache__/*.pyc 数量必须 = 0，实得 "
        f"{len(leftovers)}：\n" + "\n".join(leftovers)
    )
