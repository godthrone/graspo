#!/usr/bin/env python3
"""环境版本门禁：**声明版本 == 镜像内版本**（宪法 §6 环境可复现）。

职责：把"依赖声明"与"执行载体里真正装了什么"钉在一起，防止版本只靠传递
依赖漂移。判定来源：

1. `pyproject.toml` 的**全部 `==` 精确 pin**（运行时依赖 + extras + dev 组，
   经**显式排除清单**过滤）——**包集合自动派生，不手工列举**（手工列表必然漏；
   本门禁第一版就漏了 pydantic，这正是宪法 §2 防呆要避免的）。
2. `docker/Dockerfile.msswift` / `docker/Dockerfile` 的 `pip install name==version`
   pin（自动解析），与 pyproject 逐包比对——两个声明点不得分叉。
3. 镜像内实测值（`importlib.metadata.version`，与 `pip show` 同源）。

比较规则：**release 段必须相等**；`+local` 标签（如 torch 的 `+cu130`）编码
CUDA 构建渠道，属于镜像属性，比较时归一化剔除（PEP 440 本地版本语义）。
CUDA 渠道本身的核对由构建配方里的 `torch.version.cuda` 断言负责。

用法：
    python3 scripts/check_env_versions.py --observed-json '{...}'   # 离线/负向测试
    python3 scripts/check_env_versions.py --local                   # 当前解释器（镜像构建期）
    python3 scripts/check_env_versions.py --from-container graspo-msswift:4.5.3

退出码：0 = 一致；1 = 不一致（fail-closed）；2 = 用法/IO 错误。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DOCKERFILES = (
    PROJECT_ROOT / "docker" / "Dockerfile.msswift",
    PROJECT_ROOT / "docker" / "Dockerfile",
)

#: **显式排除清单**：包名 -> 理由。排除必须在此写明理由——
#: 不能靠"没写进列表"来隐式排除（否则必然漏包）。
EXCLUDED_FROM_GATE: dict[str, str] = {
    "pytest": "dev extra：开发工具，不属于镜像运行时依赖",
    "ruff": "dev extra：开发工具，不属于镜像运行时依赖",
    "mypy": "dev extra：开发工具，不属于镜像运行时依赖",
    "openpyxl": "dev extra：本机表格工具，不属于镜像运行时依赖",
}

#: 容器内取版本的探针脚本（多行，避免 -c 中的引号地狱）。
_CONTAINER_PROBE = """
import importlib.metadata as m, json
out = {}
for p in %r:
    try:
        out[p] = m.version(p)
    except m.PackageNotFoundError:
        continue    # 该包未安装 ⇒ 显式跳过（不改变 out 的既有口径）
print(json.dumps(out))
"""


def release_segment(version: str) -> str:
    """取 PEP 440 release 段：`2.11.0+cu130` -> `2.11.0`。"""
    return version.split("+", 1)[0].strip()


def _normalize_name(name: str) -> str:
    return name.split("[", 1)[0].strip().lower().replace("_", "-")


def _pin_from_requirement(item: object) -> tuple[str, str] | None:
    """从 `name==x.y.z` / `name[extra]==x.y.z` 形式的依赖项解析 (name, version)。

    版本段**排除 shell 引号**（`'` / `"`）：Dockerfile 里 pin 常写在 `sh -c '…'` 内，
    闭合引号可能落在版本号前后（见 `parse_dockerfile_pins()`）；合法 PEP 440 版本
    不含引号，排除它是严格化而非放松。
    """
    text = str(item).strip()
    match = re.match(r"^([A-Za-z0-9_.\-]+(?:\[[^\]]*\])?)\s*==\s*([^\s;#\"']+)", text)
    if match is None:
        return None
    return _normalize_name(match.group(1)), match.group(2).strip()


def declared_pins(pyproject: Path = PYPROJECT) -> dict[str, str]:
    """自动派生 pyproject 中全部 `==` 精确 pin（含 extras 与 dependency-groups）。"""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    raw: list[object] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        raw += list(extra)
    for group in data.get("dependency-groups", {}).get("dev", []):
        raw.append(group)

    pins: dict[str, str] = {}
    for item in raw:
        parsed = _pin_from_requirement(item)
        if parsed is not None:
            pins[parsed[0]] = parsed[1]
    return pins


def gated_packages(pyproject: Path = PYPROJECT) -> dict[str, str]:
    """门禁包集合 = 全部精确 pin − 显式排除清单（自动派生，§2 防呆）。"""
    pins = declared_pins(pyproject)
    return {name: version for name, version in pins.items() if name not in EXCLUDED_FROM_GATE}


def parse_dockerfile_pins(dockerfile: Path) -> dict[str, str]:
    """自动解析 Dockerfile 中 `pip install ... name==version` 的全部 pin。

    ★ 自适应 shell 引号：pin 常写在 `sh -eu -c '…'` / `sh -c "…"` 内，按空白切出的
    token 可能被引号包裹或尾随引号（`torchvision==0.26.0'`、`"ms-swift==4.5.3"`、
    `pillow==11.3.0;'`）——**一律剥掉引号**再解析，避免合法 shell 引号把门禁打破
    （否则误报"声明不一致"，且 Dockerfile 内的 `RUN check_env_versions.py --local` 会让
    镜像构建直接失败）。负例回归见 `tests/e2e/test_check_env_versions.py`。
    """
    if not dockerfile.is_file():
        return {}
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    pins: dict[str, str] = {}
    for line in text.splitlines():
        if "pip install" not in line and "uv pip install" not in line:
            continue
        for token in line.split():
            parsed = _pin_from_requirement(token.strip("'\""))
            if parsed is not None:
                pins[parsed[0]] = parsed[1]
    return pins


def compare_versions(
    declared: dict[str, str],
    observed: dict[str, str],
    *,
    packages: tuple[str, ...] | None = None,
) -> list[str]:
    """比较声明与实测；返回人类可读的不一致列表（空列表 = 一致）。"""
    names = packages if packages is not None else tuple(sorted(declared))
    problems: list[str] = []
    for name in names:
        want = declared.get(name)
        got = observed.get(name)
        if want is None:
            problems.append(f"{name}: 声明缺失（pyproject 未 pin）")
            continue
        if got is None:
            problems.append(f"{name}: 镜像内缺失（实测未返回该包）")
            continue
        if release_segment(got) != release_segment(want):
            problems.append(f"{name}: 声明 {want} != 镜像实测 {got}")
    return problems


def compare_declaration_sites(declared: dict[str, str]) -> list[str]:
    """Dockerfile 的 pip pin 是否与 pyproject 声明一致（两个声明点不得分叉）。"""
    problems: list[str] = []
    for dockerfile in DOCKERFILES:
        pins = parse_dockerfile_pins(dockerfile)
        for name, version in pins.items():
            declared_version = declared.get(name)
            if declared_version is None:
                problems.append(
                    f"{dockerfile.name}: pip pin {name}=={version} 在 pyproject 中无对应声明"
                )
            elif release_segment(version) != release_segment(declared_version):
                problems.append(
                    f"{dockerfile.name}: {name}=={version} != pyproject 声明 {declared_version}"
                )
    return problems


def local_versions(packages: tuple[str, ...]) -> dict[str, str]:
    """当前解释器内实测版本（镜像构建期用 `.venv/bin/python --local`）。"""
    observed: dict[str, str] = {}
    for name in packages:
        try:
            observed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return observed


def query_container_versions(image: str, *, python_exec: str = "python") -> dict[str, str]:
    """进容器实测版本。

    锁卡采用**单一路径**：只设置 `NVIDIA_VISIBLE_DEVICES`（与容器内
    `gpu_lock_guard` 的断言同源），不再叠加 `--gpus`（两者可能互相覆盖，
    语义有歧义）。宿主侧前置断言由 `scripts/gpu_lock_guard.py` 负责。
    """
    packages = tuple(gated_packages())
    command = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=0",
        image,
        python_exec,
        "-c",
        _CONTAINER_PROBE % (list(packages),),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"container version query failed (rc={completed.returncode}): "
            f"{completed.stderr.strip()[:500]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def run(args: argparse.Namespace) -> int:
    declared = declared_pins()
    gated = gated_packages()

    if args.observed_json is not None:
        observed: dict[str, str] = json.loads(args.observed_json)
    elif args.local:
        observed = local_versions(tuple(gated))
    elif args.from_container is not None:
        observed = query_container_versions(args.from_container, python_exec=args.python)
    else:
        print("FATAL: need --observed-json, --local, or --from-container", file=sys.stderr)
        return 2

    problems = compare_declaration_sites(declared)
    problems += compare_versions(gated, observed, packages=tuple(sorted(gated)))

    print(f"gated packages ({len(gated)}, auto-derived): {sorted(gated)}")
    print(f"excluded (explicit): {sorted(EXCLUDED_FROM_GATE)}")
    print(f"declared: { {p: gated.get(p) for p in sorted(gated)} }")
    print(f"observed: { {p: observed.get(p) for p in sorted(gated)} }")

    if problems:
        print("VERSION GATE FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("VERSION GATE OK: declared == image (release 段一致)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Declared vs image dependency version gate.")
    parser.add_argument("--observed-json", default=None, help="Observed versions as JSON.")
    parser.add_argument("--local", action="store_true", help="Observe current interpreter.")
    parser.add_argument("--from-container", default=None, help="Image name to query versions from.")
    parser.add_argument("--python", default="python", help="Python executable inside the image.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
