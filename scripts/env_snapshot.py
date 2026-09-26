#!/usr/bin/env python3
"""跑批环境快照：跑前 / 跑后各采一次，落盘可比对 JSON，并支持两次快照相减。

采集维度（HTML §4 执行解读第 6 条：环境不保证 100% 冻结，跑批前后各采一次）：

* ``git``：分支 / ``rev-parse HEAD`` / 工作区是否干净（``status --porcelain``）
* ``config``：配置目录（默认 ``samples/configs/matrix54-v2/``）**总指纹**——
  逐文件 sha256 排序后聚合，任何一格配置改动都会改变指纹
* ``image``：训练镜像 ID / RepoDigests / 创建时间 / 大小（``docker image inspect``）
* ``gpu``：``nvidia-smi`` 摘要（型号 / UUID / 总显存 / 已用 / 利用率 / 驱动版本）
* ``disk``：仓库根与产根的余量（``shutil.disk_usage``）
* ``containers``：``docker ps`` 摘要（尽力而为，权限不足则如实记错误）

冻结维度（变则本轮档不可比）：``git.head`` / ``config.fingerprint_sha256`` /
``image.id`` / ``image.repo_digests``。其余维度（显存占用等）只记录、不判不可比。

用法::

    # 跑批前
    .venv/bin/python scripts/env_snapshot.py --run-root /path/to/r1 --phase before
    # 跑批后
    .venv/bin/python scripts/env_snapshot.py --run-root /path/to/r1 --phase after
    # 相减（列出变化项；冻结维度变化即打印"哪些档不可比"）
    .venv/bin/python scripts/env_snapshot.py --diff /path/to/r1/env-snapshot-before.json \\
        /path/to/r1/env-snapshot-after.json

退出码：0 = 采集成功 / 两次快照冻结维度一致；3 = 冻结维度发生变化；2 = 参数或 IO 错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "samples" / "configs" / "matrix54-v2"

#: 冻结维度（变 ⇒ 本轮档不可比）。
FROZEN_KEYS = [
    "git.head",
    "config.fingerprint_sha256",
    "image.id",
    "image.repo_digests",
]
#: 快照自身的时间戳不是环境量，diff 时忽略。
DIFF_IGNORE = {"snapshot_time", "phase", "run_root"}

_CMD_TIMEOUT = 20


def run_cmd(cmd: list[str], timeout: int = _CMD_TIMEOUT) -> dict[str, Any]:
    """执行命令，返回 ``{ok, returncode, stdout, stderr, error}``；失败不抛异常。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "", "error": "命令不存在"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "",
                "error": f"超时（>{timeout}s）"}
    except OSError as exc:  # pragma: no cover - 环境相关
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "",
                "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "error": None,
    }


# ---------------------------------------------------------------------------
# 各维度采集
# ---------------------------------------------------------------------------
def collect_git(repo_root: Path) -> dict[str, Any]:
    head = run_cmd(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    branch = run_cmd(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_cmd(["git", "-C", str(repo_root), "status", "--porcelain"])
    return {
        "head": head["stdout"] or None,
        "branch": branch["stdout"] or None,
        "dirty": bool(status["stdout"]),
        "porcelain": status["stdout"].splitlines() if status["stdout"] else [],
        "error": head["error"] or status["error"],
    }


def collect_config(config_dir: Path) -> dict[str, Any]:
    if not config_dir.is_dir():
        return {"dir": str(config_dir), "exists": False, "file_count": 0,
                "fingerprint_sha256": None, "files": {}}
    files: dict[str, str] = {}
    for path in sorted(config_dir.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[str(path.relative_to(config_dir))] = digest
    aggregate = hashlib.sha256()
    for name, digest in sorted(files.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\x00")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "dir": str(config_dir),
        "exists": True,
        "file_count": len(files),
        "fingerprint_sha256": aggregate.hexdigest(),
        "files": files,
    }


def collect_image(image: str | None) -> dict[str, Any]:
    if not image:
        return {"requested": None, "available": False, "error": "未指定镜像（--image / GRASPO_IMAGE）"}
    result = run_cmd([
        "docker", "image", "inspect", image,
        "--format", "{{.Id}}|{{json .RepoDigests}}|{{.Created}}|{{.Size}}",
    ])
    if not result["ok"]:
        return {"requested": image, "available": False, "id": None,
                "repo_digests": None, "created": None, "size_bytes": None,
                "error": result["error"] or result["stderr"] or f"docker inspect 退出 {result['returncode']}"}
    raw = result["stdout"].splitlines()[0] if result["stdout"] else ""
    parts = raw.split("|")
    digests: list[str] = []
    if len(parts) > 1:
        try:
            digests = json.loads(parts[1]) or []
        except json.JSONDecodeError:
            digests = [parts[1]]
    return {
        "requested": image,
        "available": True,
        "id": parts[0] if parts else None,
        "repo_digests": digests,
        "created": parts[2] if len(parts) > 2 else None,
        "size_bytes": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
        "error": None,
    }


def collect_gpu() -> dict[str, Any]:
    query = "index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu"
    result = run_cmd([
        "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
    ])
    if not result["ok"]:
        return {"available": False, "gpus": [],
                "error": result["error"] or result["stderr"] or "nvidia-smi 不可用"}
    gpus = []
    for line in result["stdout"].splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 7:
            continue
        try:
            gpus.append({
                "index": int(cells[0]),
                "name": cells[1],
                "uuid": cells[2],
                "driver_version": cells[3],
                "memory_total_mib": int(cells[4]),
                "memory_used_mib": int(cells[5]),
                "utilization_gpu_pct": int(cells[6]),
            })
        except ValueError:
            continue
    compute_apps = run_cmd([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    return {
        "available": True,
        "gpus": gpus,
        "compute_apps": compute_apps["stdout"].splitlines() if compute_apps["ok"] else [],
        "error": None,
    }


def collect_disk(paths: dict[str, Path]) -> dict[str, Any]:
    disks: dict[str, Any] = {}
    for name, path in paths.items():
        try:
            usage = shutil.disk_usage(path)
            disks[name] = {
                "path": str(path),
                "total_gib": round(usage.total / 1024 ** 3, 2),
                "used_gib": round(usage.used / 1024 ** 3, 2),
                "free_gib": round(usage.free / 1024 ** 3, 2),
            }
        except OSError as exc:
            disks[name] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return disks


def collect_containers() -> dict[str, Any]:
    result = run_cmd([
        "docker", "ps", "--format", "{{.ID}}|{{.Image}}|{{.Names}}|{{.Status}}",
    ])
    if not result["ok"]:
        return {"available": False, "containers": [],
                "error": result["error"] or result["stderr"] or f"docker ps 退出 {result['returncode']}"}
    return {"available": True, "containers": result["stdout"].splitlines(), "error": None}


def collect_snapshot(repo_root: Path, run_root: Path, config_dir: Path,
                     image: str | None, phase: str) -> dict[str, Any]:
    return {
        "schema": "graspo.env_snapshot.v1",
        "phase": phase,
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "git": collect_git(repo_root),
        "config": collect_config(config_dir),
        "image": collect_image(image),
        "gpu": collect_gpu(),
        "disk": collect_disk({"repo_root": repo_root, "run_root": run_root}),
        "containers": collect_containers(),
    }


# ---------------------------------------------------------------------------
# 摘要与 diff
# ---------------------------------------------------------------------------
def print_summary(snapshot: dict[str, Any]) -> None:
    git = snapshot.get("git", {})
    cfg = snapshot.get("config", {})
    image = snapshot.get("image", {})
    gpu = snapshot.get("gpu", {})
    print(f"[env-snapshot] phase={snapshot.get('phase')} time={snapshot.get('snapshot_time')}")
    print(f"  git: head={git.get('head')} branch={git.get('branch')} dirty={git.get('dirty')}")
    print(f"  config: {cfg.get('file_count')} files sha256={cfg.get('fingerprint_sha256')}")
    if image.get("available"):
        print(f"  image: id={image.get('id')} digests={image.get('repo_digests')}")
    else:
        print(f"  image: 不可得（{image.get('error')}）")
    if gpu.get("available"):
        total = sum(item["memory_total_mib"] for item in gpu["gpus"])
        used = sum(item["memory_used_mib"] for item in gpu["gpus"])
        print(f"  gpu: {len(gpu['gpus'])} 卡 已用/总量 = {used}/{total} MiB；"
              f"compute_apps={len(gpu.get('compute_apps') or [])}")
    else:
        print(f"  gpu: 不可得（{gpu.get('error')}）")
    for name, disk in (snapshot.get("disk") or {}).items():
        if "free_gib" in disk:
            print(f"  disk[{name}]: free={disk['free_gib']} GiB / total={disk['total_gib']} GiB")


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[str], list[str]]:
    """返回 ``(冻结维度变化, 非冻结维度变化)``（每个元素是人读一行）。"""
    flat_before: dict[str, Any] = {}
    flat_after: dict[str, Any] = {}
    _flatten("", before, flat_before)
    _flatten("", after, flat_after)
    keys = sorted(set(flat_before) | set(flat_after))
    frozen: list[str] = []
    other: list[str] = []
    for key in keys:
        if key in DIFF_IGNORE or key.startswith(("snapshot_time", "phase", "run_root")):
            continue
        old = flat_before.get(key, "<缺失>")
        new = flat_after.get(key, "<缺失>")
        if old == new:
            continue
        line = f"  {key}: {old!r} -> {new!r}"
        if any(key == frozen_key or key.startswith(frozen_key + ".") for frozen_key in FROZEN_KEYS):
            frozen.append(line)
        else:
            other.append(line)
    return frozen, other


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="产根；快照写到 <run-root>/env-snapshot-<phase>.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="显式输出路径（默认 <run-root>/env-snapshot-<phase>.json）")
    parser.add_argument("--phase", choices=["before", "after"], default=None,
                        help="跑前 / 跑后；决定默认文件名")
    parser.add_argument("--image", default=os.environ.get("GRASPO_IMAGE"),
                        help="训练镜像 tag（默认取 $GRASPO_IMAGE）")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help=f"配置目录（默认 {DEFAULT_CONFIG_DIR}）")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
                        help="不采集，只对两份快照相减并打印变化项")
    args = parser.parse_args(argv)

    if args.diff:
        try:
            before = json.loads(args.diff[0].read_text(encoding="utf-8"))
            after = json.loads(args.diff[1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FATAL: 读快照失败：{exc}", file=sys.stderr)
            return 2
        print(f"[env-snapshot --diff] {args.diff[0]} -> {args.diff[1]}")
        frozen, other = diff_snapshots(before, after)
        print(f"  冻结维度变化 {len(frozen)} 项：")
        print("\n".join(frozen) if frozen else "    （无）")
        print(f"  非冻结维度变化 {len(other)} 项：")
        print("\n".join(other) if other else "    （无）")
        if frozen:
            print("  结论：环境发生了冻结维度变化 ⇒ **本轮全部档标记为不可比**，"
                  "如实报告，不得悄悄重跑（HTML §4 执行解读第 6 条）。")
            return 3
        print("  结论：冻结维度一致 ⇒ 各档之间可比。")
        return 0

    if args.phase is None:
        print("FATAL: 采集模式必须给 --phase before|after（或使用 --diff）", file=sys.stderr)
        return 2
    run_root = args.run_root or args.repo_root
    out = args.out or (run_root / f"env-snapshot-{args.phase}.json")
    snapshot = collect_snapshot(args.repo_root, run_root, args.config_dir, args.image, args.phase)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"FATAL: 写快照失败 {out}：{exc}", file=sys.stderr)
        return 2
    print_summary(snapshot)
    print(f"  已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
