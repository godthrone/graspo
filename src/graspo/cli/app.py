from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graspo.core.schema import GraspoConfig


@dataclass(slots=True)
class LaunchPlan:
    command: list[str]
    env: dict[str, str]
    backend: str
    uses_torchrun: bool
    nproc_per_node: int
    nnodes: int


def cmd_export(args: argparse.Namespace) -> int:
    config = GraspoConfig.from_yaml(args.config)
    if args.base_model:
        config.model.model_path = args.base_model
    from graspo.backends.graspoflow.lora_io import export_from_checkpoint

    export_from_checkpoint(
        args.checkpoint,
        args.output,
        export_format=args.format,
        base_model_path=config.model.model_path,
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "format": args.format,
                "output": args.output,
                "base_model": config.model.model_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    plan = build_launch_plan(args.config)
    print(
        json.dumps(
            {
                "backend": plan.backend,
                "uses_torchrun": plan.uses_torchrun,
                "nnodes": plan.nnodes,
                "nproc_per_node": plan.nproc_per_node,
                "command": plan.command,
            },
            ensure_ascii=False,
        )
    )
    completed = subprocess.run(plan.command, env=plan.env, check=False)
    return int(completed.returncode)


def build_launch_plan(config_path: str | Path, config: GraspoConfig | None = None) -> LaunchPlan:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise SystemExit(f"Config file does not exist: {config_path}")
    config = config or GraspoConfig.from_yaml(config_path)

    from graspo.backends import select_backend

    selection = select_backend(config)
    launch = config.launch
    nnodes = int(launch.nnodes)
    if nnodes < 1:
        raise SystemExit("launch.nnodes must be >= 1")

    nproc_per_node = _resolve_nproc_per_node(config, selection.name)
    _validate_launch_paths(config)
    _validate_launch_world(config, selection.name, nnodes, nproc_per_node)

    env = _build_launch_env(config)
    python = str(launch.python or sys.executable)
    train_command = [python, "-m", "graspo.cli.train_worker", "--config", str(config_path)]

    uses_torchrun = nnodes * nproc_per_node > 1
    if uses_torchrun:
        command = _torchrun_prefix(config, python) + [
            f"--nnodes={nnodes}",
            f"--node_rank={int(launch.node_rank)}",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_addr={launch.master_addr}",
            f"--master_port={int(launch.master_port)}",
            "-m",
            "graspo.cli.train_worker",
            "--config",
            str(config_path),
        ]
    else:
        command = train_command

    return LaunchPlan(
        command=command,
        env=env,
        backend=selection.name,
        uses_torchrun=uses_torchrun,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
    )


def _resolve_nproc_per_node(config: GraspoConfig, backend: str) -> int:
    launch = config.launch
    if launch.nproc_per_node is not None:
        nproc_per_node = int(launch.nproc_per_node)
    else:
        expected_world = _graspoflow_world_size(config)
        nnodes = int(launch.nnodes)
        if expected_world % nnodes != 0:
            raise SystemExit(
                "graspoflow world size must divide evenly across launch.nnodes "
                f"({expected_world} % {nnodes} != 0)"
            )
        nproc_per_node = expected_world // nnodes
    if nproc_per_node < 1:
        raise SystemExit("launch.nproc_per_node must be >= 1")
    return nproc_per_node


def _validate_launch_world(
    config: GraspoConfig,
    backend: str,
    nnodes: int,
    nproc_per_node: int,
) -> None:
    actual_world = nnodes * nproc_per_node
    expected_world = _graspoflow_world_size(config)
    if actual_world != expected_world:
        raise SystemExit(
            "graspoflow launch world size must match "
            "tp_size × pp_size "
            f"({actual_world} != {expected_world})"
        )

    gpus = _format_gpus(config.launch.gpus)
    if gpus is not None and len(gpus.split(",")) != nproc_per_node:
        raise SystemExit(
            "launch.gpus count must match launch.nproc_per_node "
            f"({len(gpus.split(','))} != {nproc_per_node})"
        )


def _graspoflow_world_size(config: GraspoConfig) -> int:
    return int(config.graspoflow.tp_size) * int(config.graspoflow.pp_size)


def _validate_launch_paths(config: GraspoConfig) -> None:
    _require_config_value(config.model.model_path, "model.model_path")
    _require_config_value(config.data.train_path, "data.train_path")
    _require_config_value(config.training.output_dir, "training.output_dir")
    data_path = Path(config.data.train_path)
    if not data_path.is_file():
        raise SystemExit(f"data.train_path does not exist: {data_path}")
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)


def _require_config_value(value: Any, name: str) -> None:
    text = str(value or "").strip()
    if not text or "${" in text or text.startswith("<"):
        raise SystemExit(f"{name} must be set in the YAML config")


def _build_launch_env(config: GraspoConfig) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in config.launch.env.items():
        env[str(key)] = str(value)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    gpus = _format_gpus(config.launch.gpus)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus

    src_dir = _project_src_dir()
    if src_dir.is_dir():
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(src_dir) if not current else f"{src_dir}{os.pathsep}{current}"
    return env


def _format_gpus(value: list[int] | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = [str(int(item)) for item in value]
    if not parts:
        return None
    return ",".join(parts)


def _torchrun_prefix(config: GraspoConfig, python: str) -> list[str]:
    if config.launch.torchrun:
        return [str(config.launch.torchrun)]
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun]
    return [python, "-m", "torch.distributed.run"]


def _project_src_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "src"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graspo", description="GRASPO training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Launch training from a single YAML config.")
    launch.add_argument("--config", "-c", required=True)
    launch.set_defaults(func=cmd_launch)

    export = subparsers.add_parser(
        "export", help="Export a native GRASPO checkpoint to a portable model artifact."
    )
    export.add_argument("--config", "-c", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--format", choices=["peft-adapter", "merged-hf"], required=True)
    export.add_argument("--output", "-o", required=True)
    export.add_argument("--base-model", help="Override config.model.model_path for export.")
    export.set_defaults(func=cmd_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
