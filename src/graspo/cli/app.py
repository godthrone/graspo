"""CLI 命令入口：``graspo launch`` / ``graspo export`` / 工具命令。

配置驱动：解析 --config → 加载并校验配置 → 构建 torchrun 启动计划。
工具命令（validate-reward / evaluate-checkpoint / analyze-profile）实现在
``cli.tools``，按配置驱动命令约束：只 print 或输出由 config 决定。
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graspo.core.gpu_guard import require_gpu_lock_or_exit
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
    _require_config_value(config.export.checkpoint_path, "export.checkpoint_path")
    _require_config_value(config.export.export_output, "export.export_output")
    _require_config_value(config.model.model_path, "model.model_path")
    from graspo.flow.lora.lora_io import export_from_checkpoint

    export_from_checkpoint(
        config.export.checkpoint_path,
        config.export.export_output,
        export_format=config.export.export_format,
        base_model_path=config.model.model_path,
    )
    print(
        json.dumps(
            {
                "checkpoint": config.export.checkpoint_path,
                "format": config.export.export_format,
                "output": config.export.export_output,
                "base_model": config.model.model_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_validate_reward(args: argparse.Namespace) -> int:
    """校验 reward 评分链路：加载数据 → 评分 → 只 print，不落盘。"""
    from graspo.cli.tools import validate_reward_scores
    from graspo.flow.data_io import load_jsonl

    samples = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    completions: list[str] = []
    if args.completions:
        with Path(args.completions).open("r", encoding="utf-8") as handle:
            for line in handle:
                completions.append(json.loads(line)["completion"])
    scores = validate_reward_scores(samples, completions)
    for score in scores:
        print(json.dumps(score, ensure_ascii=False))
    if scores:
        print(
            json.dumps(
                {
                    "samples": len(scores),
                    "mean": round(sum(score["reward"] for score in scores) / len(scores), 6),
                    "all_right": sum(1 for score in scores if score["all_right"]),
                },
                ensure_ascii=False,
            )
        )
    return 0


def cmd_evaluate_checkpoint(args: argparse.Namespace) -> int:
    """评测 checkpoint：生成 rollout groups 并评分，输出到 config 决定的目录。"""
    from graspo.cli.tools import run_evaluate

    summary = run_evaluate(
        args.config,
        args.data,
        checkpoint=args.checkpoint,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_analyze_profile(args: argparse.Namespace) -> int:
    """汇总运行目录指标，落盘六份分析文件并打印文件路径（不打印数据）。"""
    from graspo.cli.tools import run_analyze

    run_analyze(args.run_dirs, skip_warmup_steps=args.skip_warmup_steps)
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    # 锁卡守卫在构建启动计划之前执行（fail-closed）：未显式锁卡 / 含生产卡 6,7 /
    # 超过 4 卡，一律拒绝，连子进程都不创建。容器侧 runtime 的 `void` 哨兵按
    # 实测可见卡判定（F-1，见 core/gpu_guard 模块 docstring）。
    devices = require_gpu_lock_or_exit()
    plan = build_launch_plan(args.config, smoke=bool(getattr(args, "smoke", False)))
    print(
        json.dumps(
            {
                "backend": plan.backend,
                "uses_torchrun": plan.uses_torchrun,
                "nnodes": plan.nnodes,
                "nproc_per_node": plan.nproc_per_node,
                "gpu_lock": list(devices),
                "command": plan.command,
            },
            ensure_ascii=False,
        )
    )
    completed = subprocess.run(plan.command, env=plan.env, check=False)
    return int(completed.returncode)


def build_launch_plan(
    config_path: str | Path,
    config: GraspoConfig | None = None,
    *,
    smoke: bool = False,
) -> LaunchPlan:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise SystemExit(f"Config file does not exist: {config_path}")
    config = config or GraspoConfig.from_yaml(config_path)

    from graspo.flow.backend_selection import select_backend

    selection = select_backend(config)
    launch = config.launch

    # msswift 后端：委托给 ms-swift CLI
    if selection.name == "msswift":
        return _build_msswift_launch_plan(config_path, config, smoke=smoke)

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
        command = _torchrun_prefix(python) + [
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

    if smoke:
        # 冒烟模式：跑 1 步即停，验证环境链路（模型加载、多模态、训练 forward）。
        # 通过追加 --smoke 传给 worker，不修改用户 config 文件。
        command.append("--smoke")

    return LaunchPlan(
        command=command,
        env=env,
        backend=selection.name,
        uses_torchrun=uses_torchrun,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
    )


def _build_msswift_launch_plan(
    config_path: Path,
    config: GraspoConfig,
    *,
    smoke: bool = False,
) -> LaunchPlan:
    """构建 ms-swift 后端的启动计划：**进程内 Python API 路线**（决策 D6）。

    **为什么不再把 graspo YAML 交给 ms-swift CLI**（E1 遗留问题的闭合）：
    ms-swift 的 ``swift sft|rlhf --config <yaml>`` 只认它自己的参数 YAML，不认识
    graspo 的配置段；E1 的旧实现正是这么做的，因此那条链路从未真正跑通
    （E1 报告 §5.4 第 2 条已如实记录）。按 D6，主路线是 ``import swift`` 作库：
    训练在 ``graspo.cli.train_worker`` 进程内完成，配置→ms-swift 参数的映射由
    ``flow/msswift/_config_mapping.py`` 负责（单一真相源），训练器由
    ``flow/msswift/trainer.py``（RL）与 ``flow/msswift/sft_trainer.py``（SFT）提供。

    因此本函数产出的命令与 native 后端**同形**（同一个 worker 入口），差别只在
    并行层：native 用 ``dp×tp×pp`` 推导进程数，msswift 用 ``msswift.nproc_per_node``
    （S1 数据并行，ms-swift 侧就是 torchrun 的进程数）。

    Args:
        config_path: graspo YAML 路径（作为 worker 的输入定位参数，§10.1）。
        config: 已校验的 ``GraspoConfig``。
        smoke: 冒烟边界，透传给 worker（不修改 config 文件）。

    Returns:
        ``LaunchPlan``；``uses_torchrun`` 表示是否需要 torchrun 拉起多进程。
    """
    from graspo.flow.msswift._config_mapping import launcher_env

    python = str(config.launch.python or sys.executable)
    nnodes = int(config.launch.nnodes)
    if nnodes < 1:
        raise SystemExit("launch.nnodes must be >= 1")

    # 进程数真相源：msswift 段优先（与 ms-swift 的 NPROC_PER_NODE 对齐），
    # 否则回落到 launch.nproc_per_node。两者都不给、也不是多节点时，单进程即可
    # （GRPO/SFT 的单卡路径不需要分布式初始化）。
    if config.msswift.nproc_per_node is not None:
        nproc_per_node = int(config.msswift.nproc_per_node)
    else:
        nproc_per_node = int(config.launch.nproc_per_node or 1)
    if nproc_per_node < 1:
        raise SystemExit("msswift.nproc_per_node / launch.nproc_per_node must be >= 1")

    _validate_launch_paths(config)

    env = _build_launch_env(config)
    # S1 数据并行：ms-swift 侧由 launcher 环境变量承载（T1 复验结论）。显式覆盖，
    # 让用户只在 YAML 里配置一处（§1.4）。
    env.update(launcher_env(config))
    env.setdefault("NPROC_PER_NODE", str(nproc_per_node))

    uses_torchrun = nnodes * nproc_per_node > 1
    if uses_torchrun:
        command = _torchrun_prefix(python) + [
            f"--nnodes={nnodes}",
            f"--node_rank={int(config.launch.node_rank)}",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_addr={config.launch.master_addr}",
            f"--master_port={int(config.launch.master_port)}",
            "-m",
            "graspo.cli.train_worker",
            "--config",
            str(config_path),
        ]
    else:
        command = [python, "-m", "graspo.cli.train_worker", "--config", str(config_path)]

    if smoke:
        command.append("--smoke")

    return LaunchPlan(
        command=command,
        env=env,
        backend="msswift",
        uses_torchrun=uses_torchrun,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
    )


def _resolve_nproc_per_node(config: GraspoConfig, backend: str) -> int:
    launch = config.launch
    if launch.nproc_per_node is not None:
        nproc_per_node = int(launch.nproc_per_node)
    else:
        expected_world = _native_world_size(config)
        nnodes = int(launch.nnodes)
        if expected_world % nnodes != 0:
            raise SystemExit(
                "native world size must divide evenly across launch.nnodes "
                f"({expected_world} % {nnodes} != 0)"
            )
        nproc_per_node = expected_world // nnodes
    if nproc_per_node < 1:
        raise SystemExit("launch.nproc_per_node must be >= 1")
    if nproc_per_node > 32:
        raise SystemExit(
            f"launch.nproc_per_node={nproc_per_node} exceeds sane maximum 32; "
            "check your config or reduce dp_size/tp_size/pp_size"
        )
    return nproc_per_node


def _validate_launch_world(
    config: GraspoConfig,
    backend: str,
    nnodes: int,
    nproc_per_node: int,
) -> None:
    actual_world = nnodes * nproc_per_node
    expected_world = _native_world_size(config)
    if actual_world != expected_world:
        raise SystemExit(
            "native launch world size must match "
            "dp_size × tp_size × pp_size "
            f"({actual_world} != {expected_world})"
        )
    # 防呆（§2.3 边界校验即防呆）：单节点内每节点进程数不能超过可见 GPU 数，否则
    # 多个 rank 挤同一卡并让 NCCL 默认组缓冲集中到 cuda:0——正是"每 GPU 多进程 +
    # 显存不均"的配置缺口（工作日志 P1）。
    import torch

    n_gpus = int(torch.cuda.device_count())
    if n_gpus >= 1 and nproc_per_node > n_gpus:
        raise SystemExit(
            f"launch.nproc_per_node={nproc_per_node} exceeds visible GPU count "
            f"{n_gpus}; reduce dp_size*tp_size*pp_size or expose more GPUs via --gpus."
        )


def _native_world_size(config: GraspoConfig) -> int:
    return int(config.native.dp_size) * int(config.native.tp_size) * int(config.native.pp_size)


def _validate_launch_paths(config: GraspoConfig) -> None:
    _require_config_value(config.model.model_path, "model.model_path")
    _require_config_value(config.data.train_path, "data.train_path")
    data_path = Path(config.data.train_path)
    if not data_path.is_file():
        raise SystemExit(f"data.train_path does not exist: {data_path}")
    # output_dir 现在总是有默认值（配置备份约定），但需确保目录提前创建好
    from graspo.flow.lora.lora_io import prepare_output_dir

    prepare_output_dir(config.training.output_dir, overwrite=config.training.overwrite_output_dir)


def _require_config_value(value: Any, name: str) -> None:
    text = str(value or "").strip()
    if not text or "${" in text or text.startswith("<"):
        raise SystemExit(f"{name} must be set in the YAML config")


def _build_launch_env(config: GraspoConfig) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 日志目录身份**不再**由环境变量注入（§7.1/§10.1）：worker 从同一份 config 的
    # `training.run_name` 解析（见 cli/train_worker.main → flow/logging.set_run_id），
    # 每个 rank 读到同一个值。旧的 `GRASPO_RUN_ID` 注入已删除（§18.1 不留负债）。

    src_dir = _project_src_dir()
    if src_dir.is_dir():
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(src_dir) if not current else f"{src_dir}{os.pathsep}{current}"
    return env


def _torchrun_prefix(python: str) -> list[str]:
    """Distributed launcher in the venv — no system binary dependency."""
    return [python, "-m", "torch.distributed.run"]


def _project_src_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "src"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graspo", description="GRASPO training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Launch training from a single YAML config.")
    launch.add_argument("--config", "-c", required=True)
    launch.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: run 1 training step then stop. Verifies model loading, "
        "multimodal pipeline, and training forward without a long run.",
    )
    launch.set_defaults(func=cmd_launch)

    export = subparsers.add_parser(
        "export", help="Export a native GRASPO checkpoint to a portable model artifact."
    )
    export.add_argument("--config", "-c", required=True)
    export.set_defaults(func=cmd_export)

    validate = subparsers.add_parser(
        "validate-reward",
        help="Validate reward scoring on a data file. Prints per-sample scores, writes nothing.",
    )
    validate.add_argument("--data", required=True, help="JSONL data file to score.")
    validate.add_argument(
        "--limit", type=int, default=0, help="Only score the first N samples (0 = all)."
    )
    validate.add_argument(
        "--completions", default="", help="Optional JSONL of explicit completions to score."
    )
    validate.set_defaults(func=cmd_validate_reward)

    evaluate = subparsers.add_parser(
        "evaluate-checkpoint",
        help=(
            "Evaluate a checkpoint by generating rollout groups and scoring rewards. "
            "Output goes to the config's output_dir/evaluate/."
        ),
    )
    evaluate.add_argument("--config", "-c", required=True)
    evaluate.add_argument("--data", required=True, help="Evaluation JSONL path.")
    evaluate.add_argument("--checkpoint", help="Recoverable native checkpoint directory to load.")
    evaluate.add_argument(
        "--limit", type=int, default=0, help="Optional number of samples to evaluate; 0 means all."
    )
    evaluate.set_defaults(func=cmd_evaluate_checkpoint)

    analyze = subparsers.add_parser(
        "analyze-profile",
        help="Summarize profiling + rollout attribution from one or more run directories. "
        "Writes six analysis files (profile/steps/epochs/errors/attribution/perf) into "
        "<run_dir>/logs/ and prints their paths.",
    )
    analyze.add_argument("run_dirs", nargs="+", help="One or more GRASPO output directories.")
    analyze.add_argument(
        "--skip-warmup-steps",
        type=int,
        default=1,
        help="Train steps skipped for mean timing.",
    )
    analyze.set_defaults(func=cmd_analyze_profile)

    from graspo.cli.gpu_monitor import build_gpu_monitor_parser

    build_gpu_monitor_parser(subparsers)

    from graspo.cli.eval_commands import build_eval_parser

    build_eval_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
