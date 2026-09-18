"""``graspo eval`` 子命令的实现：prepare / serve-plan / run / delta。

**职责**：把 ``graspo.eval`` 的编排能力接到 CLI 上——解析 ``--eval-config``、
打印"要上机跑什么"的命令、执行评测、算 Δ。

**本文件不负责**：口径/聚合/锁卡的具体实现（在 ``graspo.eval`` 里）；
**不执行任何容器生命周期操作**（起/停/删 vLLM 容器交给
``scripts/eval_serve_vllm.sh`` 与 ``scripts/eval_cleanup.sh``，后者只打印命令）。

**CLI 参数纪律（宪法 §10.1）**：``--eval-config`` 是**输入定位参数**（指哪份配置、
哪份产物），不是配置内容；所有影响产物的取值都在配置里。因此这里没有
``--temperature``、``--output-dir``、``--gpus`` 之类的覆盖开关——那些
会破坏"一份配置对应一次产物"的承诺。

**为什么要有 ``prepare`` 子命令而不是直接跑**：用户本轮明令**不上机**。
``prepare`` 只做纯计算（识别 checkpoint 形态、解析目标目录），并在 stdout 打印
后续需要人工/后续轮次执行的命令——不碰 GPU、不起容器、不写大文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from graspo.core.schema import EvalConfig, GraspoConfig
from graspo.eval.evaluate import compute_delta, load_report
from graspo.eval.guard import (
    GpuGuardError,
    repo_relative,
    resolve_gpu_plan,
    sample_device_memory_mib,
)
from graspo.eval.merged_export import (
    CheckpointKind,
    classify_checkpoint,
    existing_merged_output,
    merge_peft_checkpoint,
    resolve_peft_adapter_dir,
)
from graspo.eval.orchestrator import (
    EVAL_REPORT_FILENAME,
    load_eval_dataset,
    resolve_eval_target,
    run_evaluation,
)


def _load_eval_config(args: argparse.Namespace) -> EvalConfig:
    """从 ``--eval-config`` 或 ``--config`` 的 ``eval:`` 段取配置（二选一）。

    ``--config`` 指向训练配置时，其 ``eval:`` 段是**可选**的（缺省为 ``None``）。
    这里把"没写 eval 段"转成一条可操作的报错，而不是让后续代码撞 NoneType
    （宪法 §13.1：错误要在边界上转成用户能看懂的信息）。
    """
    if args.eval_config:
        return EvalConfig.from_yaml(args.eval_config)
    config = GraspoConfig.from_yaml(args.config)
    if config.eval is None:
        raise SystemExit(
            f"{args.config} has no `eval:` section, so the eval chain has nothing to run.\n"
            "  Either add an `eval:` section (see samples/configs/eval_example.yaml)\n"
            "  or point --eval-config at a standalone eval YAML."
        )
    return config.eval


def cmd_export(args: argparse.Namespace) -> int:
    """把训练 checkpoint 导出为 merged-hf（唯一需要 GPU/权重的 eval 步骤）。

    - PEFT adapter（ms-swift 产物）→ 本函数内合并（``merge_peft_checkpoint``）。
    - native GRASPO checkpoint → 打印既有 ``graspo export`` 的命令，不重复实现。

    需要显式 ``--gpus``：合并虽然可以跑在 CPU 上，但在目标 GPU 服务器的镜像环境里
    ``torch`` 只在容器内可用，且必须走同一套锁卡防呆，因此这里同样要求显式卡
    列表（fail-closed），并把可见卡信息写进 stdout 供留痕。
    （目标服务器清单与连接方式见 `infra` skill；本机环境记录在 `.local/` 下。）
    """
    config = _load_eval_config(args)
    plan = resolve_gpu_plan(config.gpus)
    if not config.checkpoint_path:
        raise SystemExit("eval.checkpoint_path is required for `graspo eval export`")
    classification = classify_checkpoint(config.checkpoint_path)
    print(
        json.dumps(
            {
                "checkpoint_kind": classification.kind.value,
                "evidence": list(classification.evidence),
                "gpus": plan.devices,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if classification.kind is CheckpointKind.BASE_MODEL:
        print("checkpoint is already a servable base model — nothing to merge")
        return 0
    if classification.kind is CheckpointKind.GRASPO_NATIVE:
        print(
            "# native GRASPO checkpoint: use the existing exporter (it validates its own format)\n"
            "uv run graspo export --config <graspo config with export.checkpoint_path set>"
        )
        return 0

    if config.merged_output_dir is None:
        raise SystemExit("eval.merged_output_dir is required to merge a PEFT adapter")
    adapter = resolve_peft_adapter_dir(config.checkpoint_path)
    if adapter is None:
        raise SystemExit(
            f"no adapter_config.json found under {config.checkpoint_path} "
            "(searched 3 levels) — point eval.checkpoint_path at the PEFT output dir"
        )
    destination = merge_peft_checkpoint(
        adapter_dir=adapter,
        base_model_path=config.base_model_path,
        output_dir=config.merged_output_dir,
        allow_overwrite=args.allow_overwrite,
    )
    print(
        json.dumps(
            {
                "adapter_dir": str(adapter),
                "merged_output_dir": str(destination),
                "complete": existing_merged_output(destination),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """只读预检：识别 checkpoint 形态、解析要服务的目录、打印后续命令。

    不写大文件、不起容器、不碰 GPU（唯一例外是 ``--sample-gpu-memory`` 时
    做一次显存采样，那也是只读的 ``nvidia-smi``）。
    """
    config = _load_eval_config(args)
    model, ready_dir = resolve_eval_target(
        role=config.role,
        served_name=config.served_model_name,
        checkpoint_path=config.checkpoint_path,
        base_model_path=config.base_model_path,
        export_format=config.export_format,
        merged_output_dir=config.merged_output_dir,
    )
    plan = resolve_gpu_plan(config.gpus)
    dataset, overlap, train_subset = load_eval_dataset(
        config.dataset_path,
        train_path=config.train_dataset_path,
        train_limit=config.train_limit,
        dataset_path_for_report=repo_relative(config.dataset_path),
        train_path_for_report=(
            repo_relative(config.train_dataset_path) if config.train_dataset_path else None
        ),
    )

    payload: dict[str, object] = {
        "role": model.role,
        "served_name": model.served_name,
        "model_directory": ready_dir,
        "merged_output_dir": config.merged_output_dir,
        "needs_merge": ready_dir != config.base_model_path and model.role != "base",
        "gpus": plan.devices,
        "gpu_flag": plan.docker_gpus_flag(),
        "dataset": {
            "path": repo_relative(config.dataset_path),
            "sha256": dataset.sha256,
            "sample_count": len(dataset.samples),
            "distinct_images": dataset.distinct_image_count,
        },
        # 三态：None=未做分析 / 0=做了但重叠为空 / >0=做了且剔除了 N 条
        "overlap_analysis_performed": overlap is not None,
        "overlap_sample_count": len(overlap) if overlap is not None else None,
        "train_subset": train_subset.model_dump() if train_subset is not None else None,
        "output_dir": repo_relative(config.output_dir),
        "report_path": str(Path(config.output_dir) / EVAL_REPORT_FILENAME),
    }

    if config.checkpoint_path:
        classification = classify_checkpoint(config.checkpoint_path)
        payload["checkpoint_kind"] = classification.kind.value
        payload["checkpoint_evidence"] = list(classification.evidence)
        if classification.kind is CheckpointKind.PEFT_ADAPTER:
            adapter = resolve_peft_adapter_dir(config.checkpoint_path)
            payload["peft_adapter_dir"] = str(adapter) if adapter else None
        if classification.kind is CheckpointKind.GRASPO_NATIVE:
            payload["merge_note"] = (
                "native GRASPO checkpoint: run scripts/eval_export_checkpoint.sh "
                "(wraps the existing `graspo export --config`) before serving"
            )
        if classification.kind is CheckpointKind.PEFT_ADAPTER:
            payload["merge_note"] = (
                "PEFT adapter (ms-swift output): run scripts/eval_export_checkpoint.sh "
                "which merges via graspo.eval.merged_export.merge_peft_checkpoint"
            )

    if config.merged_output_dir and existing_merged_output(config.merged_output_dir):
        payload["merged_output_state"] = "complete (already merged; reuse it)"
    elif config.merged_output_dir:
        payload["merged_output_state"] = "absent or incomplete"

    if args.sample_gpu_memory:
        try:
            readings = sample_device_memory_mib(plan)
            payload["gpu_memory_mib"] = {
                str(reading.index): {
                    "name": reading.name,
                    "used": reading.used_mib,
                    "total": reading.total_mib,
                }
                for reading in readings
            }
        except GpuGuardError as exc:
            payload["gpu_memory_error"] = str(exc)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(_serve_command(config, ready_dir))
    return 0


def _serve_command(config: EvalConfig, model_dir: str) -> str:
    """打印起服务所需的命令骨架（由人工/后续轮次执行，不在本轮跑）。"""
    template = Path(__file__).resolve().parents[3] / "scripts" / "eval_serve_vllm.sh"
    return (
        "\n# next steps (run on the GPU host, after the lock guard is in place):\n"
        f"EVAL_MODEL_DIR={model_dir} \\\n"
        f"EVAL_SERVED_NAME={config.served_model_name} \\\n"
        f"EVAL_GPUS={config.gpus} \\\n"
        f"EVAL_PORT={_port_from_endpoint(config.endpoint)} \\\n"
        f"  bash {template}\n"
        f"uv run graspo eval run --eval-config <this file>\n"
    )


def _port_from_endpoint(endpoint: str) -> int:
    """从 ``http://host:port`` 里取出端口；取不到时用 v3 的历史端口。"""
    tail = endpoint.rstrip("/").rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 18889


def cmd_run(args: argparse.Namespace) -> int:
    """执行评测：连 vLLM → 推理 test 集 → 聚合 → 落盘产物。"""
    config = _load_eval_config(args)
    model, _ready_dir = resolve_eval_target(
        role=config.role,
        served_name=config.served_model_name,
        checkpoint_path=config.checkpoint_path,
        base_model_path=config.base_model_path,
        export_format=config.export_format,
        merged_output_dir=config.merged_output_dir,
    )
    dataset, overlap, train_subset = load_eval_dataset(
        config.dataset_path,
        train_path=config.train_dataset_path,
        train_limit=config.train_limit,
        dataset_path_for_report=repo_relative(config.dataset_path),
        train_path_for_report=(
            repo_relative(config.train_dataset_path) if config.train_dataset_path else None
        ),
    )
    report = run_evaluation(
        base_url=config.endpoint,
        model=model,
        dataset=dataset,
        output_dir=config.output_dir,
        gpus=config.gpus,
        seed=config.seed,
        max_workers=config.max_workers,
        overlap_indices=overlap,
        train_subset=train_subset,
        environment_fingerprint=_environment_fingerprint(),
        dataset_split=config.dataset_split,
        dataset_path_for_report=repo_relative(config.dataset_path),
    )
    summary = report.summary
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "report": str(Path(config.output_dir) / EVAL_REPORT_FILENAME),
                "role": report.model.role,
                "temperature": report.decoding.temperature,
                "criteria_version": report.criteria.version,
                "samples_total": summary.sample_count_total,
                "samples_valid": summary.sample_count_valid,
                "samples_error": summary.sample_count_error,
                "correct": summary.correct,
                "accuracy_percent": summary.accuracy_percent,
                "overlap_analysis_performed": summary.overlap_analysis_performed,
                "accuracy_percent_excluding_overlap": summary.accuracy_percent_excluding_overlap,
                "train_subset": (
                    report.train_subset.model_dump() if report.train_subset is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_delta(args: argparse.Namespace) -> int:
    """Δ 配对：base 产物 vs 训练后产物 → 绝对百分点差 + 阈值结论。"""
    before = load_report(args.before)
    after = load_report(args.after)
    config = _load_eval_config(args) if (args.eval_config or args.config) else None
    result = compute_delta(
        before,
        after,
        graspo_threshold_pp=config.graspo_threshold_pp if config else 20.0,
        sft_threshold_percent=config.sft_threshold_percent if config else 50.0,
        expect_role_before="" if args.allow_any_baseline_role else "base",
        role_after=args.after_role,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _environment_fingerprint() -> dict[str, object]:
    """采集可复现所需的环境指纹。**不含宿主绝对路径**（宪法 §15）。"""
    from graspo.eval.guard import sample_driver_version

    return {
        "python_version": sys.version.split()[0],
        "driver_version": sample_driver_version(),
        "hostname_recorded": False,
        "cwd_relative": os.path.basename(os.getcwd()),
    }


def build_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册 ``graspo eval`` 子命令组（prepare / run / delta）。"""
    eval_parser = subparsers.add_parser(
        "eval",
        help=(
            "Evaluate a model on the ELAM V5 test set through vLLM "
            "(temperature locked to 0). Subcommands: prepare / run / delta."
        ),
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    prepare = eval_sub.add_parser(
        "prepare",
        help=(
            "Read-only preflight: classify the checkpoint, resolve the servable directory, "
            "print the commands you will need. Starts nothing."
        ),
    )
    _add_config_flags(prepare)
    prepare.add_argument(
        "--sample-gpu-memory",
        action="store_true",
        help="Also sample memory on the visible cards only (read-only nvidia-smi -i).",
    )
    prepare.set_defaults(func=cmd_prepare)

    export = eval_sub.add_parser(
        "export",
        help=(
            "Export a checkpoint to merged-hf so it can be served: merges a PEFT/LoRA "
            "adapter (ms-swift output) into the base model. Native GRASPO checkpoints "
            "are delegated to the existing `graspo export`."
        ),
    )
    _add_config_flags(export)
    export.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Allow replacing a non-empty merged_output_dir. Off by default (pre-authorized "
            "fallback, constitution §3.3). The old contents are listed before removal."
        ),
    )
    export.set_defaults(func=cmd_export)

    run = eval_sub.add_parser("run", help="Run the evaluation against a running vLLM server.")
    _add_config_flags(run)
    run.set_defaults(func=cmd_run)

    delta = eval_sub.add_parser(
        "delta",
        help="Pair a base report with a post-training report and compare against the thresholds.",
    )
    delta.add_argument("--before", required=True, help="Baseline eval_report.json (base model).")
    delta.add_argument("--after", required=True, help="Post-training eval_report.json.")
    delta.add_argument("--eval-config", default="", help="Optional EvalConfig YAML for thresholds.")
    delta.add_argument(
        "--config", "-c", default="", help="Optional GraspoConfig YAML (eval: section)."
    )
    delta.add_argument(
        "--after-role",
        default=None,
        choices=["base", "after", "sft", "other"],
        help="Override the role of the --after report (drives which threshold applies).",
    )
    delta.add_argument(
        "--allow-any-baseline-role",
        action="store_true",
        help=(
            "Do NOT require the baseline report to be the base model. "
            "Off by default: the GRASPO delta anchor is base, by user ruling."
        ),
    )
    delta.set_defaults(func=cmd_delta)


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """加配置定位参数。二者互斥（一份配置是唯一真相源，§1.4）。"""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--eval-config",
        default="",
        help="Path to a standalone EvalConfig YAML (dataset/model/gpus/output only).",
    )
    group.add_argument(
        "--config",
        "-c",
        default="",
        help="Path to a GraspoConfig YAML; its `eval:` section is used.",
    )
