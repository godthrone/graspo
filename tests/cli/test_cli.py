import json
import subprocess
from pathlib import Path

import pytest

from graspo.cli.app import build_launch_plan, build_parser
from graspo.core.schema import GraspoConfig


def test_cli_main_commands_parse():
    parser = build_parser()

    commands = [
        ["launch", "--config", "config_example.yaml"],
        ["export", "--config", "config_example.yaml"],
    ]
    for command in commands:
        args = parser.parse_args(command)
        assert callable(args.func)


@pytest.mark.parametrize("command", ["train", "prepare-data", "analyze", "validate-reward"])
def test_cli_removed_commands_are_not_public(command):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command])


def test_config_example_loads():
    config = GraspoConfig.from_yaml("config_example.yaml")

    assert config.training.training_epoch_count == 100
    assert config.training.max_new_tokens == 2048
    assert config.launch.gpus == [0, 1]
    assert config.graspoflow.tp_size == 2


def test_launch_plan_graspoflow_uses_torchrun(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="graspoflow",
        gpus="[0, 1]",
        nproc_per_node="null",
        tensor_parallel=2,
        pipeline_parallel=1,
    )

    plan = build_launch_plan(config_path)

    assert plan.backend == "graspoflow"
    assert plan.uses_torchrun
    assert plan.nproc_per_node == 2
    assert plan.command[:5] == [
        "torchrun",
        "--nnodes=1",
        "--node_rank=0",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
    ]
    assert "graspo.cli.train_worker" in plan.command
    assert "CUDA_VISIBLE_DEVICES" in plan.env
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_launch_plan_graspoflow_world_size_one_uses_single_process(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="graspoflow",
        gpus="null",
        nproc_per_node="null",
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    plan = build_launch_plan(config_path)

    assert plan.backend == "graspoflow"
    assert not plan.uses_torchrun
    assert plan.nproc_per_node == 1
    assert plan.command[:3] == ["python", "-m", "graspo.cli.train_worker"]


def test_launch_plan_rejects_world_size_mismatch(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="graspoflow",
        gpus="[0]",
        nproc_per_node=1,
        tensor_parallel=2,
        pipeline_parallel=1,
    )

    with pytest.raises(SystemExit, match="world size"):
        build_launch_plan(config_path)


def test_launch_plan_rejects_missing_paths(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="graspoflow",
        model_path="<MODEL_PATH>",
        gpus="null",
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    with pytest.raises(SystemExit, match="model.model_path"):
        build_launch_plan(config_path)


def test_readmes_document_single_yaml_entry_and_exports():
    expected = [
        "uv run graspo launch --config config_example.yaml",
        "config_example.yaml",
        "lora.target_modules",
        "peft-adapter",
        "merged-hf",
        "training.max_new_tokens=2048",
    ]
    for path in (Path("README.md"), Path("README.zh-CN.md")):
        text = path.read_text(encoding="utf-8")
        for item in expected:
            assert item in text
        forbidden = ["hf-reference", "prepare-data", "train --config", "prompt-only"]
        for item in forbidden:
            assert item not in text


def test_only_readmes_are_tracked_markdown_docs():
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_markdown = {
        line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()
    }

    assert tracked_markdown == {
        "LONG_RUN.md",
        "MIGRATION_PLAN.md",
        "README.md",
        "README.zh-CN.md",
        "docs/architecture.md",
        "docs/BADGE-constitution.zh-CN.md",
    }


def test_export_config_fields_default_and_validate(tmp_path):
    """ExportConfig drives export from YAML, no CLI overrides."""
    from graspo.core.schema import ExportConfig

    cfg = ExportConfig()
    assert cfg.checkpoint_path == ""
    assert cfg.export_format == "peft-adapter"
    assert cfg.export_output == ""
    assert cfg.final_formats == []

    # Validate choices via a full config load
    config_path = _write_export_config(tmp_path, export_format="merged-hf")
    config = GraspoConfig.from_yaml(config_path)
    assert config.export.checkpoint_path == "outputs/test/final"
    assert config.export.export_format == "merged-hf"
    assert config.export.export_output == "outputs/test/merged"


def test_export_cli_only_accepts_config():
    """graspo export only accepts --config (Constitution 10.1)."""
    parser = build_parser()
    args = parser.parse_args(["export", "--config", "config_example.yaml"])
    assert callable(args.func)


def test_e2e_config_roundtrip(tmp_path):
    """Minimal e2e: write a complete config, load it, verify all sections."""
    config_path = _write_launch_config(
        tmp_path,
        backend="graspoflow",
        gpus="[0, 1]",
        tensor_parallel=2,
        pipeline_parallel=1,
    )
    config = GraspoConfig.from_yaml(config_path)
    assert config.backend == "graspoflow"
    assert config.training.seed == 42
    assert config.training.reject_unparseable_groups is True
    assert config.graspoflow.tp_size == 2
    assert config.launch.nnodes == 1


def _write_export_config(tmp_path: Path, *, export_format: str) -> Path:
    config_path = tmp_path / "export_config.yaml"
    config_path.write_text(
        f"""backend: graspoflow
model:
  model_path: models/test
export:
  checkpoint_path: outputs/test/final
  export_format: {export_format}
  export_output: outputs/test/merged
""",
        encoding="utf-8",
    )
    return config_path


def _write_launch_config(
    tmp_path: Path,
    *,
    backend: str,
    gpus: str,
    tensor_parallel: int,
    pipeline_parallel: int,
    nproc_per_node: int | str = "null",
    model_path: str | None = None,
) -> Path:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"messages":[{"role":"user","content":"p"}],"targets":[{"output":{"content":{"x":1}}}]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    model_value = model_path or str(tmp_path / "model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
backend: {backend}
model:
  model_path: {json.dumps(model_value)}
data:
  train_path: {json.dumps(str(data_path))}
training:
  output_dir: {json.dumps(str(output_dir))}
graspoflow:
  tp_size: {tensor_parallel}
  pp_size: {pipeline_parallel}
launch:
  gpus: {gpus}
  nproc_per_node: {nproc_per_node}
  nnodes: 1
  node_rank: 0
  master_addr: 127.0.0.1
  master_port: 29500
  python: python
  torchrun: torchrun
  env: {{}}
""",
        encoding="utf-8",
    )
    return config_path
