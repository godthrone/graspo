import json
import subprocess
from pathlib import Path

import pytest

from graspo.cli.app import build_launch_plan, build_parser
from graspo.core.schema import GraspoConfig


def test_cli_validate_reward():
    parser = build_parser()
    args = parser.parse_args(["validate-reward", "--data", "data/sample.jsonl", "--limit", "1"])

    assert args.func(args) == 0


def test_cli_validate_reward_tool_call_sample():
    parser = build_parser()
    args = parser.parse_args(
        ["validate-reward", "--data", "data/sample_tool_call.jsonl", "--limit", "1"]
    )

    assert args.func(args) == 0


def test_cli_main_commands_parse():
    parser = build_parser()

    commands = [
        ["launch", "--config", "config_example.yaml"],
        [
            "export",
            "--config",
            "config_example.yaml",
            "--checkpoint",
            "outputs/run/final",
            "--format",
            "peft-adapter",
            "--output",
            "adapter",
        ],
        [
            "export",
            "--config",
            "config_example.yaml",
            "--checkpoint",
            "outputs/run/final",
            "--format",
            "merged-hf",
            "--output",
            "merged",
            "--base-model",
            "model",
        ],
    ]
    for command in commands:
        args = parser.parse_args(command)
        assert callable(args.func)


@pytest.mark.parametrize("command", ["train", "prepare-data", "analyze"])
def test_cli_removed_commands_are_not_public(command):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command])


def test_config_example_loads():
    config = GraspoConfig.from_yaml("config_example.yaml")

    assert config.training.training_epoch_count == 100
    assert config.training.max_new_tokens == 2048
    assert config.launch.gpus == [0, 1]
    assert config.native_tp.tensor_model_parallel_size == 2


def test_launch_plan_native_tp_uses_torchrun(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native-tp",
        gpus="[0, 1]",
        nproc_per_node="null",
        tensor_parallel=2,
        pipeline_parallel=1,
    )

    plan = build_launch_plan(config_path)

    assert plan.backend == "native-tp"
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


def test_launch_plan_native_tp_world_size_one_uses_single_process(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native-tp",
        gpus="null",
        nproc_per_node="null",
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    plan = build_launch_plan(config_path)

    assert plan.backend == "native-tp"
    assert not plan.uses_torchrun
    assert plan.nproc_per_node == 1
    assert plan.command[:3] == ["python", "-m", "graspo.cli.train_worker"]


def test_launch_plan_rejects_world_size_mismatch(tmp_path):
    config_path = _write_launch_config(
        tmp_path,
        backend="native-tp",
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
        backend="native-tp",
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

    assert tracked_markdown == {"README.md", "README.zh-CN.md"}


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
        '{"messages":[{"role":"user","content":"p"}],"ground_truth":{"x":1}}\n',
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
backend_config:
  native_tp:
    tensor_model_parallel_size: {tensor_parallel}
    pipeline_model_parallel_size: {pipeline_parallel}
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
