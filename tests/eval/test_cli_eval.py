"""``graspo eval`` CLI 的解析与只读预检测试。

重点：
1. ``eval`` 子命令组可解析（prepare / export / run / delta）；
2. ``--eval-config`` 与 ``--config`` **互斥**且必需（一份配置是唯一真相源，§1.4）；
3. ``eval prepare`` 是**只读**的——不启容器、不碰 GPU、不写大文件；
4. 没有 ``eval:`` 段的训练配置会得到**可操作**的报错，而不是 NoneType 崩溃。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspo.cli.app import build_parser

#: 本模块夹具用 gpus "0,1"，并把 6/7 声明为保留（生产）卡——显式声明部署事实，
#: 不依赖 `core.gpu_guard` 里曾经写死的元组（2026-09-28 裁定，见 gpu_guard 模块头）。
_DEPLOYMENT_FACT = {
    "GRASPO_ALLOWED_GPU_INDICES": "0,1,2,3,4,5",
    "GRASPO_RESERVED_GPU_INDICES": "6,7",
}


@pytest.fixture(autouse=True)
def _inject_deployment_fact(monkeypatch):
    for key, value in _DEPLOYMENT_FACT.items():
        monkeypatch.setenv(key, value)


def _eval_yaml(tmp_path: Path, **overrides: str) -> Path:
    """写一份可直接用于 `eval prepare` 的配置（base 角色，无需 checkpoint）。"""
    data_dir = tmp_path / "data"
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "a.jpg").write_bytes(b"\xff\xd8\xff")
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "../images/a.jpg"},
                    {"type": "text", "text": "go"},
                ],
            }
        ],
        "targets": [
            {"output": {"tool_calls": [{"name": "rotate_arm", "arguments": {"action_type": "l"}}]}}
        ],
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "test.jsonl").write_text(
        json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fields = {
        "dataset_path": str(data_dir / "test.jsonl"),
        "base_model_path": str(tmp_path / "models" / "base"),
        # 产物路径必须绝对（fail-closed）：相对路径的落点会随进程 cwd 漂移。
        # 目录放 tmp_path 下，幂等；只要求"绝对 + 含 /.local/"。
        "output_dir": str(tmp_path / ".local" / "eval" / "runs" / "cli-test"),
        "role": "base",
        "gpus": "0,1",
        "served_model_name": "graspo-eval",
        "endpoint": "http://127.0.0.1:18889",
    }
    fields.update(overrides)
    path = tmp_path / "eval.yaml"
    path.write_text(
        "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n", encoding="utf-8"
    )
    return path


def test_eval_subcommands_parse():
    parser = build_parser()
    cases = [
        ["eval", "prepare", "--eval-config", "e.yaml"],
        ["eval", "prepare", "--config", "c.yaml"],
        ["eval", "export", "--eval-config", "e.yaml"],
        ["eval", "run", "--eval-config", "e.yaml"],
        ["eval", "delta", "--before", "b.json", "--after", "a.json"],
    ]
    for case in cases:
        args = parser.parse_args(case)
        assert callable(args.func), case


def test_eval_config_flags_are_mutually_exclusive_and_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "run", "--eval-config", "e.yaml", "--config", "c.yaml"])


def test_eval_parser_has_no_output_overriding_flags():
    """配置驱动纪律（§10.1）：不得存在 --output-dir / --temperature / --gpus 覆盖开关。"""
    parser = build_parser()
    for command in (["eval", "run", "--eval-config", "e.yaml"],):
        args = parser.parse_args(command)
        for forbidden in ("output_dir", "temperature", "gpus", "output"):
            assert not hasattr(args, forbidden), f"{forbidden} must not be a CLI flag"


def test_eval_prepare_is_read_only_and_reports_plan(tmp_path, capsys):
    """prepare 只做识别与打印——product 里不该出现任何容器/GPU 启动痕迹。"""
    from graspo.cli.eval_commands import cmd_prepare

    config_path = _eval_yaml(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["eval", "prepare", "--eval-config", str(config_path)])
    assert cmd_prepare(args) == 0

    captured = capsys.readouterr().out
    payload = json.loads(captured[: captured.index("\n}\n") + 3])
    assert payload["role"] == "base"
    assert payload["model_directory"] == str(tmp_path / "models" / "base")
    assert payload["gpus"] == [0, 1]
    assert payload["gpu_flag"] == '"device=0,1"'
    assert payload["dataset"]["sample_count"] == 1
    # 重叠分析溯源：未提供训练集时，产物里必须能看出"没做"
    assert payload["overlap_analysis_performed"] is False
    assert payload["overlap_sample_count"] is None
    assert payload["train_subset"] is None
    # prepare 不创建产物目录
    assert not (tmp_path / ".local").exists()
    # 打印的后续命令里包含起服务与跑评测两步
    assert "eval_serve_vllm.sh" in captured
    assert "graspo eval run" in captured


def test_prepare_reports_overlap_analysis_provenance_when_train_set_given(tmp_path, capsys):
    """给了训练集 ⇒ 产物必须能看出"做过分析"，并带上切片 sha256 与长度。"""
    from graspo.cli.eval_commands import cmd_prepare

    config_path = _eval_yaml(
        tmp_path,
        train_dataset_path=str(tmp_path / "data" / "test.jsonl"),
    )
    parser = build_parser()
    args = parser.parse_args(["eval", "prepare", "--eval-config", str(config_path)])
    assert cmd_prepare(args) == 0

    captured = capsys.readouterr().out
    payload = json.loads(captured[: captured.index("\n}\n") + 3])
    assert payload["overlap_analysis_performed"] is True
    assert payload["overlap_sample_count"] == 1
    subset = payload["train_subset"]
    assert subset["sample_count"] == 1
    assert len(subset["sha256"]) == 64


def test_prepare_rejects_production_cards(tmp_path):
    """gpus 含 6/7 时，配置加载阶段就拒绝——不等到起容器。"""
    config_path = _eval_yaml(tmp_path, gpus='"6"')
    from graspo.core.schema import EvalConfig

    with pytest.raises(SystemExit):
        EvalConfig.from_yaml(config_path)


def test_train_config_without_eval_section_gives_actionable_error(tmp_path, capsys):
    """`--config` 指向不含 eval 段的训练配置 → 明确提示，而不是撞 NoneType。"""
    from graspo.cli.eval_commands import cmd_prepare
    from graspo.core.schema import GraspoConfig

    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        """backend: native
model:
  model_path: models/test
data:
  train_path: samples/data/sample.jsonl
training:
  output_dir: outputs/test
""",
        encoding="utf-8",
    )
    config = GraspoConfig.from_yaml(train_config)
    assert config.eval is None

    parser = build_parser()
    args = parser.parse_args(["eval", "prepare", "--config", str(train_config)])
    with pytest.raises(SystemExit, match="has no `eval:` section"):
        cmd_prepare(args)
