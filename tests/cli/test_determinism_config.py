"""``launch.determinism`` config 落点的接线测试（宪法 §10.1：config 优先 + 生效值落日志）。

**为什么必须有这一组**

B3 把 ``graspo launch --determinism*`` 的 12 个开关同时登记进 config
（``launch.determinism``，见 ``core/schema.py:DeterminismConfig``），兼容期语义是
「CLI 与 config 并存，**config 显式声明优先**」。这条语义必须能**机核**，不能靠人读 diff，
否则"同一 config 因 CLI 不同而改变效果"（§10.1 推论 1 禁止的形态）会静默回来。

本文件钉住四件事：

1. config 无该段 ⇒ 与合并功能引入前**逐字相同**（CLI 单独生效）；
2. config 显式字段**覆盖** CLI（含 `enabled` 总开关）；
3. config 与 :class:`DeterminismSwitch` 的字段名**逐字段同名**（防两处漂移）；
4. 生效值出现在 `graspo launch` 的启动 JSON（`determinism_effective`）里。
"""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

from graspo.cli import app
from graspo.cli.app import build_launch_plan, build_parser
from graspo.core.determinism import DETERMINISM_SWITCH_FIELDS, DeterminismSwitch
from graspo.core.schema import DeterminismConfig, GraspoConfig


def _write_config(tmp_path: Path, determinism_block: str = "") -> Path:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"messages":[{"role":"user","content":"p"}],"targets":[{"output":{"content":{"x":1}}}]}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
backend: native
model:
  model_path: {json.dumps(str(tmp_path / "model"))}
data:
  train_path: {json.dumps(str(data_path))}
training:
  output_dir: {json.dumps(str(tmp_path / "out"))}
native:
  tp_size: 1
  pp_size: 1
launch:
  nproc_per_node: 1
  nnodes: 1
  node_rank: 0
  master_addr: 127.0.0.1
  master_port: 29500
  python: python
{determinism_block}""",
        encoding="utf-8",
    )
    return config_path


def _parse_launch_flags(*flags: str):
    return build_parser().parse_args(["launch", "--config", "x.yaml", *flags])


#: 12 个开关的 config 段（全部显式声明，用于"config 全字段可声明"验证）。
def _full_block(**overrides: bool) -> str:
    lines = ["  determinism:"]
    for name in sorted(DETERMINISM_SWITCH_FIELDS):
        value = overrides.get(name, None)
        lines.append(f"    {name}: {'null' if value is None else str(value).lower()}")
    return "\n".join(lines) + "\n"


# ── 字段名对拍：config 段与 switch 类逐字段同名 ─────────────────────────────


def test_config_section_fields_match_switch_fields():
    """config 段字段集 = ``DeterminismSwitch`` 字段集（防两处各写一半，§1.4）。"""
    assert set(DeterminismConfig.model_fields) == set(DETERMINISM_SWITCH_FIELDS)
    assert {item.name for item in dataclasses.fields(DeterminismSwitch)} == set(
        DETERMINISM_SWITCH_FIELDS
    )


def test_all_fields_default_to_none():
    """缺省全 None = "未声明" ⇒ 不覆盖 CLI（兼容期语义的前提）。"""
    assert all(value is None for value in DeterminismConfig().model_dump().values())


# ── 兼容：无 config 段 ⇒ 与合并功能引入前逐字相同 ──────────────────────────


def test_absent_section_keeps_cli_behaviour(tmp_path):
    config = GraspoConfig.from_yaml(_write_config(tmp_path))
    cli_only = app._switch_from_args(_parse_launch_flags("--determinism"))
    merged = app._switch_from_args(_parse_launch_flags("--determinism"), config)
    assert merged == cli_only == DeterminismSwitch(enabled=True)

    no_flag = app._switch_from_args(_parse_launch_flags())
    assert app._switch_from_args(_parse_launch_flags(), config) == no_flag == DeterminismSwitch()


# ── config 优先：显式声明覆盖 CLI ──────────────────────────────────────────


def test_config_enabled_master_without_cli_flag(tmp_path):
    """config 里 ``enabled: true`` 等价于 CLI ``--determinism``（默认细项全开）。"""
    config = GraspoConfig.from_yaml(
        _write_config(tmp_path, _full_block(enabled=True, probe_first_step=True))
    )
    switch = app._switch_from_args(_parse_launch_flags(), config)
    assert switch.enabled is True
    assert switch.cudnn is True
    assert switch.nccl_deterministic is True
    assert switch.probe_first_step is True


def test_config_field_overrides_cli_flag(tmp_path):
    """config 的显式 ``false`` **覆盖** CLI 的 ``--determinism``（config 优先）。"""
    config = GraspoConfig.from_yaml(_write_config(tmp_path, _full_block(enabled=True, cudnn=False)))
    switch = app._switch_from_args(_parse_launch_flags("--determinism"), config)
    assert switch.enabled is True
    assert switch.cudnn is False, "config 显式声明必须覆盖 CLI"
    # 未被 config 声明的字段仍走 CLI 取值。
    assert switch.cublas_workspace_config is True


def test_config_can_turn_master_off_over_cli(tmp_path):
    """反向覆盖：config ``enabled: false`` 也能压下 CLI ``--determinism``。"""
    config = GraspoConfig.from_yaml(_write_config(tmp_path, _full_block(enabled=False)))
    switch = app._switch_from_args(_parse_launch_flags("--determinism"), config)
    assert switch.enabled is False


def test_config_knob_reaches_launch_env(tmp_path, monkeypatch):
    """config 的细项在**启动计划**里真正生效（经唯一渲染点 ``_build_launch_env``）。"""
    from graspo.cli.app import _build_launch_env

    for name in ("CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "NCCL_PROTO", "NCCL_DETERMINISTIC"):
        monkeypatch.delenv(name, raising=False)
    config = GraspoConfig.from_yaml(
        _write_config(
            tmp_path,
            _full_block(enabled=True, cublas_workspace_config=False),
        )
    )
    switch = app._switch_from_args(_parse_launch_flags(), config)
    env = _build_launch_env(config, switch)
    assert "CUBLAS_WORKSPACE_CONFIG" not in env, "config 关掉的细项不得注入"
    assert env["NCCL_ALGO"] == "Ring"


def test_switch_reaches_worker_spec(tmp_path):
    """config 生效后的 switch 经 ``--determinism-spec`` 传给 worker 且可无歧义还原。"""
    config = GraspoConfig.from_yaml(
        _write_config(tmp_path, _full_block(enabled=True, nccl_proto=False))
    )
    switch = app._switch_from_args(_parse_launch_flags(), config)
    plan = build_launch_plan(_write_config(tmp_path), determinism=switch)
    assert "--determinism-spec" in plan.command
    restored = DeterminismSwitch.from_spec(plan.determinism_spec)
    assert restored.enabled is True
    assert restored.nccl_proto is False


# ── 生效值落启动日志 ───────────────────────────────────────────────────────


def test_effective_values_are_logged_in_launch_json(tmp_path, monkeypatch, capsys):
    """★ §10.1 证据：``graspo launch`` 的启动 JSON 带 config∪CLI 合并后的全字段生效值。"""
    config_path = _write_config(tmp_path, _full_block(enabled=True, nccl_algo=False))
    args = _parse_launch_flags("--determinism")
    args.config = str(config_path)
    monkeypatch.setattr(app, "require_gpu_lock_or_exit", lambda: [4, 5])
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda command, env, check: types.SimpleNamespace(returncode=0),
    )

    exit_code = app.cmd_launch(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    effective = payload["determinism_effective"]
    assert set(effective) == set(DETERMINISM_SWITCH_FIELDS), "生效值必须是全字段快照"
    assert effective["enabled"] is True
    assert effective["nccl_algo"] is False, "config 的显式值必须体现在生效值里"
    assert "NCCL_ALGO" not in payload["determinism_env"]
