"""``graspo launch`` 的确定性开关接线测试（默认关零变化 + 开启后显式可见）。

**为什么必须有这一组**

任务要求"不设开关时行为与现在逐字相同"——这句话必须能**机核**，不能靠人读 diff。
本文件把它落成三条可执行断言：

1. 默认配置下 ``build_launch_plan(...).env`` 里**一个**确定性开关变量都不多；
2. 默认配置下 stdout 里**一行** determinism 文本都不打；
3. 默认配置下 **worker 命令行逐字不变**（不追加 ``--determinism-spec``）。

开启时的断言方向相反：环境变量齐全、dry-run JSON 里显式列出、stdout 打印副作用。
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from graspo.cli import app
from graspo.cli.app import _build_launch_env, build_launch_plan, build_parser
from graspo.core.determinism import DETERMINISM_ENV_KNOBS, DeterminismSwitch

_KNOB_VARS = tuple(knob.env_var for knob in DETERMINISM_ENV_KNOBS)

#: ``launch`` 子命令里与确定性有关的**全部** flag（含开与关两个方向）。
#: 参数化测试据此逐项验证"flag → 开关字段"的映射，不留没接线的 flag。
_DETERMINISM_FLAGS = (
    "--determinism",
    "--determinism-strict",
    "--determinism-no-cublas-workspace",
    "--determinism-no-cudnn",
    "--determinism-no-torch-algorithms",
    "--determinism-no-nccl-algo",
    "--determinism-no-nccl-proto",
    "--determinism-nccl-deterministic",
    "--determinism-probe-first-step",
)


def _write_config(tmp_path: Path) -> Path:
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
""",
        encoding="utf-8",
    )
    return config_path


def _parse_launch_flags(*flags: str):
    """按 CLI 语法解析 launch 参数，返回 Namespace（与 ``cmd_launch`` 收到的一致）。"""
    return build_parser().parse_args(["launch", "--config", "x.yaml", *flags])


def _clean_knob_vars(monkeypatch) -> None:
    """把宿主环境里既有的同名变量摘掉——否则"零变化"会被环境噪声污染。"""
    import os

    for name in _KNOB_VARS:
        monkeypatch.delenv(name, raising=False)
    assert all(name not in os.environ for name in _KNOB_VARS)


# ── 默认关：零变化 ─────────────────────────────────────────────────────────


def test_default_off_adds_no_env_var_and_prints_nothing(tmp_path, monkeypatch, capsys):
    _clean_knob_vars(monkeypatch)
    from graspo.core.schema import GraspoConfig

    config = GraspoConfig.from_yaml(_write_config(tmp_path))

    env = _build_launch_env(config)

    assert all(name not in env for name in _KNOB_VARS), "默认关不得注入任何确定性变量"
    assert capsys.readouterr().out == "", "默认关不得多打任何一行"
    # 既有行为仍在（防止"删干净"式的假通过）。
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert "PYTHONPATH" in env


def test_default_off_command_and_plan_are_unchanged(tmp_path, monkeypatch):
    _clean_knob_vars(monkeypatch)
    config_path = _write_config(tmp_path)

    plan_default = build_launch_plan(config_path)
    plan_explicit_none = build_launch_plan(config_path, determinism=DeterminismSwitch())

    assert plan_default.determinism_env == {}
    assert plan_default.determinism_spec == ""
    assert "--determinism-spec" not in plan_default.command
    assert all(name not in plan_default.env for name in _KNOB_VARS)
    # 显式传一个"全关"开关与不传，结果逐字相同（默认值与显式全关等价）。
    assert plan_default.command == plan_explicit_none.command


def test_default_off_worker_command_has_no_determinism_argument(tmp_path, monkeypatch):
    """机核：默认关时 worker 命令行里没有任何 determinism 痕迹。"""
    _clean_knob_vars(monkeypatch)
    plan = build_launch_plan(_write_config(tmp_path), smoke=True)
    assert not any("determinism" in part for part in plan.command)
    assert plan.command[-1] == "--smoke"


# ── flag → 开关字段 的映射（唯一映射点）────────────────────────────────────


def test_no_flags_means_all_off():
    switch = app._switch_from_args(_parse_launch_flags())
    assert switch == DeterminismSwitch()
    assert switch.enabled is False


def test_master_flag_enables_the_expected_defaults():
    switch = app._switch_from_args(_parse_launch_flags("--determinism"))
    assert switch.enabled is True
    assert switch.cublas_workspace_config is True
    assert switch.cudnn is True
    assert switch.torch_deterministic_algorithms is True
    assert switch.warn_only is True
    assert switch.nccl_algo is True
    assert switch.nccl_proto is True
    # 版本相关项**默认不打开**（不假定本机 NCCL 支持）。
    assert switch.nccl_deterministic is False
    assert switch.probe_first_step is False


@pytest.mark.parametrize(
    ("flag", "field", "expected"),
    (
        ("--determinism-strict", "warn_only", False),
        ("--determinism-no-cublas-workspace", "cublas_workspace_config", False),
        ("--determinism-no-cudnn", "cudnn", False),
        ("--determinism-no-torch-algorithms", "torch_deterministic_algorithms", False),
        ("--determinism-no-nccl-algo", "nccl_algo", False),
        ("--determinism-no-nccl-proto", "nccl_proto", False),
        ("--determinism-nccl-deterministic", "nccl_deterministic", True),
        ("--determinism-probe-first-step", "probe_first_step", True),
    ),
)
def test_each_flag_maps_to_its_switch_field(flag, field, expected):
    switch = app._switch_from_args(_parse_launch_flags("--determinism", flag))
    assert getattr(switch, field) is expected


def test_every_determinism_flag_is_reachable_from_the_parser():
    """每个声称存在的 flag 都必须真被 argparse 接受并落到同名 dest（防"写了个不存在的 flag"）。"""
    for flag in _DETERMINISM_FLAGS:
        dest = flag.lstrip("-").replace("-", "_")
        args = _parse_launch_flags(flag)
        assert getattr(args, dest) is True, flag


# ── 开启：开关齐全且显式 ───────────────────────────────────────────────────


def test_enabled_injects_all_knobs_into_launch_env(tmp_path, monkeypatch, capsys):
    _clean_knob_vars(monkeypatch)
    from graspo.core.schema import GraspoConfig

    config = GraspoConfig.from_yaml(_write_config(tmp_path))
    switch = app._switch_from_args(
        _parse_launch_flags("--determinism", "--determinism-nccl-deterministic")
    )

    env = _build_launch_env(config, switch)
    out = capsys.readouterr().out

    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["NCCL_ALGO"] == "Ring"
    assert env["NCCL_PROTO"] == "Simple"
    assert env["NCCL_DETERMINISTIC"] == "1"
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in out
    assert "显存峰值" in out, "CUBLAS_WORKSPACE_CONFIG 的显存副作用必须随开关一起打印"
    # NCCL 变量的支持性结论必须显式出现（三态之一），不得静默。
    assert "支持性静态探测" in out
    assert any(token in out for token in ("已识别", "不认识", "未确认"))


def test_enabled_plan_renders_env_and_spec_for_dry_run(tmp_path, monkeypatch):
    _clean_knob_vars(monkeypatch)
    config_path = _write_config(tmp_path)

    plan = build_launch_plan(
        config_path, determinism=app._switch_from_args(_parse_launch_flags("--determinism"))
    )

    assert plan.determinism_env == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
    }
    assert plan.env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # spec 传给了 worker，且能被 worker 侧无歧义还原。
    assert "--determinism-spec" in plan.command
    spec = plan.command[plan.command.index("--determinism-spec") + 1]
    assert plan.determinism_spec == spec
    assert DeterminismSwitch.from_spec(spec) == app._switch_from_args(
        _parse_launch_flags("--determinism")
    )


def test_nccl_deterministic_stays_off_by_default_when_enabled(tmp_path, monkeypatch):
    """总开关打开但没传 ``--determinism-nccl-deterministic`` ⇒ 不注入该版本相关变量。"""
    _clean_knob_vars(monkeypatch)
    from graspo.core.schema import GraspoConfig

    config = GraspoConfig.from_yaml(_write_config(tmp_path))
    switch = app._switch_from_args(_parse_launch_flags("--determinism"))
    env = _build_launch_env(config, switch)
    assert "NCCL_DETERMINISTIC" not in env


def test_probe_can_be_enabled_without_any_pinning(tmp_path, monkeypatch):
    """A/B 的臂 A：探针开、钉定关 ⇒ 不注入任何环境变量，但 spec 里带探针声明。"""
    _clean_knob_vars(monkeypatch)
    from graspo.core.schema import GraspoConfig

    config = GraspoConfig.from_yaml(_write_config(tmp_path))
    switch = app._switch_from_args(_parse_launch_flags("--determinism-probe-first-step"))
    assert switch.enabled is False
    env = _build_launch_env(config, switch)
    assert all(name not in env for name in _KNOB_VARS)
    plan = build_launch_plan(
        _write_config(tmp_path), determinism=switch
    )
    assert plan.determinism_env == {}
    assert DeterminismSwitch.from_spec(plan.determinism_spec).probe_first_step is True


def test_cmd_launch_json_reports_determinism_env_and_spec(tmp_path, monkeypatch, capsys):
    """``graspo launch`` 的 dry-run JSON 里必须能看到将注入的变量与将传给 worker 的 spec。"""
    _clean_knob_vars(monkeypatch)
    config_path = _write_config(tmp_path)
    args = _parse_launch_flags(
        "--determinism", "--determinism-no-nccl-algo", "--determinism-no-nccl-proto"
    )
    args.config = str(config_path)
    monkeypatch.setattr(app, "require_gpu_lock_or_exit", lambda: [0, 1])
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda command, env, check: types.SimpleNamespace(returncode=0),
    )

    exit_code = app.cmd_launch(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["determinism_env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert DeterminismSwitch.from_spec(payload["determinism_spec"]).enabled is True


#: 各环境变量开关的"关闭"flag（默认即为开的那些项）与"打开"flag（默认关的那些项）。
_KNOB_OFF_FLAG = {
    "cublas_workspace_config": "--determinism-no-cublas-workspace",
    "nccl_algo": "--determinism-no-nccl-algo",
    "nccl_proto": "--determinism-no-nccl-proto",
}
_KNOB_ON_FLAG = {
    "nccl_deterministic": "--determinism-nccl-deterministic",
}


@pytest.mark.parametrize("knob", DETERMINISM_ENV_KNOBS, ids=lambda k: k.env_var)
def test_every_knob_is_reachable_end_to_end(tmp_path, monkeypatch, knob):
    """表里的每个环境变量开关都必须能经 CLI 真正注入（不留"写了但没接线"的表项）。

    做法：对目标开关取"开"，对其余每一个取"关"，然后断言环境里**恰好只有**目标变量。
    """
    _clean_knob_vars(monkeypatch)
    from graspo.core.schema import GraspoConfig

    config = GraspoConfig.from_yaml(_write_config(tmp_path))
    flags = ["--determinism"]
    if knob.field in _KNOB_ON_FLAG:
        flags.append(_KNOB_ON_FLAG[knob.field])
    for other in DETERMINISM_ENV_KNOBS:
        if other.field == knob.field:
            continue
        if other.field in _KNOB_OFF_FLAG:
            flags.append(_KNOB_OFF_FLAG[other.field])

    switch = app._switch_from_args(_parse_launch_flags(*flags))
    env = _build_launch_env(config, switch)

    assert env.get(knob.env_var) == knob.default_value
    assert [name for name in _KNOB_VARS if name in env] == [knob.env_var]
