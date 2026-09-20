"""确定性钉定开关的机核测试（宪法 §6 / §1.4 / §2.2）。

覆盖三条必须能机核的承诺：

1. **默认关零变化**：``enabled=False`` ⇒ 环境变量增量为空、torch 步骤清单为空、
   横幅为空；并且**产物记录为 None**（不写文件）。
2. **开启后开关齐全且显式**：环境变量增量完整、横幅逐条打印含副作用、
   ``NCCL_*`` 支持性结论显式（已识别 / 该版本不认识 / 未确认三态，绝不静默）。
3. **单一真相源**：环境变量型开关的取值只在 ``core/determinism.py`` 出现一次；
   清单里的字段名与 ``schema.py::DeterminismConfig`` 的字段一一对应。

本文件**不需要 torch**（``core/determinism.py`` 模块级不导入 torch，§1.3）。
"""

from __future__ import annotations

import json
from pathlib import Path

from graspo.core.determinism import (
    DETERMINISM_ENV_KNOBS,
    DETERMINISM_SWITCH_FIELDS,
    DeterminismSwitch,
    NcclVariableSupport,
    apply_torch_determinism,
    determinism_artifact,
    determinism_env_delta,
    format_determinism_banner,
    format_nccl_support_report,
    nccl_version,
    probe_nccl_variable_support,
    torch_determinism_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _enabled(**overrides) -> DeterminismSwitch:
    """总开关打开、细项取默认值的开关对象。"""
    return DeterminismSwitch(enabled=True, **overrides)


# ── ⓪ 开关声明通道（--determinism-spec）─────────────────────────────────────


def test_spec_roundtrip_and_default_off():
    assert DeterminismSwitch().to_spec() == "", "全关必须渲染成空 spec（零追加）"
    assert DeterminismSwitch.from_spec("") == DeterminismSwitch()
    assert DeterminismSwitch.from_spec(None) == DeterminismSwitch()

    switch = DeterminismSwitch(enabled=True, probe_first_step=True, nccl_deterministic=True)
    assert DeterminismSwitch.from_spec(switch.to_spec()) == switch


def test_spec_rejects_unknown_keys_instead_of_silently_ignoring():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="未知的 determinism 开关"):
        DeterminismSwitch.from_spec('{"enabld": 1}')
    with _pytest.raises(ValueError, match="不是合法 JSON"):
        DeterminismSwitch.from_spec("{oops}")
    with _pytest.raises(ValueError, match="必须是 JSON 对象"):
        DeterminismSwitch.from_spec("[1, 2]")


def test_bind_active_switch_is_the_single_read_point():
    from graspo.core.determinism import active_switch, bind_active_switch

    original = active_switch()
    try:
        assert active_switch() == DeterminismSwitch(), "未绑定时必须是全关（零变化）"
        bound = bind_active_switch(DeterminismSwitch(enabled=True))
        assert active_switch() is bound
    finally:
        bind_active_switch(original)


# ── ① 默认关零变化 ─────────────────────────────────────────────────────────


def test_default_off_injects_nothing_and_prints_nothing():
    section = DeterminismSwitch()
    assert determinism_env_delta(section) == {}
    assert determinism_env_delta(None) == {}
    assert torch_determinism_steps(section) == []
    assert format_determinism_banner(section) == []
    assert determinism_artifact(section) is None


def test_default_off_never_imports_torch():
    """默认关时 ``apply_torch_determinism`` 必须在**导入 torch 之前**就返回空。

    机核方式：把 ``torch`` 从 ``sys.modules`` 里摘掉并塞一个会抛异常的哨兵——
    若默认关路径真的碰了 torch，本测试会以异常失败（而不是静默通过）。
    """
    import sys

    sentinel = object()
    original = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = None  # type: ignore[assignment]  # import torch 会直接 TypeError
    try:
        assert apply_torch_determinism(DeterminismSwitch()) == []
    finally:
        if original is sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original  # type: ignore[assignment]


# ── ② 开启后开关齐全且显式 ─────────────────────────────────────────────────


def test_enabled_injects_all_selected_env_knobs():
    section = _enabled(nccl_deterministic=True)
    delta = determinism_env_delta(section)
    assert delta == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
        "NCCL_DETERMINISTIC": "1",
    }


def test_individual_knobs_can_be_switched_off():
    section = _enabled(nccl_algo=False, nccl_proto=False)
    delta = determinism_env_delta(section)
    assert "NCCL_ALGO" not in delta
    assert "NCCL_PROTO" not in delta
    assert delta["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_banner_prints_side_effect_of_cublas_workspace():
    lines = format_determinism_banner(_enabled())
    text = "\n".join(lines)
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    # §2.2：副作用必须随开关一起显式出现（显存峰值风险不能只写在文档里）。
    assert "显存峰值" in text
    assert "torch.use_deterministic_algorithms(True, warn_only=True)" in text
    assert "torch.backends.cudnn.benchmark=False" in text


def test_torch_steps_are_empty_when_all_torch_knobs_off():
    section = _enabled(torch_deterministic_algorithms=False, cudnn=False)
    assert torch_determinism_steps(section) == []
    # 环境变量型开关不受 torch 细项影响（两半场互相独立，§1.4）。
    assert determinism_env_delta(section)


def test_artifact_records_env_torch_and_side_effects():
    artifact = determinism_artifact(_enabled())
    assert artifact is not None
    assert artifact["enabled"] is True
    assert artifact["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert artifact["torch"] == [
        "torch.use_deterministic_algorithms(True, warn_only=True)",
        "torch.backends.cudnn.deterministic=True",
        "torch.backends.cudnn.benchmark=False",
    ]
    assert "显存峰值" in artifact["side_effects"]["CUBLAS_WORKSPACE_CONFIG"]
    # JSON 可序列化（采集/人工读都要能直接 dump）。
    json.dumps(artifact)


# ── ③ NCCL 支持性探测：三态显式，不得静默 ──────────────────────────────────


def test_nccl_probe_returns_empty_for_empty_input():
    assert probe_nccl_variable_support([]) == []
    assert format_nccl_support_report({}) == []


def test_nccl_probe_detects_literal_in_library(tmp_path):
    lib = tmp_path / "libnccl.so.2"
    lib.write_bytes(b"\x7fELF" + b"\x00" * 64 + b"NCCL_ALGO\x00" + b"\x00" * 16)
    results = probe_nccl_variable_support(["NCCL_ALGO", "NCCL_DETERMINISTIC"], library_paths=[lib])
    by_name = {item.env_var: item for item in results}
    assert by_name["NCCL_ALGO"].accepted is True
    assert str(lib) in by_name["NCCL_ALGO"].evidence
    # ★ 库已定位但不含字面量 ⇒ **明确判否**（该版本不认它，注入只会 warn）。
    assert by_name["NCCL_DETERMINISTIC"].accepted is False
    assert "不认此变量" in by_name["NCCL_DETERMINISTIC"].evidence


def test_nccl_probe_reports_unconfirmed_when_library_missing(tmp_path):
    results = probe_nccl_variable_support(["NCCL_ALGO"], library_paths=[tmp_path / "nope.so"])
    assert results[0].accepted is None
    assert "未确认" in results[0].evidence


def test_nccl_report_marks_silent_failure_explicitly(monkeypatch):
    report = format_nccl_support_report({"NCCL_ALGO": "Ring"})
    assert isinstance(report, list) and report
    assert "支持性静态探测" in report[0]
    assert any("运行时核实" in line for line in report)

    # 三态渲染的判词：库认得但该版本不认识 ⇒ 必须出现"不认识/静默失效"字样。
    fake_false = NcclVariableSupport(env_var="NCCL_ALGO", accepted=False, evidence="e")
    monkeypatch.setattr(
        "graspo.core.determinism.probe_nccl_variable_support",
        lambda env_vars, **kwargs: [fake_false],
    )
    text = "\n".join(format_nccl_support_report({"NCCL_ALGO": "Ring"}))
    assert "不认识" in text and "静默失效" in text

    # 探测不了 ⇒ 必须明说"未确认"，不得当成已接受。
    fake_none = NcclVariableSupport(env_var="NCCL_ALGO", accepted=None, evidence="e")
    monkeypatch.setattr(
        "graspo.core.determinism.probe_nccl_variable_support",
        lambda env_vars, **kwargs: [fake_none],
    )
    text = "\n".join(format_nccl_support_report({"NCCL_ALGO": "Ring"}))
    assert "未确认" in text


def test_nccl_deterministic_default_off():
    """版本相关的变量默认**不**注入（不假定本机支持，§2.2 显式）。"""
    knob = next(k for k in DETERMINISM_ENV_KNOBS if k.env_var == "NCCL_DETERMINISTIC")
    assert knob.support_probe is True
    assert "NCCL_DETERMINISTIC" not in determinism_env_delta(DeterminismSwitch())


def test_nccl_version_is_recorded_as_string_or_none():
    version = nccl_version()
    assert version is None or isinstance(version, str)


# ── ④ 单一真相源 ───────────────────────────────────────────────────────────


def test_knob_table_fields_match_switch_fields():
    """表里的字段名与 :class:`DeterminismSwitch` 的字段名必须一一对应（§1.4）。"""
    knob_fields = {knob.field for knob in DETERMINISM_ENV_KNOBS}
    assert knob_fields <= DETERMINISM_SWITCH_FIELDS, (
        "determinism.py 的表与 DeterminismSwitch 不同步："
        f"表里有而开关对象没有 = {sorted(knob_fields - DETERMINISM_SWITCH_FIELDS)}"
    )
    # 反向：开关对象里的每个字段都必须被某个消费点读走，不留死字段。
    consumed = knob_fields | {
        "enabled",
        "warn_only",
        "torch_deterministic_algorithms",
        "cudnn",
        "probe_first_step",
    }
    assert DETERMINISM_SWITCH_FIELDS == consumed, (
        "DeterminismSwitch 的字段与消费点不一致："
        f"多出 {sorted(DETERMINISM_SWITCH_FIELDS - consumed)}"
    )


def test_cublas_workspace_value_has_exactly_one_definition():
    """``:4096:8`` 这个取值在全仓 ``src/`` 里只准出现一次（§1.4 单一真相源）。"""
    hits: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if '":4096:8"' in text:
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == ["src/graspo/core/determinism.py"], (
        f"CUBLAS_WORKSPACE_CONFIG 的取值出现了多处定义：{hits}"
    )


def test_probe_phase_is_registered_as_diagnostic_not_step_metrics():
    """探针 phase 名两处字面量必须同步（result_judge 不能 import flow，故靠本测试守）。"""
    from graspo.core.result_judge import DIAGNOSTIC_PHASES, STEP_METRICS_PHASES
    from graspo.flow.msswift.first_step_probe import PROBE_PHASE

    assert PROBE_PHASE in DIAGNOSTIC_PHASES
    assert PROBE_PHASE not in STEP_METRICS_PHASES
