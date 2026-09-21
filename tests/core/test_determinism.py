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
    DETERMINISM_TORCH_KNOBS,
    DeterminismSwitch,
    NcclVariableSupport,
    apply_env_determinism,
    apply_torch_determinism,
    determinism_artifact,
    determinism_env_delta,
    determinism_verify_artifact,
    enabled_torch_knobs,
    format_determinism_banner,
    format_determinism_verify_report,
    format_nccl_support_report,
    nccl_version,
    probe_nccl_variable_support,
    torch_determinism_steps,
    verify_env_determinism,
    verify_torch_determinism,
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
    section = _enabled()  # ★ NCCL_DETERMINISTIC 已纳入总开关 ⇒ 不必再显式打开
    delta = determinism_env_delta(section)
    assert delta == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
        "NCCL_DETERMINISTIC": "1",
    }


def test_individual_knobs_can_be_switched_off():
    section = _enabled(nccl_algo=False, nccl_proto=False, nccl_deterministic=False)
    delta = determinism_env_delta(section)
    assert "NCCL_ALGO" not in delta
    assert "NCCL_PROTO" not in delta
    assert "NCCL_DETERMINISTIC" not in delta
    assert delta["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_banner_prints_side_effect_of_cublas_workspace():
    lines = format_determinism_banner(_enabled())
    text = "\n".join(lines)
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    # §2.2：副作用必须随开关一起显式出现（显存峰值风险不能只写在文档里）。
    assert "显存峰值" in text
    assert "torch.use_deterministic_algorithms(True, warn_only=True)" in text
    assert "torch.backends.cudnn.benchmark=False" in text


def test_banner_prints_side_effect_of_every_new_torch_knob():
    """新增的 torch 后端项有**性能/显存**副作用 ⇒ 必须随开关一起打印（§2.2）。

    这是"副作用只写在文档里"的负向对照：只要某条语句打印了却没有副作用后缀，
    或某条副作用文本缺了，本测试就红。
    """
    text = "\n".join(format_determinism_banner(_enabled()))
    for knob in DETERMINISM_TORCH_KNOBS:
        # 语句本身必须在（除 torch.use_deterministic_algorithms 外全部来自表）。
        assert f"torch.{knob.target}={knob.pinned_value}" in text
        assert knob.side_effect in text, f"{knob.target} 的副作用没被打印"


def test_torch_steps_are_empty_when_all_torch_knobs_off():
    section = _enabled(
        torch_deterministic_algorithms=False,
        cudnn=False,
        pin_bf16_reduced_precision_reduction=False,
        pin_fp16_reduced_precision_reduction=False,
        pin_cudnn_tf32=False,
    )
    assert torch_determinism_steps(section) == []
    assert enabled_torch_knobs(section) == ()
    # 环境变量型开关不受 torch 细项影响（两半场互相独立，§1.4）。
    assert determinism_env_delta(section)


def test_artifact_records_env_torch_and_side_effects():
    artifact = determinism_artifact(_enabled())
    assert artifact is not None
    assert artifact["enabled"] is True
    assert artifact["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # ★ 全量清单（含本包新增的三项）—— 逐条写死，新增项时这里必须一并更新。
    assert artifact["torch"] == [
        "torch.use_deterministic_algorithms(True, warn_only=True)",
        "torch.backends.cudnn.deterministic=True",
        "torch.backends.cudnn.benchmark=False",
        "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False",
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False",
        "torch.backends.cudnn.allow_tf32=False",
    ]
    assert "显存峰值" in artifact["side_effects"]["CUBLAS_WORKSPACE_CONFIG"]
    # torch 后端项的性能/显存副作用同样入产物（§2.2）。
    assert "变慢" in artifact["torch_side_effects"][
        "torch.backends.cudnn.allow_tf32=False"
    ]
    # ★ NCCL 支持性探测只针对**真正的 NCCL 变量**：CUBLAS_WORKSPACE_CONFIG 不是
    #   NCCL 变量，混进去必然报 accepted:false（看起来像"NCCL 不认它"的假结论）。
    assert {item["env_var"] for item in artifact["nccl_support"]} == {
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_DETERMINISTIC",
    }
    # 未 apply 时环境回读必须**报出偏差**（声明 ≠ 生效，不得静默）。
    assert artifact["env_verify"], "未 apply 时 env_verify 必须报出偏差"
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


def test_nccl_deterministic_is_now_part_of_the_master_switch():
    """★ 本包的改动：``NCCL_DETERMINISTIC`` 由"表里有、默认关"变成**纳入总开关**。

    旧行为（``nccl_deterministic=False``）在 AM1 的清单核对里被判为"可控而未控"。
    新行为必须两条都能机核：
    ① 总开关打开 ⇒ 注入（默认值 True，不需任何额外 flag）；
    ② 总开关**关闭** ⇒ 一条都不注入（默认关零变化，逐字不变）。
    """
    knob = next(k for k in DETERMINISM_ENV_KNOBS if k.env_var == "NCCL_DETERMINISTIC")
    assert knob.support_probe is True, "版本相关的变量必须保留支持性探测"
    assert DeterminismSwitch().nccl_deterministic is True, "已纳入总开关"
    assert determinism_env_delta(DeterminismSwitch(enabled=True))["NCCL_DETERMINISTIC"] == "1"
    assert "NCCL_DETERMINISTIC" not in determinism_env_delta(DeterminismSwitch())
    assert "NCCL_DETERMINISTIC" not in determinism_env_delta(_enabled(nccl_deterministic=False))


def test_nccl_version_is_recorded_as_string_or_none():
    version = nccl_version()
    assert version is None or isinstance(version, str)


# ── ④ 单一真相源 ───────────────────────────────────────────────────────────


def test_knob_table_fields_match_switch_fields():
    """两张表里的字段名与 :class:`DeterminismSwitch` 的字段名必须一一对应（§1.4）。"""
    knob_fields = {knob.field for knob in DETERMINISM_ENV_KNOBS}
    torch_fields = {knob.field for knob in DETERMINISM_TORCH_KNOBS}
    for label, fields in (("env", knob_fields), ("torch", torch_fields)):
        assert fields <= DETERMINISM_SWITCH_FIELDS, (
            f"determinism.py 的 {label} 表与 DeterminismSwitch 不同步："
            f"表里有而开关对象没有 = {sorted(fields - DETERMINISM_SWITCH_FIELDS)}"
        )
    # 反向：开关对象里的每个字段都必须被某个消费点读走，不留死字段。
    consumed = knob_fields | torch_fields | {
        "enabled",
        "warn_only",
        "torch_deterministic_algorithms",
        "probe_first_step",
    }
    assert DETERMINISM_SWITCH_FIELDS == consumed, (
        "DeterminismSwitch 的字段与消费点不一致："
        f"多出 {sorted(DETERMINISM_SWITCH_FIELDS - consumed)}"
    )
    # 两张表不得互相污染：表项类型与适用半场必须一一对应（防止"torch 项被塞进 env 表"）。
    assert all(isinstance(knob.env_var, str) for knob in DETERMINISM_ENV_KNOBS)
    assert not hasattr(DETERMINISM_ENV_KNOBS[0], "target")
    assert not hasattr(DETERMINISM_TORCH_KNOBS[0], "env_var")


def test_torch_knob_targets_have_exactly_one_definition_in_src():
    """每个 torch 属性的点分路径在全仓 ``src/`` 里只准出现一次（§1.4 单一真相源）。"""
    for knob in DETERMINISM_TORCH_KNOBS:
        hits: list[str] = []
        needle = f'"{knob.target}"'
        for path in sorted((REPO_ROOT / "src").rglob("*.py")):
            if needle in path.read_text(encoding="utf-8"):
                hits.append(str(path.relative_to(REPO_ROOT)))
        assert hits == ["src/graspo/core/determinism.py"], (
            f"{knob.target} 的路径出现了多处定义：{hits}"
        )


# ── ⑤ 默认关零变化（改后逐字不变）＋"声明 = 生效"守卫 ──────────────────────


def test_default_off_writes_no_env_and_imports_no_torch(monkeypatch):
    """默认关：``apply_env_determinism`` 不碰 ``os.environ``，且不 import torch。

    ★ 这是"默认关时行为与现在逐字相同"的机核之一：改前那条路径上根本不存在
    ``apply_env_determinism``，改后它必须**零动作**（不是"写入了同样的值"）。
    """
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("NCCL_DETERMINISTIC", raising=False)
    import sys

    sentinel = object()
    original = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        assert apply_env_determinism(DeterminismSwitch()) == {}
        assert apply_env_determinism(None) == {}  # type: ignore[arg-type]
        assert DeterminismSwitch().to_spec() == ""  # 命令行一个字符都不追加
        assert format_determinism_verify_report(DeterminismSwitch()) == []
    finally:
        if original is sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original  # type: ignore[assignment]


def test_apply_env_determinism_lands_the_values_and_verify_confirms(monkeypatch):
    """开启时取值必须真的进 ``os.environ``，且回读核实在 apply 前报偏差、apply 后干净。

    ★ 负向对照（改前实测的真实形态）：只声明不落地 ⇒ 回读必须报出偏差，不得静默。
    """
    for name in ("CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "NCCL_PROTO", "NCCL_DETERMINISTIC"):
        monkeypatch.delenv(name, raising=False)
    switch = DeterminismSwitch(enabled=True)

    # ① 未 apply ⇒ 四项全部报偏差（"开关开着但这些项没生效"必须看得见）。
    before = verify_env_determinism(switch)
    assert len(before) == 4
    assert all("未设置" in item for item in before)

    # ② apply ⇒ 写入的增量逐条等于渲染结果，回读偏差清零。
    applied = apply_env_determinism(switch)
    assert applied == determinism_env_delta(switch)
    assert verify_env_determinism(switch) == []

    # ③ 关掉某一项 ⇒ 该项不再被要求（不把"没要求"误报成"没生效"）。
    off = DeterminismSwitch(enabled=True, nccl_deterministic=False)
    assert verify_env_determinism(off) == []
    assert "NCCL_DETERMINISTIC" not in determinism_env_delta(off)


class _FakeTorch:
    """torch 的最小替身：只有被表引用的属性，值可读可写（测试不需要真 torch）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

        class _Dotted:
            def __init__(self) -> None:
                self.deterministic = False
                self.benchmark = True
                self.allow_tf32 = True

        class _Matmul:
            def __init__(self) -> None:
                self.allow_bf16_reduced_precision_reduction = True
                self.allow_fp16_reduced_precision_reduction = True

        class _Cuda:
            def __init__(self) -> None:
                self.matmul = _Matmul()

        class _Cudnn:
            def __init__(self) -> None:
                self.deterministic = False
                self.benchmark = True
                self.allow_tf32 = True

        class _Backends:
            def __init__(self) -> None:
                self.cuda = _Cuda()
                self.cudnn = _Cudnn()

        self.backends = _Backends()
        self._dotted = _Dotted()

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool) -> None:
        self.calls.append((mode, warn_only))


def test_torch_backend_knobs_are_applied_and_verified():
    """表驱动的落点：apply 后回读必须全部等于 ``pinned_value``（用替身，不依赖真 torch）。"""
    switch = DeterminismSwitch(enabled=True)
    fake = _FakeTorch()
    applied = apply_torch_determinism(switch, torch_module=fake)
    assert applied == torch_determinism_steps(switch)
    assert fake.calls == [(True, True)]
    assert verify_torch_determinism(switch, torch_module=fake) == []
    for knob in DETERMINISM_TORCH_KNOBS:
        node = fake
        for part in knob.target.split("."):
            node = getattr(node, part)
        assert node == knob.pinned_value


def test_verify_torch_determinism_catches_switches_that_did_not_take_effect():
    """★ 负向对照：构造"开关开着但项没生效"的变体，核实器必须拦住（否则本包白做）。

    两种形态都要报出：① 属性停留在默认值（没被 apply）；② 属性路径根本不存在。
    """
    switch = DeterminismSwitch(enabled=True)

    # ① 没 apply ⇒ 每个请求项都必须在偏差清单里（逐条带 target 与请求值）。
    untouched = _FakeTorch()
    mismatches = verify_torch_determinism(switch, torch_module=untouched)
    assert len(mismatches) == len(DETERMINISM_TORCH_KNOBS)
    assert all("★ 开关开着但没生效" in item for item in mismatches)
    for knob in DETERMINISM_TORCH_KNOBS:
        assert any(knob.target in item for item in mismatches), knob.target

    # ② 路径不存在（CPU-only 构建 / 属性改名）⇒ 必须报"读不到"，不得当作已生效。
    broken = _FakeTorch()
    del broken.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    broken_mismatches = verify_torch_determinism(switch, torch_module=broken)
    assert any("读不到该属性" in item for item in broken_mismatches)
    assert any("allow_bf16_reduced_precision_reduction" in item for item in broken_mismatches)

    # ③ 关掉开关 ⇒ 不再请求 ⇒ 不再报偏差（不把"没要求"误报成"没生效"）。
    off = DeterminismSwitch(
        enabled=True,
        cudnn=False,
        pin_bf16_reduced_precision_reduction=False,
        pin_fp16_reduced_precision_reduction=False,
        pin_cudnn_tf32=False,
    )
    assert verify_torch_determinism(off, torch_module=untouched) == []


def test_verify_report_is_printed_even_when_everything_is_fine(monkeypatch):
    """无偏差时也要打一行确认：否则"没能核实"与"全都到位"在日志里长得一样（§2.2）。

    纯渲染测试：两条分支都用替身核实器驱动，不依赖本机 torch / os.environ 状态。
    """
    switch = DeterminismSwitch(enabled=True)

    monkeypatch.setattr(
        "graspo.core.determinism.verify_env_determinism", lambda _switch: []
    )
    monkeypatch.setattr(
        "graspo.core.determinism.verify_torch_determinism", lambda _switch, **kw: []
    )
    ok_lines = format_determinism_verify_report(switch)
    assert ok_lines and "偏差 0 条" in ok_lines[0]
    assert not any("★ 未生效" in line for line in ok_lines)

    monkeypatch.setattr(
        "graspo.core.determinism.verify_torch_determinism",
        lambda _switch, **kw: ["torch.backends.cudnn.allow_tf32=False：回读实际为 True"],
    )
    bad_lines = format_determinism_verify_report(switch)
    assert "偏差 1 条" in bad_lines[0]
    assert any("★ 未生效" in line and "allow_tf32" in line for line in bad_lines)

    # 未启用 ⇒ 一行都不打（默认关零变化）。
    assert format_determinism_verify_report(DeterminismSwitch()) == []


def test_torch_knob_targets_exist_in_the_real_torch():
    """表里的点分路径必须在真 torch 上真的存在（否则 verify 只会报"读不到"）。"""
    import pytest as _pytest

    torch = _pytest.importorskip("torch")
    for knob in DETERMINISM_TORCH_KNOBS:
        node = torch
        for part in knob.target.split("."):
            node = getattr(node, part)
        assert isinstance(node, bool), f"{knob.target} 不是布尔开关：{node!r}"


def test_verify_artifact_records_request_and_readback():
    """产物第二行：请求集 + 回读值 + 偏差清单（"声明 = 生效"的落盘证据）。"""
    switch = DeterminismSwitch(enabled=True)
    fake = _FakeTorch()
    apply_torch_determinism(switch, torch_module=fake)
    # verify_artifact 读真 torch（见其 docstring）：这里只用替身无法注入，
    # 因此断言结构而不断言 torch 读数 —— 真机读数由 run 产物给出。
    record = determinism_verify_artifact(switch)
    assert record is not None
    assert record["event"] == "determinism_verify"
    assert set(record["env_requested"]) == {
        "CUBLAS_WORKSPACE_CONFIG",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_DETERMINISTIC",
    }
    assert record["torch_requested"] == torch_determinism_steps(switch)
    assert isinstance(record["mismatches"], list)
    json.dumps(record)
    assert determinism_verify_artifact(DeterminismSwitch()) is None


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
