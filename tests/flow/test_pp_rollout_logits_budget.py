"""b3（缺陷 P6）：PP rollout 末 stage logits 的**启动期显存预算闸门**。

背景（228 实测，2026-09-22）：末 stage 曾对**整条序列**做 ``norm+lm_head`` ⇒
T035 形状 `(batch=64, seq≈2052) × vocab=248320` 的 logits 要 **60–93 GiB**，
80GB 卡物理不可能 ⇒ rank1 OOM 死 / 卡在分配器路径，rank0 在下一个会合点上**无界挂死**。

b1 把"只算末位"做成了代码行为；本闸门是它的**回归围栏**：
- 行为与估算**共用** `core.schema.PP_ROLLOUT_PREFILL_LAST_ONLY` ⇒ 不可能漂移；
- 一旦该常量被关回 ``False``（= 又退回整条序列 logits），T035 形状**启动期就被拒**，
  而不是跑 15 分钟后炸。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="native runtime config validation needs torch")

from graspo.core import schema as schema_module  # noqa: E402
from graspo.core.schema import GraspoConfig  # noqa: E402


def _runtime() -> Any:
    """**按调用时解析** ``graspo.flow.runtime``（不在模块级 import）。

    为什么必须这样：``tests/flow/adapters/test_adapter_plugin.py`` 会用
    ``spec_from_file_location`` 把 ``sys.modules["graspo.flow.runtime"]`` **换成一个
    新的模块对象**（为绕过 ``flow/__init__`` 的 torch 导入链）。于是"经父包属性拿到的
    runtime"与"经 ``sys.modules`` 拿到的 runtime"会是**两个不同的对象**：
    在模块级 ``from graspo.flow.runtime import validate_native_runtime_config`` +
    ``from graspo.flow import runtime as runtime_module`` 同时使用时，monkeypatch
    打在一个对象上、被测函数来自另一个 ⇒ 测试会以"莫名 DID NOT RAISE"失败
    （实测踩到）。按调用时 ``import_module`` 拿同一个对象即可免疫。
    """
    import importlib

    return importlib.import_module("graspo.flow.runtime")


def _estimate(**kwargs: Any) -> int:
    return _runtime().estimate_pp_rollout_logits_bytes(**kwargs)


def _read_vocab(path: Any) -> int | None:
    return _runtime().read_model_vocab_size(path)


def _resolve_budget(native: Any) -> int | None:
    return _runtime().resolve_pp_rollout_logits_budget_bytes(native)


def _validate(config: GraspoConfig) -> None:
    _runtime().validate_native_runtime_config(config)


#: 228 上 Qwen3.5-9B 的真实词表（嵌在 `text_config` 下，实测）。
_VOCAB = 248_320


def _model_dir(tmp_path: Path, *, vocab: int | None = _VOCAB, nested: bool = True) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if vocab is None:
        return model_dir
    payload: dict[str, Any] = {"model_type": "qwen3_5"}
    if nested:
        payload["text_config"] = {"vocab_size": vocab}
    else:
        payload["vocab_size"] = vocab
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def _t035_like(
    tmp_path: Path,
    *,
    model_dir: Path,
    budget_gib: float,
    allow: bool = True,
    pp_size: int = 2,
    train_method: str = "graspo",
) -> GraspoConfig:
    return GraspoConfig.model_validate(
        {
            "train_method": train_method,
            "backend": "native",
            "tuner_type": "full",
            "model": {"model_path": str(model_dir), "torch_dtype": "bfloat16"},
            "data": {"max_prompt_length": 8192},
            "training": {"rollout_queue_batch_size": 8, "rollout_group_size": 8},
            "native": {
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": pp_size,
                "allow_unverified_pp_rollout": allow,
                "pp_rollout_logits_budget_gib": budget_gib,
            },
        }
    )


# ── 纯计算 ──────────────────────────────────────────────────────────────────


def test_estimate_is_the_product_of_shape_and_dtype() -> None:
    # 64 × 8192 × 248320 × 2 B ≈ 232 GiB（b1 之前的整条序列口径）
    assert (
        _estimate(batch=64, positions=8192, vocab_size=_VOCAB, dtype_bytes=2)
        == 64 * 8192 * _VOCAB * 2
    )


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"batch": 0, "positions": 1, "vocab_size": 2, "dtype_bytes": 2}, "batch"),
        ({"batch": 1, "positions": 0, "vocab_size": 2, "dtype_bytes": 2}, "positions"),
        ({"batch": 1, "positions": 1, "vocab_size": 0, "dtype_bytes": 2}, "vocab_size"),
        ({"batch": 1, "positions": 1, "vocab_size": 2, "dtype_bytes": 0}, "dtype_bytes"),
    ],
)
def test_estimate_rejects_non_positive(kwargs: dict[str, int], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _estimate(**kwargs)


def test_read_model_vocab_size_handles_nested_and_missing(tmp_path: Path) -> None:
    assert _read_vocab(_model_dir(tmp_path / "nested")) == _VOCAB
    assert _read_vocab(_model_dir(tmp_path / "flat", nested=False)) == _VOCAB
    assert _read_vocab(_model_dir(tmp_path / "absent", vocab=None)) is None
    assert _read_vocab(tmp_path / "does-not-exist") is None


def test_explicit_budget_wins_over_auto(tmp_path: Path) -> None:
    cfg = _t035_like(tmp_path, model_dir=_model_dir(tmp_path), budget_gib=3.0)
    assert _resolve_budget(cfg.native) == 3 * 1024**3


def test_auto_budget_is_derived_or_explicitly_skipped(tmp_path: Path) -> None:
    """``0 = auto``：有 GPU 就用"最小可见卡 × 50%"，没有就返回 None（并告警）。"""
    cfg = _t035_like(tmp_path, model_dir=_model_dir(tmp_path), budget_gib=0.0)
    budget = _resolve_budget(cfg.native)
    assert budget is None or budget > 0


# ── 闸门行为 ────────────────────────────────────────────────────────────────


def test_pre_b1_shape_would_be_refused_at_startup(tmp_path: Path) -> None:
    """**判别力核心**：把"只算末位"关掉 ⇒ T035 形状**启动期即拒**（b3 的围栏价值）。"""
    model_dir = _model_dir(tmp_path)
    cfg = _t035_like(tmp_path, model_dir=model_dir, budget_gib=1.0)
    monkey_on = _runtime().PP_ROLLOUT_PREFILL_LAST_ONLY
    try:
        _runtime().PP_ROLLOUT_PREFILL_LAST_ONLY = False
        with pytest.raises(RuntimeError) as info:
            _validate(cfg)
        message = str(info.value)
        assert "last-stage logits" in message
        assert "budget" in message
        assert "PP_ROLLOUT_PREFILL_LAST_ONLY=False" in message
        # 估算落在 240 GiB 量级（64 × 8192 × 248320 × 2B = 260.4 GB = 242.5 GiB）
        assert "242." in message
        assert "positions=8192" in message
    finally:
        _runtime().PP_ROLLOUT_PREFILL_LAST_ONLY = monkey_on


def test_b1_shape_passes_the_gate(tmp_path: Path) -> None:
    """b1 之后的真实行为（positions=1）⇒ 估算 ~30 MiB，轻松通过。"""
    cfg = _t035_like(tmp_path, model_dir=_model_dir(tmp_path), budget_gib=1.0)
    assert _runtime().PP_ROLLOUT_PREFILL_LAST_ONLY is True
    _validate(cfg)  # 不抛


def test_gate_skips_with_an_explicit_warning_when_vocab_is_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _t035_like(tmp_path, model_dir=_model_dir(tmp_path, vocab=None), budget_gib=1.0)
    with caplog.at_level("WARNING"):
        _validate(cfg)  # 不抛：无法估算 ⇒ 显式告警 + 跳过
    assert any("预算闸门跳过" in record.message for record in caplog.records), caplog.text


def test_gate_is_scoped_to_native_pp_rollout(tmp_path: Path) -> None:
    """pp_size=1 与 SFT（无 rollout）都不受影响——46 档的既有形状逐字不变。"""
    model_dir = _model_dir(tmp_path)
    _validate(_t035_like(tmp_path, model_dir=model_dir, budget_gib=1e-6, pp_size=1))
    _validate(
        _t035_like(tmp_path, model_dir=model_dir, budget_gib=1e-6, train_method="sft", allow=False)
    )


def test_negative_budget_is_a_boundary_error(tmp_path: Path) -> None:
    cfg = _t035_like(tmp_path, model_dir=_model_dir(tmp_path), budget_gib=-1.0)
    with pytest.raises(ValueError, match="pp_rollout_logits_budget_gib"):
        _validate(cfg)


def test_prefill_last_only_constant_is_true_by_default() -> None:
    """b1 的行为与闸门估算共用这个常量；默认必须是 True（否则 T035 会被拒）。"""
    assert schema_module.PP_ROLLOUT_PREFILL_LAST_ONLY is True
