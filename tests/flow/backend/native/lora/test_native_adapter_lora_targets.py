"""native 适配器 LoRA 目标解析：全参档不解析 + PP stage 归属豁免（缺陷 N1/N2 回归）。

**背景（228 实测，2026-10-22/27）**：

- **N1**：M011（9B / SFT / **全参** / native / 1 卡）334 s 崩溃，
  ``ValueError: lora.target_modules is required``（``core/lora.py:170``）。
  真因 = native 建模型时**不按 ``tuner_type`` 分支**，全参档也去解析 LoRA 正则。
- **N2**：``pp_size>1`` 的 native 档报同一句错。真因（已推翻"LoRA 注入空集"旧诊断）=
  pipeline stage 只持有部分层，而"未实现 target"差集**没按 stage 归属豁免** ⇒
  本 stage 不持有的层被当成"必须命中"。

本文件锁三件事（都不需要 GPU / 真权重 / 真 checkpoint）：

1. **N1**：``tuner_type: full`` 不解析 ``lora.target_modules``——键缺失、甚至写成
   一条合法但不命中任何模块的正则，都不得报错，且传给模型的 target 集合为空；
2. **N1 golden**：``tuner_type: lora`` 下解析结果与该正则在全模型上的命中集合
   **逐字相同**（不因本轮改动缩小/放大），证明 LoRA 路径行为不变；
3. **N2**：stage 归属豁免**只**对"本 stage 不持有"的层生效——
   ``pp=1`` 不豁免；``pp>1`` 上本 stage 持有的层缺失照旧 fail-closed，且报错只点名该层。

★ 判据域（真实 HF 模块名）由本文件**独立枚举**（:func:`_layer_hf_targets`），不复用
``native_qwen_lora_available_targets``，避免"用被测代码验证被测代码"。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.backend.native.models.common.native_qwen_config import NativeQwenConfig
from graspo.flow.backend.native.models.qwen3 import model_adapter as qwen3_adapter_module
from graspo.flow.backend.native.models.qwen35_36 import model_adapter as qwen35_adapter_module
from graspo.flow.backend.native.parallel.placement_plan import build_placement_plan

_NUM_LAYERS = 8
_KEY_PREFIX = "model.language_model"

#: 覆盖**语言侧**（注意力 + MLP）全部层的正则——真实配置里 ``LORA_TARGET_MODULES_REGEX``
#: 的同族形态（这里另写一条：本文件要能独立算出期望集合）。
_LANGUAGE_REGEX = (
    r"^(?:model\.)?language_model\..*\.(q_proj|k_proj|v_proj|o_proj"
    r"|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)$"
)

#: 一条**合法但一条都不命中**纯文本模型的正则（视觉族）——用来证明全参档"真的没解析"
#: （若解析了，``resolve_lora_target_modules`` 会抛 "matched none of the model's …"）。
_VISUAL_ONLY_REGEX = r"^model\.visual\..*$"


# ── 独立枚举判据域（真实 HF 模块名）─────────────────────────────────────────


def _layer_type(idx: int) -> str:
    return "full_attention" if idx % 3 == 0 else "linear_attention"


def _layer_hf_targets(idx: int) -> list[str]:
    """第 *idx* 层可能被 LoRA 的真实 HF 模块名（与 HF/PEFT 命名空间一致）。"""
    prefix = f"{_KEY_PREFIX}.layers.{idx}"
    if _layer_type(idx) == "full_attention":
        attn = [f"{prefix}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
    else:
        attn = [
            f"{prefix}.linear_attn.{name}"
            for name in ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_b", "in_proj_a")
        ]
    mlp = [f"{prefix}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")]
    return attn + mlp


def _layer_hf_targets_dense(idx: int) -> list[str]:
    """qwen3（dense）族的第 *idx* 层 LoRA 可命中模块（HF 命名空间 ``model.layers.N.*``）。"""
    prefix = f"model.layers.{idx}"
    return [
        *(f"{prefix}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")),
        *(f"{prefix}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")),
    ]


def _all_hf_targets() -> list[str]:
    return [target for idx in range(_NUM_LAYERS) for target in _layer_hf_targets(idx)]


def _expected_targets(regex: str) -> set[str]:
    """这条正则在全模型上的命中集合（期望值，独立于被测实现）。"""
    pattern = re.compile(regex)
    return {target for target in _all_hf_targets() if pattern.fullmatch(target)}


def _stage_owned_layers(*, pp_size: int, pp_rank: int) -> tuple[int, ...]:
    """本 stage 持有的层号——直接取自仓位规划（stage 归属的唯一真相源）。"""
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3_5_text",
        num_hidden_layers=_NUM_LAYERS,
        tp_size=1,
        pp_size=pp_size,
        tp_rank=0,
        pp_rank=pp_rank,
        layer_types=[_layer_type(idx) for idx in range(_NUM_LAYERS)],
    )
    return tuple(plan.local_layer_indices)


def _hf_config() -> NativeQwenConfig:
    return NativeQwenConfig(
        {
            "num_hidden_layers": _NUM_LAYERS,
            "layer_types": [_layer_type(idx) for idx in range(_NUM_LAYERS)],
            "has_vision_config": False,
        },
        family="qwen3_5_text",
        key_prefix=_KEY_PREFIX,
    )


def _hf_config_dense() -> NativeQwenConfig:
    """qwen3（dense）族的最小配置：无视觉塔、无 ``layer_types``（全注意力）。"""
    return NativeQwenConfig(
        {"num_hidden_layers": _NUM_LAYERS, "has_vision_config": False},
        family="qwen3",
        key_prefix="model",
    )


# ── 模型/构建器替身：只模拟 ``_load_model`` 真正会读的接口 ──────────────────


class _FakeNativeModel:
    """native 模型替身：**只**为本 stage 持有的层建 LoRA 目标（= 真 pp>1 行为）。"""

    def __init__(
        self,
        *,
        built_layer_indices: tuple[int, ...],
        layer_targets: Any = _layer_hf_targets,
        dropped: frozenset[str] = frozenset(),
    ) -> None:
        self.config = SimpleNamespace(has_vision_config=False)
        self.training = True
        paths: list[str] = []
        for idx in built_layer_indices:
            paths.extend(layer_targets(idx))
        self._paths = tuple(sorted(path for path in paths if path not in dropped))

    def enabled_lora_hf_module_paths(self) -> tuple[str, ...]:
        return self._paths

    def train(self, mode: bool = True) -> _FakeNativeModel:
        self.training = mode
        return self


class _BuilderRecorder:
    """``build_native_qwen_model`` 的替身：记录入参，按 placement 造 stage 内模型。"""

    def __init__(
        self,
        *,
        layer_targets: Any = _layer_hf_targets,
        dropped: frozenset[str] = frozenset(),
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._layer_targets = layer_targets
        self._dropped = dropped

    def __call__(self, **kwargs: Any) -> _FakeNativeModel:
        self.calls.append(kwargs)
        placement = kwargs["placement"]
        return _FakeNativeModel(
            built_layer_indices=tuple(placement.local_layer_indices),
            layer_targets=self._layer_targets,
            dropped=self._dropped,
        )

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "build_native_qwen_model 一次都没被调用"
        return self.calls[-1]


def _model_dir(tmp_path: Path) -> Path:
    """一个只含空索引的最小 HF 目录（``SafetensorIndex`` 不触盘读权重）。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )
    return model_dir


def _load_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    config: GraspoConfig,
    pp_rank: int = 0,
    dropped: frozenset[str] = frozenset(),
    adapter_module: Any = qwen35_adapter_module,
    adapter_cls_name: str = "Qwen35ModelAdapter",
    hf_config: Any = None,
    layer_targets: Any = _layer_hf_targets,
) -> tuple[Any, _BuilderRecorder]:
    """在无 GPU/无权重条件下跑一遍 ``_load_model``（默认 ``Qwen35ModelAdapter``）。"""
    recorder = _BuilderRecorder(layer_targets=layer_targets, dropped=dropped)
    monkeypatch.setattr(adapter_module, "build_native_qwen_model", recorder)

    adapter = getattr(adapter_module, adapter_cls_name)(config)
    # setup() 之前 ParallelDegree 的秩恒为 0；这里直接给出 setup 后的真实秩（pp_rank）。
    adapter.parallel = type(adapter.parallel)(
        rank=pp_rank,
        local_rank=pp_rank,
        world_size=int(config.native.pp_size),
        tp_size=int(config.native.tp_size),
        tp_rank=0,
        pp_size=int(config.native.pp_size),
        pp_rank=pp_rank,
        dp_size=int(config.native.dp_size),
        dp_rank=0,
    )
    adapter._load_model(
        hf_config if hf_config is not None else _hf_config(), _model_dir(tmp_path)
    )
    return adapter, recorder


def _config(
    *,
    tuner_type: str,
    target_modules: str | None,
    pp_size: int = 1,
    layer_ranges: list[list[int]] | None = None,
) -> GraspoConfig:
    return GraspoConfig(
        tuner_type=tuner_type,
        native={"tp_size": 1, "dp_size": 1, "pp_size": pp_size, "layer_ranges": layer_ranges},
        lora={"target_modules": target_modules, "r": 8, "alpha": 16, "dropout": 0.0},
        model={"model_path": "unused-in-unit-test"},
    )


# ── 缺陷 N1：全参档不得要求/解析 LoRA 目标 ──────────────────────────────────


def test_full_builds_model_without_lora_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N1-a｜``full`` + 不写 ``lora.target_modules`` ⇒ 能建模型，target 集合为空。"""
    _, recorder = _load_model(
        monkeypatch, tmp_path, config=_config(tuner_type="full", target_modules=None)
    )

    assert recorder.last["full_param"] is True
    assert recorder.last["lora_targets"] == set()


def test_full_tolerates_missing_target_modules_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N1-b｜``lora`` 段里**键不存在**的配置在 ``full`` 档下不得报错。"""
    config = GraspoConfig(
        tuner_type="full",
        native={"tp_size": 1, "dp_size": 1, "pp_size": 1},
        model={"model_path": "unused-in-unit-test"},
    )
    assert "target_modules" not in config.lora.model_fields_set

    _, recorder = _load_model(monkeypatch, tmp_path, config=config)

    assert recorder.last["lora_targets"] == set()


def test_full_does_not_even_compile_the_lora_regex(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N1-c｜``full`` + 合法但零命中的正则 ⇒ 不解析（否则会抛 "matched none …"）。"""
    _, recorder = _load_model(
        monkeypatch,
        tmp_path,
        config=_config(tuner_type="full", target_modules=_VISUAL_ONLY_REGEX),
    )

    assert recorder.last["lora_targets"] == set()


def test_lora_path_resolves_the_same_targets_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N1-d（golden）｜``lora`` 档解析结果 = 该正则在全模型上的命中集合，逐字不变。"""
    _, recorder = _load_model(
        monkeypatch,
        tmp_path,
        config=_config(tuner_type="lora", target_modules=_LANGUAGE_REGEX),
    )

    assert recorder.last["full_param"] is False
    assert recorder.last["lora_targets"] == _expected_targets(_LANGUAGE_REGEX)


# ── 缺陷 N2：PP stage 归属豁免 ──────────────────────────────────────────────


def test_pp1_does_not_exempt_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N2-a｜``pp=1``（本 rank 持有全部层）⇒ 缺一个 target 照旧报错（不豁免）。"""
    missing = f"{_KEY_PREFIX}.layers.5.mlp.down_proj"

    with pytest.raises(ValueError) as excinfo:
        _load_model(
            monkeypatch,
            tmp_path,
            config=_config(tuner_type="lora", target_modules=_LANGUAGE_REGEX, pp_size=1),
            dropped=frozenset({missing}),
        )

    assert missing in str(excinfo.value)


def test_pp2_exempts_layers_this_stage_does_not_hold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N2-b｜``pp=2`` 的非首 stage：**别的 stage 的层**不算"本机未实现" ⇒ 不报错。"""
    config = _config(tuner_type="lora", target_modules=_LANGUAGE_REGEX, pp_size=2)
    adapter, recorder = _load_model(monkeypatch, tmp_path, config=config, pp_rank=1)

    owned = tuple(adapter.placement.local_layer_indices)
    assert 0 < len(owned) < _NUM_LAYERS, owned  # 前置：stage 1 确实只持有部分层
    # 解析结果仍是**全模型**命中集合（未按 stage 缩小 —— 与 ms-swift/PEFT 同一匹配域）
    assert recorder.last["lora_targets"] == _expected_targets(_LANGUAGE_REGEX)


def test_pp2_still_fails_when_an_owned_layer_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N2-c｜``pp=2`` 但**本 stage 持有**的层没建出来 ⇒ 仍 fail-closed，且只点名该层。"""
    owned = _stage_owned_layers(pp_size=2, pp_rank=1)
    assert 0 < len(owned) < _NUM_LAYERS, owned
    not_owned = sorted(set(range(_NUM_LAYERS)) - set(owned))
    assert not_owned, owned

    missing = f"{_KEY_PREFIX}.layers.{owned[0]}.mlp.down_proj"
    out_of_stage = f"{_KEY_PREFIX}.layers.{not_owned[0]}.mlp.down_proj"

    with pytest.raises(ValueError) as excinfo:
        _load_model(
            monkeypatch,
            tmp_path,
            config=_config(tuner_type="lora", target_modules=_LANGUAGE_REGEX, pp_size=2),
            pp_rank=1,
            dropped=frozenset({missing}),
        )

    message = str(excinfo.value)
    assert missing in message
    # 判据不得退回"本 stage 不持有的层也算缺失"（那正是缺陷 N2 的旧行为）
    assert out_of_stage not in message


# ── 同族同缺陷：qwen3（dense）适配器一并锁住（pp>1 仅 manual layer_ranges 可达）────

_DENSE_REGEX = r"^model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def test_qwen3_full_builds_model_without_lora_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """qwen3 侧 N1｜``full`` + 不写 ``lora.target_modules`` ⇒ 能建模型，target 为空。"""
    _, recorder = _load_model(
        monkeypatch,
        tmp_path,
        config=_config(tuner_type="full", target_modules=None),
        adapter_module=qwen3_adapter_module,
        adapter_cls_name="Qwen3ModelAdapter",
        hf_config=_hf_config_dense(),
        layer_targets=_layer_hf_targets_dense,
    )

    assert recorder.last["full_param"] is True
    assert recorder.last["lora_targets"] == set()


def test_qwen3_pp2_manual_ranges_exempt_layers_outside_the_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """qwen3 侧 N2｜``pp=2`` + 手动层区间：非首 stage 不持有前 4 层 ⇒ 不报错。"""
    config = _config(
        tuner_type="lora",
        target_modules=_DENSE_REGEX,
        pp_size=2,
        layer_ranges=[[0, 4], [4, _NUM_LAYERS]],
    )
    adapter, recorder = _load_model(
        monkeypatch,
        tmp_path,
        config=config,
        pp_rank=1,
        adapter_module=qwen3_adapter_module,
        adapter_cls_name="Qwen3ModelAdapter",
        hf_config=_hf_config_dense(),
        layer_targets=_layer_hf_targets_dense,
    )

    assert adapter.placement.local_layer_indices == (4, 5, 6, 7)
    expected = {
        target
        for idx in range(_NUM_LAYERS)
        for target in _layer_hf_targets_dense(idx)
    }
    assert recorder.last["lora_targets"] == expected
