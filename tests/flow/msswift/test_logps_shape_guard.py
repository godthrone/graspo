"""per-token logps 形状守卫的单测（T037 越界的可读化）。

覆盖四件事：
1. 切片口径的**纯函数**边界（含 T037 实测签名 S=L=2382 ⇒ 行 2381 / 列 2382）；
2. 守卫在自洽形状上**静默放行**（不改变任何数值、不抛）——用例取 T037 冻结配置插桩实测的读数；
3. 守卫在不自洽形状上抛带**六个量 + file:line** 的异常；词表维越界另抛，词表未知时不误报；
4. 接线：``GraspoMsSwiftGRPOTrainer`` 覆写了 ``_get_logps_via_local_forward``，且覆写体是
   "前置断言 + ``super()``"，不复制上游切片。

被守卫的越界与判据推导见 ``graspo/flow/msswift/_logps_shape_guard.py`` 的模块 docstring。
"""

import pytest

from graspo.flow.msswift._logps_shape_guard import (
    MS_SWIFT_LOGITS_TO_KEEP_SITE,
    MS_SWIFT_LOGPS_SLICE_SITE,
    LogpsShapeGuardError,
    check_logps_shapes,
    logps_slice_shapes,
)


class _Shape:
    """最小张量替身：只需要 ``shape`` / ``numel()`` / ``max()``。"""

    def __init__(self, shape, max_id: int = 0):
        self.shape = tuple(shape)
        self._max = max_id

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def max(self) -> int:
        return self._max


class _Cfg:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size


class _TextCfg:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class _Model:
    def __init__(self, vocab_size=248320, text: bool = True):
        self.config = _Cfg(vocab_size)
        if text:
            self.config.text_config = _TextCfg(vocab_size)


# ── ① 切片口径（纯函数）─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seq_len", "keep", "expected"),
    [
        (10, 5, (5, 5)),  # S ≥ L+1：自洽
        (6, 5, (5, 5)),  # S = L+1 边界：自洽
        (5, 5, (4, 5)),  # S = L：差 1（T037 的触发点）
        (4, 5, (3, 4)),  # S < L：差 1
        (1, 0, (0, 0)),  # 退化但自洽
    ],
)
def test_slice_shapes_table(seq_len, keep, expected):
    assert logps_slice_shapes(seq_len, keep) == expected


def test_t037_signature_is_reproduced_as_shape_pair():
    """T037 原始报错 ``index [2382,1] vs self [2381,248320]`` 的行/列对。"""
    assert logps_slice_shapes(2382, 2382) == (2381, 2382)


@pytest.mark.parametrize("bad", [0, -1])
def test_slice_shapes_rejects_bad_seq_len(bad):
    with pytest.raises(ValueError):
        logps_slice_shapes(bad, 1)


def test_slice_shapes_rejects_negative_keep():
    with pytest.raises(ValueError):
        logps_slice_shapes(4, -1)


# ── ② 自洽形状：静默放行（返回 None，不改数值）────────────────────────────


def test_guard_passes_on_live_t037_shapes():
    """T037 插桩实测的一批真实读数（S ≥ L+1）必须全部放行。"""
    for seq_len, keep in [(2411, 29), (2412, 30), (2446, 64), (2456, 74), (2398, 16)]:
        assert (
            check_logps_shapes(
                input_ids=_Shape((1, seq_len)),
                logits_to_keep=keep,
                padding_free=False,
                is_multimodal=True,
                dynamic_num_samples=False,
                model=_Model(),
            )
            is None
        )


def test_guard_returns_none_without_model():
    assert (
        check_logps_shapes(
            input_ids=_Shape((1, 100)),
            logits_to_keep=10,
            padding_free=True,
            is_multimodal=False,
            dynamic_num_samples=False,
        )
        is None
    )


# ── ③ 不自洽形状：抛，且带六个量与 file:line ─────────────────────────────


def test_guard_raises_with_six_numbers_and_sites():
    with pytest.raises(LogpsShapeGuardError) as excinfo:
        check_logps_shapes(
            input_ids=_Shape((1, 2382)),
            logits_to_keep=2382,
            padding_free=False,
            is_multimodal=True,
            dynamic_num_samples=False,
        )
    msg = str(excinfo.value)
    for needle in (
        "S=2382",
        "L=2382",
        "logits_rows=2381",
        "index_cols=2382",
        "padding_free=False",
        "is_multimodal=True",
        "dynamic_num_samples=False",
        MS_SWIFT_LOGITS_TO_KEEP_SITE,
        MS_SWIFT_LOGPS_SLICE_SITE,
    ):
        assert needle in msg, needle
    assert excinfo.value.context["predicted_logits_slice_rows"] == 2381
    assert excinfo.value.context["predicted_input_ids_for_logps_cols"] == 2382


def test_guard_raises_on_vocab_index_overflow():
    with pytest.raises(LogpsShapeGuardError) as excinfo:
        check_logps_shapes(
            input_ids=_Shape((1, 32), max_id=248320),  # == vocab_size 即越界
            logits_to_keep=8,
            padding_free=False,
            is_multimodal=True,
            dynamic_num_samples=False,
            model=_Model(vocab_size=248320),
        )
    assert "248320" in str(excinfo.value)


def test_vocab_guard_skipped_when_vocab_unknown():
    assert (
        check_logps_shapes(
            input_ids=_Shape((1, 32), max_id=10**9),
            logits_to_keep=8,
            padding_free=False,
            is_multimodal=False,
            dynamic_num_samples=False,
            model=object(),
        )
        is None
    )


# ── ④ 接线：子类真的覆写了父类方法并调用守卫（源码级断言，不需 ms-swift）──


def test_trainer_overrides_local_forward_and_calls_guard():
    import inspect
    from pathlib import Path

    from graspo.flow.msswift import trainer as trainer_mod

    source = Path(inspect.getsourcefile(trainer_mod)).read_text(encoding="utf-8")
    assert "check_logps_shapes(" in source
    assert "def _get_logps_via_local_forward(" in source
    # 覆写体必须把控制权交回父类（只加前置断言，不重写切片）：
    assert "super()._get_logps_via_local_forward(" in source
