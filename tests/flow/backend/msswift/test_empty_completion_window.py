"""没有可训练位置的 micro-batch：**形状契约修复**的单测（A2 待落码件）。

覆盖四件事：

1. **修复生效**：全掩批（``L >= S`` 且 mask 全 False）⇒ 窗口收窄到 ``S-1``、mask 形状同步；
2. **正常批零赋值**：判据不成立的三类情形（``L < S`` / mask 里有 True / ``S < 2``）⇒ 返回
   ``None``，且 ``logits_to_keep`` 与 ``completion_mask`` 都是**同一个对象**（身份不变 ⇒ 逐位不变）；
3. **数学 loss 为 0**：用算法层自己的 ``masked_mean`` 验证——修复后该批仍是"全 False mask"，
   loss 严格为 0（只读引用 ``ripple.loss``，不修改它）；
4. **接线 + 入口对拍**：``GraspoMsSwiftGRPOTrainer`` 真的覆写了
   ``_get_per_token_logps_and_entropies``；用**真方法对象 + 替身父类**对拍：
   正常批 ``torch.equal`` 逐位相同且返回的是父类那个对象本身（原样透传），
   全掩批则在父类被调用**之前**已经修好（父类看到的是 ``S-1``）。
"""

import pytest
import torch


class _Batch:
    """``GRPOBatch`` 的最小替身：只需要 ``logits_to_keep`` / ``completion_mask``。"""

    def __init__(self, logits_to_keep: int, mask: torch.Tensor) -> None:
        self.logits_to_keep = logits_to_keep
        self.completion_mask = mask


def _mask(batch_size: int, width: int, *, trainable_at: int | None = None) -> torch.Tensor:
    mask = torch.zeros((batch_size, width), dtype=torch.bool)
    if trainable_at is not None:
        mask[0, trainable_at] = True
    return mask


# ── ① 修复生效：全掩批（T037 的形态：S = L，mask 全 False）──────────────────


def test_repairs_degenerate_batch_and_narrows_window():
    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window

    seq_len = 2382  # T037 实测：S = L = 2382
    batch = _Batch(logits_to_keep=seq_len, mask=_mask(1, seq_len))
    repaired = repair_empty_completion_window(batch, seq_len)

    assert repaired == {
        "seq_len": seq_len,
        "logits_to_keep_before": seq_len,
        "logits_to_keep_after": seq_len - 1,
    }
    assert batch.logits_to_keep == seq_len - 1
    assert tuple(batch.completion_mask.shape) == (1, seq_len - 1)
    assert not bool(batch.completion_mask.any())  # 仍是"一个可训练位置都没有"

    # 上游两处切片在新口径下重新自洽：行 = (S-1)、列 = (S-1)
    rows = (seq_len - 1) + 1 - 1
    cols = min(seq_len, batch.logits_to_keep)
    assert rows == cols == seq_len - 1


def test_repair_keeps_other_rows_untouched_when_batch_is_degenerate():
    """批级判据：该批没有任何可训练位置 ⇒ 收窄不影响任何真实 token（本来一个都没有）。"""

    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window

    batch = _Batch(logits_to_keep=5, mask=_mask(1, 5))
    assert repair_empty_completion_window(batch, 5) is not None
    assert not bool(batch.completion_mask.any())


# ── ② 正常批零赋值（对象身份不变 ⇒ 逐位不变）────────────────────────────────


def test_noop_when_contract_already_consistent():
    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window

    batch = _Batch(logits_to_keep=29, mask=_mask(1, 29, trainable_at=3))
    mask_before = batch.completion_mask
    assert repair_empty_completion_window(batch, 2411) is None
    assert batch.logits_to_keep == 29
    assert batch.completion_mask is mask_before  # 同一个对象：零赋值


def test_noop_when_trainable_positions_exist_even_if_L_ge_S():
    """``L >= S`` 但 mask 里有 True ⇒ **故意不修**（交给形状守卫 fail-closed）。"""

    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window

    batch = _Batch(logits_to_keep=5, mask=_mask(1, 5, trainable_at=4))
    mask_before = batch.completion_mask
    assert repair_empty_completion_window(batch, 5) is None
    assert batch.logits_to_keep == 5
    assert batch.completion_mask is mask_before


def test_noop_for_tiny_sequence():
    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window

    batch = _Batch(logits_to_keep=1, mask=_mask(1, 1))
    assert repair_empty_completion_window(batch, 1) is None
    assert batch.logits_to_keep == 1


def test_noop_when_attributes_missing():
    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window

    assert repair_empty_completion_window(None, 5) is None
    assert repair_empty_completion_window(object(), 5) is None


# ── ③ 数学 loss 为 0（用算法层自己的 masked_mean）───────────────────────────


def test_repaired_batch_yields_exactly_zero_loss():
    from graspo.flow.backend.msswift._empty_completion_window import repair_empty_completion_window
    from graspo.ripple.loss import masked_mean

    seq_len = 64
    batch = _Batch(logits_to_keep=seq_len, mask=_mask(1, seq_len))
    assert repair_empty_completion_window(batch, seq_len) is not None

    # 故意给非零的"逐 token 量"，证明是 mask 把它变成 0（而不是数值恰好为 0）
    per_token = torch.full_like(batch.completion_mask, 7.5, dtype=torch.float32)
    loss = masked_mean(per_token, batch.completion_mask.to(torch.float32))
    assert float(loss) == 0.0


# ── ④ 接线 + 入口对拍（真方法对象 + 替身父类；不需要 ms-swift，也不需要真 Trainer）──


class _StandInParent:
    """与 ms-swift ``GRPOTrainer`` 同签名的替身父类（只为对拍覆写体，不模拟训练）。"""

    def _get_per_token_logps_and_entropies(
        self, model, model_inputs, grpo_batch, compute_entropy=False, origin_data=None
    ):
        self.seen.append(
            (
                grpo_batch.logits_to_keep,
                tuple(grpo_batch.completion_mask.shape),
                compute_entropy,
                origin_data,
            )
        )
        return self.result


def _probe():
    """构造 ``MRO = [_Probe, GraspoMsSwiftGRPOTrainer, _StandInParent, object]`` 的实例。

    零参 ``super()`` 用定义类 + ``type(self).__mro__`` 解析 ⇒ 会落到 ``_StandInParent`` 上，
    **跑的是真方法对象**（不是抄一份方法体）。``__init__`` 不跑（``object.__new__``）。
    """
    from graspo.flow.backend.msswift.trainer import GraspoMsSwiftGRPOTrainer

    method = getattr(GraspoMsSwiftGRPOTrainer, "_get_per_token_logps_and_entropies", None)
    if method is None:  # pragma: no cover - 落码前执行时走这条（P1 尚未落盘）
        pytest.skip("P1 not landed yet: GraspoMsSwiftGRPOTrainer has no such override")
    probe_cls = type("_Probe", (GraspoMsSwiftGRPOTrainer, _StandInParent), {})
    probe = object.__new__(probe_cls)
    probe.seen = []
    probe.result = (torch.arange(4, dtype=torch.float32).reshape(1, 4), None)
    probe.method = method
    return probe


def test_override_passes_normal_batch_through_untouched():
    probe = _probe()
    batch = _Batch(logits_to_keep=29, mask=_mask(1, 29, trainable_at=3))
    mask_before = batch.completion_mask
    model_inputs = {"input_ids": torch.zeros((1, 2411), dtype=torch.long)}

    out = probe.method(probe, None, model_inputs, batch)

    # ① 逐位相同（与父类替身给出的张量 torch.equal）
    assert torch.equal(out[0], torch.arange(4, dtype=torch.float32).reshape(1, 4))
    # ② 原样透传：连对象身份都没变
    assert out is probe.result
    # ③ 父类看到的与传入完全一致（正常批零赋值 ⇒ 逐字节不变）
    assert probe.seen == [(29, (1, 29), False, None)]
    assert batch.completion_mask is mask_before
    assert batch.logits_to_keep == 29


def test_override_repairs_before_super_for_degenerate_batch():
    probe = _probe()
    batch = _Batch(logits_to_keep=2382, mask=_mask(1, 2382))
    model_inputs = {"input_ids": torch.zeros((1, 2382), dtype=torch.long)}

    probe.method(probe, None, model_inputs, batch)

    # 父类被调用时看到的已经是修好的口径（修复发生在 super() 之前）
    assert probe.seen == [(2381, (1, 2381), False, None)]
    assert batch.logits_to_keep == 2381


def test_override_passes_kwargs_through():
    probe = _probe()
    batch = _Batch(logits_to_keep=3, mask=_mask(1, 3, trainable_at=1))
    model_inputs = {"input_ids": torch.zeros((1, 100), dtype=torch.long)}

    probe.method(probe, None, model_inputs, batch, compute_entropy=True, origin_data={"x": 1})

    assert probe.seen == [(3, (1, 3), True, {"x": 1})]


# ── ⑤ 源码级接线断言（与 test_logps_shape_guard.py 的 ④ 同风格）──────────────


def test_trainer_calls_repair_before_super_in_logps_entrypoint():
    import inspect
    from pathlib import Path

    from graspo.flow.backend.msswift import trainer as trainer_mod

    source = Path(inspect.getsourcefile(trainer_mod)).read_text(encoding="utf-8")
    assert "def _get_per_token_logps_and_entropies(" in source
    assert "repair_empty_completion_window(" in source
    assert "super()._get_per_token_logps_and_entropies(" in source
    # 修复必须在 super() 之前（取第一次出现的位置比较）：
    assert source.index("repair_empty_completion_window(") < source.index(
        "super()._get_per_token_logps_and_entropies("
    )
    # 守卫模块保持原样：本包不修改 _logps_shape_guard.py 的判据（只是接线引用）
    assert "check_logps_shapes(" in source
