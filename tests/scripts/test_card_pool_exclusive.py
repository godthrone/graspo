"""★ B10 卡池契约规则 1/2 的机器断言（2026-10-14 立，T022 事故驱动）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from run_guards import check_train_infer_exclusive  # noqa: E402


def test_whole_card_free_allows_inference():
    """★ 正例：整卡空闲 ⇒ 放行。"""
    ok, why = check_train_infer_exclusive("7", {})
    assert ok, why


def test_card_held_by_training_rejects_inference():
    """★ 负例：卡已被训练容器占用、此时要起推理 ⇒ 必拒（T022 事故形态）。"""
    ok, why = check_train_infer_exclusive("4", {"4": "train"})
    assert not ok
    assert "训练" in why and "拒绝" in why


def test_card_already_serving_rejects_reuse():
    ok, why = check_train_infer_exclusive("0", {"0": "infer"})
    assert not ok
