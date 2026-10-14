"""★ B10 卡池契约规则 1/2 的机器断言（2026-10-14 立，T022 事故驱动）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from run_guards import (  # noqa: E402
    assert_no_card_reuse,
    card_roles_from_containers,
    check_train_infer_exclusive,
)


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


def test_card_roles_from_containers_reads_vllm_as_infer():
    """★ 端到端接线：从容器行汇聚 {card: role}（vllm ⇒ infer；带 TIER ⇒ train）。"""
    ps = ('t033-vllm-after-1 graspo:v6 -e NVIDIA_VISIBLE_DEVICES=5 vllm-openai --served-model-name graspo-eval\n'
          'graspo-runner -e NVIDIA_VISIBLE_DEVICES=1,3 -e TIER=T030')
    roles = card_roles_from_containers(ps)
    assert roles["5"] == "infer"
    assert roles["1"] == "train" and roles["3"] == "train"


def test_assert_no_card_reuse_rejects_inference_on_training_card():
    """★ 端到端接线（负例）：卡已被训练占用、此时择卡要求起推理 ⇒ 起跑前 fail-closed。"""
    ps = 'graspo-runner -e NVIDIA_VISIBLE_DEVICES=1,3 -e TIER=T030'
    try:
        assert_no_card_reuse(["3"], ps)
    except SystemExit as exc:
        assert "B10" in str(exc)
    else:
        raise AssertionError("★ 应 fail-closed 却放行了")


def test_assert_no_card_reuse_allows_free_card():
    """★ 端到端接线（正例）：全空 ⇒ 放行。"""
    assert_no_card_reuse(["7"], "")
