"""P0 判据：组被判「不该训」时，优势必须**显式全 0**，父类标量优势不得漏进来。

实测缺陷（2000 次 loss 调用里 48 次）：``_overwrite_advantages`` 在整组被判
``perfect_skip`` 时**直接 return** ⇒ ``injected=False`` ⇒ loss 回退到 ms-swift 父类的
标量 z-score advantage 照训，而这批恰恰是全 run 梯度最大的一批（|loss| 最高 2.474，
正常路径 0.777）。

本文件钉住三件事：
1. 组被判跳过 ⇒ 写进 ``grpo_batch.advantages`` 的必须是 0（且 ``injected`` 如实为真）；
2. 没有组可判（缺标注/缺 tokenizer）时，父类优势也**不得**漏进来，且 loss 必须显式为 0；
3. 指标能一眼区分「注入且非零 / 注入但全 0（跳过）/ 回退」，回退次数必须为 0。
"""

from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from graspo.ripple.algorithm_core import GraspoAlgorithmCore  # noqa: E402
from graspo.ripple.annotation.labeler import AnnotationInput, annotate  # noqa: E402


class _FakeTokenizer:
    """每字符一个 token 的假分词器（与既有 wiring 测试同一模式）。"""

    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]

    def __call__(self, text: str, return_offsets_mapping: bool = False, **kwargs):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]

        class _Encoding:
            input_ids = ids
            offset_mapping = offsets

            @staticmethod
            def get(key, default=None):
                return getattr(_Encoding, key, default)

        return _Encoding()


#: 组内容不可解析（纯散文）⇒ classify_group 判 invalid/retry
PROSE = "这是一段散文，没有 JSON 围栏。"
#: 可解析且组内有差异 ⇒ trainable
GOOD_JSON = '```json\n{"answer": "4"}\n```'
BAD_JSON = '```json\n{"answer": "5"}\n```'
ARD_TARGETS = [{"id": None, "output": {"content": {"answer": "4"}, "reasoning": None}}]


def _config():
    from graspo.core.schema import GraspoConfig

    return GraspoConfig.model_validate({"backend": "msswift"})


def _core() -> GraspoAlgorithmCore:
    config = _config()
    return GraspoAlgorithmCore(
        reward_config=config.reward,
        policy_ratio_clip_eps=config.training.policy_ratio_clip_eps,
    )


def _sample(request_id: str, completion: str, prompt_id: str = "p1") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_id=prompt_id,
        is_truncated=False,
        messages=[{"role": "assistant", "content": completion}],
        extra={"targets": json.dumps(ARD_TARGETS)},
    )


def _annotate(stub, completion: str):
    return annotate(
        AnnotationInput(
            completion_text=completion,
            targets=ARD_TARGETS,
            tokenizer=_FakeTokenizer(),
            format_type="json",
        )
    )


def _decision_stub() -> SimpleNamespace:
    """只带 ``_compute_graspo_token_advantages`` 所需状态的最小对象。"""
    from graspo.flow.backend.msswift import trainer as trainer_module

    stub = SimpleNamespace(
        _graspo_config=_config(),
        _graspo=_core(),
        num_generations=2,
        reward_weights=torch.ones(1, dtype=torch.float32),
        _graspo_annotations={},
        _graspo_completion_text={},
        _graspo_token_advantages={},
        _graspo_skipped_keys=set(),
        _graspo_group_decisions=[],
        _graspo_injection_failures=0,
        _graspo_retry_counts={},
        _resolve_tokenizer=lambda: _FakeTokenizer(),
        _completion_text=lambda sample: trainer_module.GraspoMsSwiftGRPOTrainer._completion_text(
            stub, sample
        ),
        _warn_injection=lambda reason: trainer_module.GraspoMsSwiftGRPOTrainer._warn_injection(
            stub, reason
        ),
        _compute=trainer_module.GraspoMsSwiftGRPOTrainer._compute_graspo_token_advantages,
    )
    return stub


def _overwrite_stub() -> SimpleNamespace:
    """只带 ``_overwrite_advantages`` 所需状态的最小对象（父类优势先填成 1.0）。"""
    from graspo.flow.backend.msswift import trainer as trainer_module

    stub = SimpleNamespace(
        _graspo_token_advantages={},
        _graspo_skipped_keys=set(),
        _graspo_advantages_injected=False,
        _graspo_advantage_state="",
        _graspo_injection_failures=0,
        split_by_mini_batches=lambda samples: [samples],
        _warn_injection=lambda reason: trainer_module.GraspoMsSwiftGRPOTrainer._warn_injection(
            stub, reason
        ),
        _overwrite=trainer_module.GraspoMsSwiftGRPOTrainer._overwrite_advantages,
    )
    return stub


def _batch(rows: int = 2, tokens: int = 4) -> SimpleNamespace:
    """父类产物：advantages 全 1.0（模拟被污染前的标量 z-score advantage）。"""
    return SimpleNamespace(
        completion_mask=torch.ones(rows, tokens),
        advantages=torch.ones(rows, tokens),
        old_per_token_logps=torch.full((rows, tokens), -0.5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. 组被判跳过 ⇒ 优势显式全 0
# ══════════════════════════════════════════════════════════════════════════════


def test_compute_records_skipped_group_keys():
    """★ 被判 skip 的组必须记进 ``_graspo_skipped_keys``（否则下游只能回退）。"""
    stub = _decision_stub()
    samples = [_sample("r1", PROSE, "p1"), _sample("r2", PROSE, "p1")]
    for sample in samples:
        stub._graspo_annotations[sample.request_id] = _annotate(stub, PROSE)
        stub._graspo_completion_text[sample.request_id] = PROSE

    stub._compute(stub, samples, torch.zeros(2, 1))

    assert stub._graspo_skipped_keys == {"r1", "r2"}, (
        f"跳过的组必须登记到 _graspo_skipped_keys，实测 {stub._graspo_skipped_keys}"
    )
    assert stub._graspo_token_advantages == {}, "跳过的组不得产出训练信号"


def test_trainable_group_is_not_recorded_as_skipped():
    """阳性对照：可训练组不得进 skipped 集合。"""
    stub = _decision_stub()
    samples = [_sample("r1", GOOD_JSON), _sample("r2", BAD_JSON)]
    for sample, text in zip(samples, [GOOD_JSON, BAD_JSON]):
        stub._graspo_annotations[sample.request_id] = _annotate(stub, text)
        stub._graspo_completion_text[sample.request_id] = text

    stub._compute(stub, samples, torch.tensor([[1.0], [0.0]]))

    assert set(stub._graspo_token_advantages) == {"r1", "r2"}
    assert stub._graspo_skipped_keys == set()


def test_skipped_group_advantages_are_written_as_zeros():
    """★ 核心判据：组被判跳过时，写进 grpo_batch 的优势必须是 **全 0**（父类的 1.0 不许留）。"""
    stub = _overwrite_stub()
    stub._graspo_skipped_keys = {"r1", "r2"}
    batch = _batch()
    samples = [_sample("r1", PROSE), _sample("r2", PROSE)]

    stub._overwrite(stub, samples, [{"grpo_batch": batch}])

    assert torch.count_nonzero(batch.advantages) == 0, (
        f"组被判跳过时必须显式全 0，实测非零 {batch.advantages.tolist()}"
    )
    assert stub._graspo_advantages_injected is True, (
        "跳过也是 graspo 掌控优势的一步，injected 必须为真"
    )
    assert stub._graspo_advantage_state == "injected_zero"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 回退路径：父类优势不得漏进来，loss 必须显式为 0
# ══════════════════════════════════════════════════════════════════════════════


def test_fallback_zeroes_parent_advantages():
    """★ 没有任何组可判（缺标注/缺 tokenizer）时，父类优势也必须被清零。"""
    stub = _overwrite_stub()
    batch = _batch()
    samples = [_sample("r1", PROSE), _sample("r2", PROSE)]

    stub._overwrite(stub, samples, [{"grpo_batch": batch}])

    assert torch.count_nonzero(batch.advantages) == 0, "回退路径不得让父类标量优势漏进 loss"
    assert stub._graspo_advantages_injected is False
    assert stub._graspo_advantage_state == "fallback"
    assert stub._graspo_injection_failures == 1, "回退必须留可读警告（计数可见）"


def test_injected_group_keeps_real_token_advantages():
    """阳性对照：可训练组写入的是真实 token 级优势（不是被一起清零）。"""
    stub = _overwrite_stub()
    stub._graspo_token_advantages = {"r1": [1.0, 2.0, -1.0, 0.0], "r2": [0.5, 0.0, 0.0, 0.0]}
    batch = _batch()
    samples = [_sample("r1", GOOD_JSON), _sample("r2", BAD_JSON)]

    stub._overwrite(stub, samples, [{"grpo_batch": batch}])

    assert batch.advantages[0].tolist() == [1.0, 2.0, -1.0, 0.0]
    assert stub._graspo_advantages_injected is True
    assert stub._graspo_advantage_state == "injected_nonzero"


def _loss_stub(injected: bool, state: str) -> SimpleNamespace:
    """只带 ``_compute_loss_and_metrics`` 所需状态的最小对象。"""
    from graspo.flow.backend.msswift import trainer as trainer_module

    batch = _batch()
    batch.advantages = torch.ones(2, 4)  # 故意留成父类的非零标量优势
    stub = SimpleNamespace(
        _graspo=_core(),
        _graspo_advantages_injected=injected,
        _graspo_advantage_state=state,
        _graspo_injection_failures=0,
        _graspo_group_decisions=[],
        _graspo_token_advantages={},
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        model=SimpleNamespace(training=True),
        # 真训练里 log_probs 带梯度图 ⇒ loss 必须可反传（哪怕值被压成 0）
        _get_per_token_logps_and_entropies=lambda model, inputs, batch_, compute_entropy=False: (
            torch.full((2, 4), -0.5, requires_grad=True),
            None,
        ),
        # SimpleNamespace 不做描述符绑定 ⇒ 指标记录显式绑到 stub 上
        _record_ratio_metrics=lambda *args: (
            trainer_module.GraspoMsSwiftGRPOTrainer._record_ratio_metrics(stub, *args)
        ),
        _compute_loss=trainer_module.GraspoMsSwiftGRPOTrainer._compute_loss_and_metrics,
    )
    return stub, batch


def test_loss_is_exactly_zero_when_advantages_fell_back():
    """★ 回退时必须**显式跳过**这一步（loss = 0），不能静默地按父类优势训练。"""
    stub, batch = _loss_stub(injected=False, state="fallback")

    loss, _metrics = stub._compute_loss(stub, None, {}, batch)

    assert float(loss.detach()) == 0.0, f"回退路径的 loss 必须为 0，实测 {float(loss.detach())}"
    assert loss.requires_grad, "loss 必须仍可反传（零梯度），否则训练循环会崩"


def test_loss_is_not_forced_to_zero_when_graspo_injected():
    """阳性对照：graspo 掌控优势时，loss 不应被强行清零。"""
    stub, batch = _loss_stub(injected=True, state="injected_nonzero")

    loss, _metrics = stub._compute_loss(stub, None, {}, batch)

    assert float(loss.detach()) != 0.0, "graspo 注入的优势不为 0 ⇒ loss 不应为 0"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 指标：三种状态一眼可分，回退次数为 0
# ══════════════════════════════════════════════════════════════════════════════


def test_metrics_distinguish_injected_nonzero_injected_zero_and_fallback():
    """★ 指标必须能区分「注入且非零 / 注入但全 0 / 回退」三种情形。"""
    for state, expected in [
        ("injected_nonzero", (1.0, 0.0, 0.0, 1.0)),
        ("injected_zero", (0.0, 1.0, 0.0, 1.0)),
        ("fallback", (0.0, 0.0, 1.0, 0.0)),
    ]:
        stub, batch = _loss_stub(injected=state != "fallback", state=state)
        stub._compute_loss(stub, None, {}, batch)
        metrics = stub._metrics["train"]
        actual = tuple(
            metrics[name][-1]
            for name in (
                "graspo/advantage_state_injected_nonzero",
                "graspo/advantage_state_injected_zero",
                "graspo/advantage_state_fallback",
                "graspo/advantages_injected",
            )
        )
        assert actual == expected, f"{state} 的指标应为 {expected}，实测 {actual}"
