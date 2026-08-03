"""runtime → adapter 调用约定测试（防 C8 类回归）。

C8 将 ABC 适配器方法改为 keyword-only 后，runtime 曾以位置参数调用
sequence_log_probs 导致 TypeError（mypy 未抓到：_require_adapter 返回
Any，Any 调用不检查参数形式）。本测试锁定 runtime 转发必须用关键字。
"""

from typing import Any

from graspo.flow.runtime import GraspoFlowRuntime


class _FakeAdapter:
    """记录收到的调用参数形式。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def train_batch(
        self,
        *,
        experiences: Any,
        optimizer_steps: int = 1,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
        **kwargs: Any,
    ) -> Any:
        # keyword-only 签名：位置调用会在进入函数前抛 TypeError
        self.calls.append(
            {"sequences": experiences, "policy_ratio_clip_eps": policy_ratio_clip_eps}
        )
        return "ok"

    def sequence_log_probs(
        self,
        *,
        sequences: Any,
        attention_mask: Any = None,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        # keyword-only 签名：位置调用会在进入函数前抛 TypeError
        self.calls.append(
            {"sequences": sequences, "attention_mask": attention_mask, "metadata": metadata}
        )
        return "ok"


class _FakeRuntime(GraspoFlowRuntime):
    def _require_adapter(self):
        return self._fake_adapter


def test_runtime_forwards_sequence_log_probs_by_keyword():
    adapter = _FakeAdapter()
    runtime = _FakeRuntime.__new__(_FakeRuntime)  # type: ignore[attr-defined]
    runtime._fake_adapter = adapter
    result = runtime.sequence_log_probs("seq", "mask", metadata={"m": 1})
    assert result == "ok"
    assert adapter.calls == [{"sequences": "seq", "attention_mask": "mask", "metadata": {"m": 1}}]


def test_runtime_forwards_train_batch_by_keyword():
    adapter = _FakeAdapter()
    runtime = _FakeRuntime.__new__(_FakeRuntime)  # type: ignore[attr-defined]
    runtime._fake_adapter = adapter
    runtime.train_batch("exp", policy_ratio_clip_eps=0.2, max_grad_norm=1.0)
    assert adapter.calls[-1]["sequences"] == "exp"
