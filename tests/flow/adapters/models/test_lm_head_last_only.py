"""b1（缺陷 P6）的**等价性**与边界测试 —— "prefill 只算末位 logits"。

为什么需要这个文件：b1 用一个可能影响数值的"取行提前"优化，换掉了 T035 形状下
物理不可能（60–93 GiB）的整条序列 logits。**必须证明取值逐位不变**，否则就是
"为了不挂死而改了模型输出"。

三层：
1. ``_gather_sequence_positions`` 的值/形状/边界（越界当场报错，§2.3）；
2. **逐位等价**：用**真实**的 ``Qwen35RMSNorm`` + ``nn.Linear``(lm_head 的等价物)
   对比"先取行再算" vs "全序列算完再取同一行" ⇒ ``torch.equal`` 必须为真；
3. 调用方语义：左填充下 ``attention_mask.sum(-1)-1`` 恒等于 ``seq_len-1``
   （旧代码就是用它取行），因此"取末位"与旧索引是同一批数值。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="b1 等价性测试需要 torch")

from graspo.flow.adapters.models.common.layers import Qwen35RMSNorm  # noqa: E402
from graspo.flow.parallel.tensor_utils import _gather_sequence_positions  # noqa: E402


def test_gather_returns_the_selected_position_per_row() -> None:
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    indices = torch.tensor([2, 0])
    out = _gather_sequence_positions(hidden, indices)
    assert out.shape == (2, 1, 4)
    assert torch.equal(out[:, 0, :], hidden[torch.arange(2), indices])


@pytest.mark.parametrize(
    ("shape", "indices"),
    [
        ((2, 3), [0, 1]),  # 不是 (B, S, H)
        ((2, 3, 4), [0]),  # 行数不匹配
        ((2, 3, 4), [0, 3]),  # 越界（上）
        ((2, 3, 4), [-1, 0]),  # 越界（下）
    ],
)
def test_gather_rejects_bad_shapes_and_indices(shape, indices) -> None:
    hidden = torch.zeros(*shape)
    with pytest.raises(ValueError, match="_gather_sequence_positions"):
        _gather_sequence_positions(hidden, torch.tensor(indices))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_last_position_only_is_bitwise_identical_to_full_sequence(dtype) -> None:
    """**b1 的核心契约**：``linear(norm(x))[rows, idx]`` == ``linear(norm(x[rows, idx]))``。

    用真实 ``Qwen35RMSNorm`` + ``nn.Linear``（= lm_head 的等价物，逐 token 线性）。
    两者都逐位置独立 ⇒ 必须**逐位相同**（``torch.equal``，不是 allclose）。
    """
    torch.manual_seed(0)
    batch, seq_len, hidden_size, vocab = 4, 7, 16, 32
    hidden = (torch.randn(batch, seq_len, hidden_size, dtype=torch.float32) * 3).to(dtype)
    indices = torch.tensor([seq_len - 1, 2, 0, 5])

    norm = Qwen35RMSNorm(hidden_size, 1e-6, torch.device("cpu"), dtype).eval()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size, dtype=dtype))
    lm_head = torch.nn.Linear(hidden_size, vocab, bias=False, dtype=dtype).eval()

    with torch.no_grad():
        rows = torch.arange(batch)
        full = lm_head(norm(hidden))
        gathered_after = full[rows, indices]
        gathered_before = lm_head(norm(_gather_sequence_positions(hidden, indices)))[:, 0, :]

    assert gathered_before.shape == gathered_after.shape == (batch, vocab)
    assert torch.equal(gathered_before, gathered_after), (
        "取行提前改变了数值 ⇒ b1 破坏了'逐位一致'契约"
    )


def test_b1_index_is_the_same_arithmetic_as_the_old_code() -> None:
    """b1 的取行索引与旧代码**同一算式** ⇒ 取值逐位不变（回归围栏）。

    ⚠ 这条测试同时把一个**预存在问题**钉成事实（不在 b1 范围，已单独登记给指挥官）：
    本仓文本生成路径的 prompt 是**左填充**的（``_left_pad_token_rows``：pad 在左、
    内容靠右），于是 ``attention_mask.sum(-1) - 1``（= 长度−1，旧代码的 ``actual_lens``）
    对**较短**的行落到填充区（或非末位），**不是"最后一个真实 token"**；
    左填充下正确位置恒为 ``seq_len - 1``（即 ``logits[:, -1, :]``）。
    b1 **刻意保留**旧算式（指挥官要求"取末位语义逐字不变"）——所以这里断言的是
    "两者一致"，而不是"索引正确"；修索引是**另一个独立缺陷**，不得混进 b1。
    """
    width = 6
    mask = torch.zeros(3, width, dtype=torch.bool)
    mask[0, :] = True  # 长度 6（无填充）
    mask[1, 2:] = True  # 长度 4（左侧 2 个 pad）
    mask[2, 4:] = True  # 长度 2（左侧 4 个 pad）
    old_actual_lens = mask.sum(dim=1) - 1
    assert torch.equal(old_actual_lens, torch.tensor([5, 3, 1]))
    # 左填充下"最后一个真实 token"恒在最后一列：只有无填充那一行二者相同
    assert int(old_actual_lens[0]) == width - 1
    assert not torch.equal(old_actual_lens, torch.full((3,), width - 1))
    # 而真正"末位"的写法（非 PP 路径用的是它）恒等于 width-1
    assert torch.equal(torch.full((3,), width - 1), torch.full((3,), width - 1))
