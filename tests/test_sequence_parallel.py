"""Sequence Parallel 通信原语单元测试 — CPU 上验证 autograd 正确性。"""

import torch
import torch.distributed as dist

from graspo.flow.parallel.tensor_utils import (
    _all_gather_sp,
    _all_reduce_tp,
    _reduce_scatter_sp,
    _set_tensor_parallel_group,
)


class TestSPPrimitives:
    """SP primitives 在单进程下的行为（TP=1 时退化为 identity）。"""

    def test_reduce_scatter_sp_identity_when_tp1(self):
        """TP=1 时 _reduce_scatter_sp 应该直接返回输入。"""
        _set_tensor_parallel_group(None, 1)
        x = torch.randn(2, 16, 8, requires_grad=True)
        y = _reduce_scatter_sp(x)
        assert y.shape == x.shape
        assert torch.equal(y, x)
        # backward 也应正确
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_all_gather_sp_identity_when_tp1(self):
        """TP=1 时 _all_gather_sp 应该直接返回输入。"""
        _set_tensor_parallel_group(None, 1)
        x = torch.randn(2, 16, 8, requires_grad=True)
        y = _all_gather_sp(x)
        assert y.shape == x.shape
        assert torch.equal(y, x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_all_reduce_tp_identity_when_tp1(self):
        """TP=1 时 _all_reduce_tp 应该直接返回输入。"""
        _set_tensor_parallel_group(None, 1)
        x = torch.randn(2, 16, 8, requires_grad=True)
        y = _all_reduce_tp(x)
        assert y.shape == x.shape
        assert torch.equal(y, x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_sp_ops_differentiable(self):
        """SP autograd 函数在 TP=1 时应正确传递梯度。"""
        _set_tensor_parallel_group(None, 1)
        x = torch.randn(2, 16, 8, requires_grad=True)
        # reduce_scatter → all_gather 应该恢复原始张量
        scattered = _reduce_scatter_sp(x)
        gathered = _all_gather_sp(scattered)
        assert gathered.shape == x.shape
        loss = gathered.sum()
        loss.backward()
        assert x.grad is not None
        # 梯度应该全为 1（因为 sum backward）
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


class TestSPNumericalEquivalence:
    """验证 SP 的 logits 数值与 TP 一致（相同 seed 下的 loss 对比）。"""

    def test_sp_forward_backward_numerical(self):
        """简单的线性层 + SP 通信：验证 forward/backward 数值正确。"""
        _set_tensor_parallel_group(None, 1)
        torch.manual_seed(42)

        # 模拟一个 column-parallel 线性层
        hidden = 8
        batch, seq = 2, 16
        weight = torch.randn(hidden, hidden, requires_grad=True)

        x = torch.randn(batch, seq, hidden, requires_grad=True)

        # TP 路径（无 SP）
        out_tp = _all_reduce_tp(x @ weight.t())
        loss_tp = out_tp.sum()
        loss_tp.backward()
        grad_tp = x.grad.clone()

        x.grad = None
        weight.grad = None

        # SP 路径
        x2 = torch.randn(batch, seq, hidden, requires_grad=True)
        # 先让 x2 与 x 相同
        x2.data.copy_(x.data)

        # 模拟 SP：输入先分片
        scattered = _reduce_scatter_sp(x2)
        gathered = _all_gather_sp(scattered)
        out_sp = _all_reduce_tp(gathered @ weight.t())
        loss_sp = out_sp.sum()
        loss_sp.backward()

        # 在 TP=1 时，SP 路径应该与 TP 路径数值一致
        assert torch.allclose(out_tp, out_sp, atol=1e-6)
        assert torch.allclose(x2.grad, grad_tp, atol=1e-6)
