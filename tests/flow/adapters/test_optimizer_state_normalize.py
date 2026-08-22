"""Tests for ``_normalize_optimizer_state_to_params`` — DP resume 的状态 device/dtype 归一。

回归：DP>1 时 checkpoint 由 dp_rank=0 经 ``broadcast_object_list`` 广播，接收 rank
拿到的是 sender device（cuda:0）上的 optimizer state；``optimizer.load_state_dict``
不会把它们迁到本 rank device，``AdamW.step`` 随后抛
"Tensors of the same index must be on the same device and the same dtype"。
修复：加载后把所有 state tensor 统一迁回参数所在 device/dtype（``step`` 保持 float32）。
"""

import torch

from graspo.flow.adapters.transformer_adapter import _normalize_optimizer_state_to_params


def _param(dtype: torch.dtype) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(4, dtype=dtype), requires_grad=True)


def test_mismatched_dtype_is_normalized():
    """exp_avg/exp_avg_sq 与参数 dtype 不一致 → 归一为参数 dtype，并返回记录。"""
    param = _param(torch.bfloat16)
    optimizer = torch.optim.AdamW([param])
    # 模拟广播回来的 state：bf16 参数却挂着 fp32 的 exp_avg（不同 dtype）
    optimizer.state[param] = {
        "step": torch.tensor(0.0, dtype=torch.float32),
        "exp_avg": torch.zeros_like(param, dtype=torch.float32),
        "exp_avg_sq": torch.zeros_like(param, dtype=torch.float32),
    }

    normalized = _normalize_optimizer_state_to_params(optimizer)

    state = optimizer.state[param]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert state["step"].dtype == torch.float32
    assert normalized


def test_matched_state_is_noop():
    """device/dtype 一致 → 不动，返回空。"""
    param = _param(torch.float32)
    optimizer = torch.optim.AdamW([param])
    optimizer.state[param] = {
        "step": torch.tensor(0.0, dtype=torch.float32),
        "exp_avg": torch.zeros_like(param),
        "exp_avg_sq": torch.zeros_like(param),
    }

    normalized = _normalize_optimizer_state_to_params(optimizer)

    assert normalized == []
    assert optimizer.state[param]["exp_avg"].dtype == torch.float32


def test_multiple_params_all_normalized():
    """多个参数各自归一，不互相污染。"""
    p1 = _param(torch.bfloat16)
    p2 = _param(torch.float32)
    optimizer = torch.optim.AdamW([p1, p2])
    optimizer.state[p1] = {
        "step": torch.tensor(1.0, dtype=torch.float32),
        "exp_avg": torch.zeros_like(p1, dtype=torch.float32),
        "exp_avg_sq": torch.zeros_like(p1, dtype=torch.float32),
    }
    optimizer.state[p2] = {
        "step": torch.tensor(1.0, dtype=torch.float32),
        "exp_avg": torch.zeros_like(p2),
        "exp_avg_sq": torch.zeros_like(p2),
    }

    _normalize_optimizer_state_to_params(optimizer)

    assert optimizer.state[p1]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[p2]["exp_avg"].dtype == torch.float32


def test_none_optimizer_safe():
    """optimizer 为 None → 安全返回空。"""
    assert _normalize_optimizer_state_to_params(None) == []
