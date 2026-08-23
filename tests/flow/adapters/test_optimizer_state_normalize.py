"""Tests for ``_normalize_optimizer_state_to_params`` — 状态 device/dtype 防御性归一。

每个 DP rank 现在 load 自己的 shard（map_location=self.device），optimizer state 应已在
本地 device。保留该归一化作为边界守卫（§2.3）：不同 torch 版本的
``optimizer.load_state_dict`` 搬迁行为不一致（现代 torch 会迁 exp_avg/exp_avg_sq，
但非 fused 的 step 不一定），任何残留的 device/dtype 不一致都会让 ``AdamW.step`` 抛
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
