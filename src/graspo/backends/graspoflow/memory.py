"""Layer 1 — Memory budget: compute max in-flight microbatches from available VRAM.

The memory budget is the bridge between Flink-style backpressure and GPU
resource limits: it converts "how much free memory do I have?" into "how many
microbatches can I safely keep in-flight at once?"

All numbers are estimates (not exact) — we always apply a safety factor.
"""

from __future__ import annotations


def estimate_per_microbatch_activation_bytes(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype_size: int = 2,  # bf16 / fp16 = 2 bytes
    gradient_checkpointing: bool = True,
) -> int:
    """Estimate activation memory per in-flight microbatch for one PP stage.

    An in-flight microbatch holds:
      - stage_output:  B × S × D × dtype    (forward output, saved for backward)
      - stage_input:   B × S × D × dtype    (input hidden states, requires_grad)

    With gradient checkpointing enabled, PyTorch also saves a small set of
    intermediates (input tensors at checkpoint boundaries) which we approximate
    as an additional 0.5× of the base activation.

    Returns bytes.
    """
    base = batch_size * seq_len * hidden_size * dtype_size

    # stage_output + stage_input
    pp_buffers = base * 2

    # gradient checkpointing overhead (saved checkpoint inputs)
    ckpt_overhead = base // 2 if gradient_checkpointing else base * 2

    return pp_buffers + ckpt_overhead


def compute_max_inflight(
    *,
    gpu_memory_free_bytes: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int = 2,
    gradient_checkpointing: bool = True,
    safety_factor: float = 0.8,
) -> int:
    """Compute the maximum number of in-flight microbatches.

    Args:
        gpu_memory_free_bytes: Free GPU memory in bytes (e.g. from
            ``torch.cuda.mem_get_info()``).
        batch_size: Microbatch batch size (sequences per microbatch).
        seq_len: Maximum sequence length.
        hidden_size: Model hidden dimension.
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32).
        gradient_checkpointing: Whether gradient checkpointing is enabled.
        safety_factor: Fraction of free memory we're willing to use (0.0-1.0).

    Returns:
        Integer >= 1, the maximum number of microbatches that can be
        in-flight simultaneously without risking OOM.
    """
    per_mb = estimate_per_microbatch_activation_bytes(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        dtype_size=dtype_bytes,
        gradient_checkpointing=gradient_checkpointing,
    )
    if per_mb <= 0:
        return 1

    available = int(gpu_memory_free_bytes * safety_factor)
    max_inflight = max(1, available // per_mb)
    return max_inflight


def get_gpu_free_memory_bytes(device: int | None = None) -> int:
    """Query free GPU memory via PyTorch.

    Args:
        device: CUDA device index (default: current device).

    Returns:
        Free memory in bytes, or 0 if CUDA is unavailable.
    """
    try:
        import torch  # lazy import — only needed at call time

        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    except (ImportError, RuntimeError, AssertionError):
        return 0
