"""Unit tests for graspoflow Layer 1 — operator, schedule, memory."""

from __future__ import annotations

import math

import pytest

from graspo.backends.graspoflow.memory import (
    compute_max_inflight,
    estimate_per_microbatch_activation_bytes,
)
from graspo.backends.graspoflow.operator import (
    ComputeOperator,
    Microbatch,
    OpBuffer,
    OpMemoryProfile,
)
from graspo.backends.graspoflow.schedule import (
    GPipeScheduler,
    OneFOneBScheduler,
    PipelineAction,
    get_scheduler,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Microbatch tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMicrobatch:
    def test_basic_construction(self):
        mb = Microbatch(idx=0, input_ids=None, attention_mask=None)
        assert mb.idx == 0
        assert mb.batch_size == 0
        assert mb.seq_len == 0

    def test_batch_and_seq_from_input_ids(self):
        import torch

        ids = torch.zeros(2, 512, dtype=torch.long)
        mb = Microbatch(idx=0, input_ids=ids, attention_mask=ids.ne(0))
        assert mb.batch_size == 2
        assert mb.seq_len == 512

    def test_batch_and_seq_from_hidden(self):
        import torch

        hidden = torch.zeros(4, 256, 2048, dtype=torch.bfloat16)
        mb = Microbatch(idx=3, hidden_states=hidden)
        assert mb.batch_size == 4
        assert mb.seq_len == 256

    def test_clone_for_retry_preserves_inputs(self):
        import torch

        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mask = torch.tensor([[True, True, True]])
        mb = Microbatch(
            idx=5,
            input_ids=ids,
            attention_mask=mask,
            old_log_probs=torch.tensor([0.5]),
        )
        clone = mb.clone_for_retry(idx=99)
        assert clone.idx == 99
        assert clone.input_ids is ids  # shallow
        assert clone.old_log_probs is mb.old_log_probs
        assert clone.hidden_states is None  # not copied
        assert clone._stage_input is None  # internal fields reset


# ═══════════════════════════════════════════════════════════════════════════════
# OpBuffer tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpBuffer:
    def test_initial_state(self):
        buf = OpBuffer(max_slots=8, name="test")
        assert not buf.is_full
        assert buf.is_empty
        assert buf.size == 0
        assert buf.waterlevel == 0.0
        assert repr(buf) == "OpBuffer('test' 0/8)"

    def test_push_pop_fifo(self):
        buf = OpBuffer(max_slots=3)
        assert buf.push(Microbatch(idx=0))
        assert buf.push(Microbatch(idx=1))
        assert buf.push(Microbatch(idx=2))
        assert buf.is_full
        assert buf.waterlevel == 1.0

        # Push to full buffer → backpressure
        assert not buf.push(Microbatch(idx=3))
        assert buf.size == 3  # unchanged

        # Pop in order
        assert buf.pop().idx == 0
        assert buf.pop().idx == 1
        assert buf.size == 1
        assert not buf.is_empty

        assert buf.pop().idx == 2
        assert buf.is_empty
        assert buf.pop() is None

    def test_peek(self):
        buf = OpBuffer(max_slots=2)
        buf.push(Microbatch(idx=10))
        assert buf.peek().idx == 10
        assert buf.size == 1  # peek doesn't remove

    def test_clear(self):
        buf = OpBuffer(max_slots=5)
        for i in range(3):
            buf.push(Microbatch(idx=i))
        buf.clear()
        assert buf.is_empty

    def test_waterlevel(self):
        buf = OpBuffer(max_slots=4)
        assert buf.waterlevel == 0.0
        buf.push(Microbatch(idx=0))
        assert buf.waterlevel == 0.25
        buf.push(Microbatch(idx=1))
        assert buf.waterlevel == 0.5
        buf.push(Microbatch(idx=2))
        assert buf.waterlevel == 0.75
        buf.push(Microbatch(idx=3))
        assert buf.waterlevel == 1.0

    def test_max_slots_must_be_positive(self):
        with pytest.raises(ValueError):
            OpBuffer(max_slots=0)


# ═══════════════════════════════════════════════════════════════════════════════
# ComputeOperator base class tests
# ═══════════════════════════════════════════════════════════════════════════════


class _MockOp(ComputeOperator):
    """Minimal concrete operator for testing the base class."""

    def __init__(self, name: str = "mock", tp_size: int = 1):
        super().__init__(name=name, tp_size=tp_size)

    def forward(self, mb: Microbatch) -> Microbatch:
        mb.hidden_states = None  # mock: consume and return
        return mb

    def backward(self, mb: Microbatch) -> Microbatch:
        return mb

    @property
    def memory_profile(self) -> OpMemoryProfile:
        return OpMemoryProfile(forward_activation_bytes=100)

    def trainable_parameters(self) -> list:
        return []


class TestComputeOperator:
    def test_basic_construction(self):
        op = _MockOp(name="stage_0", tp_size=2)
        assert op.name == "stage_0"
        assert op.tp_size == 2
        assert op.input_buffer is None
        assert op.output_buffer is None

    def test_attach_buffers(self):
        op = _MockOp()
        in_buf = OpBuffer(max_slots=4, name="in")
        out_buf = OpBuffer(max_slots=4, name="out")
        op.attach_buffers(in_buf, out_buf)
        assert op.input_buffer is in_buf
        assert op.output_buffer is out_buf

    def test_attach_buffers_none_output(self):
        op = _MockOp()
        op.attach_buffers(OpBuffer(max_slots=2), None)
        assert op.output_buffer is None  # terminal op has no output

    def test_done_clears_buffers(self):
        op = _MockOp()
        in_buf = OpBuffer(max_slots=4)
        out_buf = OpBuffer(max_slots=4)
        in_buf.push(Microbatch(idx=0))
        out_buf.push(Microbatch(idx=1))
        op.attach_buffers(in_buf, out_buf)
        op.done()
        assert in_buf.is_empty
        assert out_buf.is_empty

    def test_forward_returns_microbatch(self):
        op = _MockOp()
        mb = Microbatch(idx=0)
        result = op.forward(mb)
        assert isinstance(result, Microbatch)

    def test_repr(self):
        op = _MockOp(name="decoder", tp_size=4)
        assert "decoder" in repr(op)
        assert "4" in repr(op)


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOneFOneBScheduler:
    def test_stage_0_warmup(self):
        s = OneFOneBScheduler(pp_size=8, pp_rank=0)
        assert s._warmup_count(chunk_count=96) == 7  # pp_size - pp_rank - 1

    def test_stage_7_no_warmup(self):
        s = OneFOneBScheduler(pp_size=8, pp_rank=7)
        assert s._warmup_count(chunk_count=96) == 0

    def test_warmup_capped_by_chunk_count(self):
        s = OneFOneBScheduler(pp_size=8, pp_rank=0)
        assert s._warmup_count(chunk_count=2) == 2  # capped at chunk_count
        assert s._warmup_count(chunk_count=7) == 7

    def test_warmup_capped_by_max_inflight(self):
        s = OneFOneBScheduler(pp_size=8, pp_rank=0)
        assert s._warmup_count(chunk_count=96, max_inflight=3) == 3

    def test_plan_every_microbatch_gets_one_fwd_one_bwd(self):
        s = OneFOneBScheduler(pp_size=8, pp_rank=2)
        plan = s.plan(chunk_count=32)
        fwd_counts: dict[int, int] = {}
        bwd_counts: dict[int, int] = {}
        for step in plan:
            if step.action == PipelineAction.FORWARD:
                fwd_counts[step.microbatch_idx] = fwd_counts.get(step.microbatch_idx, 0) + 1
            else:
                bwd_counts[step.microbatch_idx] = bwd_counts.get(step.microbatch_idx, 0) + 1
        for i in range(32):
            assert fwd_counts.get(i) == 1, f"microbatch {i}: {fwd_counts.get(i, 0)} forwards"
            assert bwd_counts.get(i) == 1, f"microbatch {i}: {bwd_counts.get(i, 0)} backwards"

    def test_plan_forward_before_backward(self):
        s = OneFOneBScheduler(pp_size=4, pp_rank=0)
        plan = s.plan(chunk_count=10)
        for step in plan:
            if step.action == PipelineAction.BACKWARD:
                # Check that this microbatch was forwarded earlier
                fwd_seen = any(
                    s2.action == PipelineAction.FORWARD
                    and s2.microbatch_idx == step.microbatch_idx
                    for s2 in plan[: plan.index(step)]
                )
                assert fwd_seen, f"backward {step.microbatch_idx} without forward"

    def test_plan_phases_keys(self):
        s = OneFOneBScheduler(pp_size=4, pp_rank=0)
        phases = s.plan_phases(chunk_count=12)
        assert set(phases.keys()) == {"fill", "steady", "drain"}
        # fill + steady + drain == total fwd+bwd
        total_phase_steps = sum(len(v) for v in phases.values())
        total_plan_steps = len(s.plan(chunk_count=12))
        assert total_phase_steps == total_plan_steps

    def test_steady_phase_interleave(self):
        s = OneFOneBScheduler(pp_size=4, pp_rank=0)  # warmup=3
        phases = s.plan_phases(chunk_count=10)
        steady = phases["steady"]
        # Steady alternates: fwd(warmup+0), bwd(0), fwd(warmup+1), bwd(1), ...
        for i in range(0, len(steady), 2):
            assert steady[i].action == PipelineAction.FORWARD
            assert steady[i + 1].action == PipelineAction.BACKWARD


class TestGPipeScheduler:
    def test_all_forwards_then_all_backwards(self):
        s = GPipeScheduler(pp_size=4, pp_rank=0)
        plan = s.plan(chunk_count=5)
        actions = [step.action for step in plan]
        assert actions == [
            PipelineAction.FORWARD,
            PipelineAction.FORWARD,
            PipelineAction.FORWARD,
            PipelineAction.FORWARD,
            PipelineAction.FORWARD,
            PipelineAction.BACKWARD,
            PipelineAction.BACKWARD,
            PipelineAction.BACKWARD,
            PipelineAction.BACKWARD,
            PipelineAction.BACKWARD,
        ]

    def test_backward_reverse_order(self):
        s = GPipeScheduler(pp_size=4, pp_rank=0)
        plan = s.plan(chunk_count=5)
        bwd_indices = [s.microbatch_idx for s in plan if s.action == PipelineAction.BACKWARD]
        assert bwd_indices == [4, 3, 2, 1, 0]  # reverse

    def test_plan_phases(self):
        s = GPipeScheduler(pp_size=4, pp_rank=0)
        phases = s.plan_phases(chunk_count=5)
        assert set(phases.keys()) == {"forward", "backward"}


class TestSchedulerRegistry:
    def test_get_known_schedulers(self):
        for name in ["simple", "gpipe", "one_f_one_b", "1f1b", "async_1f1b"]:
            s = get_scheduler(name, pp_size=4, pp_rank=1)
            assert isinstance(s, (GPipeScheduler, OneFOneBScheduler))

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline scheduler"):
            get_scheduler("nonexistent", pp_size=4, pp_rank=0)

    def test_case_insensitive(self):
        s1 = get_scheduler("ONE_F_ONE_B", pp_size=4, pp_rank=0)
        s2 = get_scheduler("one_f_one_b", pp_size=4, pp_rank=0)
        assert type(s1) is type(s2)


class TestPipelineSchedulerValidation:
    def test_pp_size_must_be_positive(self):
        with pytest.raises(ValueError):
            OneFOneBScheduler(pp_size=0, pp_rank=0)

    def test_pp_rank_in_range(self):
        with pytest.raises(ValueError):
            OneFOneBScheduler(pp_size=4, pp_rank=4)

    def test_pp_rank_negative(self):
        with pytest.raises(ValueError):
            OneFOneBScheduler(pp_size=4, pp_rank=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Memory budget tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryBudget:
    def test_estimate_basic(self):
        # B=1, S=2048, D=2048, bf16
        per_mb = estimate_per_microbatch_activation_bytes(
            batch_size=1,
            seq_len=2048,
            hidden_size=2048,
            dtype_size=2,
            gradient_checkpointing=True,
        )
        # Base: 1*2048*2048*2 = 8,388,608
        # PP buffers: 2 * base = 16,777,216
        # CKPT overhead: base/2 = 4,194,304
        # Total: ~20,971,520 = 20 MB
        assert 15_000_000 < per_mb < 25_000_000, f"Expected ~20MB, got {per_mb}"
        # Sanity: should be roughly 20 MB
        assert math.isclose(per_mb / (1024 * 1024), 20, abs_tol=2)

    def test_estimate_no_checkpointing(self):
        per_mb = estimate_per_microbatch_activation_bytes(
            batch_size=1,
            seq_len=2048,
            hidden_size=2048,
            dtype_size=2,
            gradient_checkpointing=False,
        )
        # No ckpt: 2*base + 2*base = 4*base = 33,554,432 = 32 MB
        assert 30_000_000 < per_mb < 40_000_000

    def test_compute_max_inflight(self):
        max_inf = compute_max_inflight(
            gpu_memory_free_bytes=50 * 1024**3,  # 50 GB
            batch_size=1,
            seq_len=2048,
            hidden_size=2048,
            dtype_bytes=2,
            gradient_checkpointing=True,
            safety_factor=0.8,
        )
        # 50GB * 0.8 / 20MB ≈ 2000+ microbatch slots
        assert max_inf > 1000, f"Expected >1000, got {max_inf}"

    def test_compute_max_inflight_small_memory(self):
        max_inf = compute_max_inflight(
            gpu_memory_free_bytes=100 * 1024 * 1024,  # 100 MB only
            batch_size=4,
            seq_len=4096,
            hidden_size=4096,
            dtype_bytes=2,
            gradient_checkpointing=True,
            safety_factor=0.8,
        )
        # Should always return at least 1
        assert max_inf >= 1

    def test_compute_max_inflight_chunk_constrained(self):
        # Even with 50GB free, for small seq/microbatch, the constraint
        # is chunk_count, not memory
        max_inf = compute_max_inflight(
            gpu_memory_free_bytes=50 * 1024**3,
            batch_size=1,
            seq_len=128,
            hidden_size=512,
            dtype_bytes=2,
            gradient_checkpointing=True,
            safety_factor=0.8,
        )
        # Tiny activation → huge max_inflight → bounded by chunk_count in practice
        assert max_inf > 10000


# ═══════════════════════════════════════════════════════════════════════════════
# OpMemoryProfile tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpMemoryProfile:
    def test_total(self):
        p = OpMemoryProfile(
            forward_activation_bytes=1000,
            backward_intermediate_bytes=500,
            gradient_bytes=200,
        )
        assert p.total_per_microbatch == 1700

    def test_defaults_zero(self):
        p = OpMemoryProfile()
        assert p.total_per_microbatch == 0
