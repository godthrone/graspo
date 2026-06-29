"""Tests for ReplayBuffer — BADGE §11.1."""

from graspo.core.buffer import Experience, ReplayBuffer


def _make_experience(seq_id: int = 0):
    return Experience(
        sequences=None,
        old_log_probs=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        rewards=None,
        metadata={"seq": seq_id},
    )


# ── Basic operations ─────────────────────────────────────────────────────────


def test_replay_buffer_starts_empty():
    buf = ReplayBuffer()
    assert len(buf) == 0


def test_replay_buffer_append_many_increases_length():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(i) for i in range(5)])
    assert len(buf) == 5


def test_replay_buffer_append_many_multiple_times():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(0), _make_experience(1)])
    buf.append_many([_make_experience(2), _make_experience(3), _make_experience(4)])
    assert len(buf) == 5


def test_replay_buffer_take_returns_requested_count():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(i) for i in range(10)])
    taken = buf.take(3)
    assert len(taken) == 3
    assert taken[0].metadata["seq"] == 0
    assert taken[2].metadata["seq"] == 2


def test_replay_buffer_take_does_not_remove():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(i) for i in range(5)])
    _ = buf.take(3)
    assert len(buf) == 5


def test_replay_buffer_take_more_than_available_returns_all():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(0)])
    taken = buf.take(10)
    assert len(taken) == 1


def test_replay_buffer_take_from_empty_returns_empty():
    buf = ReplayBuffer()
    assert buf.take(5) == []


def test_replay_buffer_clear_empties_buffer():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(i) for i in range(10)])
    buf.clear()
    assert len(buf) == 0


# ── Limit constraint ─────────────────────────────────────────────────────────


def test_replay_buffer_limit_zero_means_unlimited():
    buf = ReplayBuffer(limit=0)
    buf.append_many([_make_experience(i) for i in range(100)])
    assert len(buf) == 100


def test_replay_buffer_limit_enforced_on_append():
    buf = ReplayBuffer(limit=5)
    buf.append_many([_make_experience(i) for i in range(10)])
    assert len(buf) == 5
    # Should keep the most recent 5 (last 5)
    assert buf[0].metadata["seq"] == 5
    assert buf[4].metadata["seq"] == 9


def test_replay_buffer_limit_enforced_across_multiple_appends():
    buf = ReplayBuffer(limit=3)
    buf.append_many([_make_experience(0), _make_experience(1)])
    buf.append_many([_make_experience(2), _make_experience(3)])
    assert len(buf) == 3
    assert buf[0].metadata["seq"] == 1
    assert buf[2].metadata["seq"] == 3


def test_replay_buffer_limit_exact_boundary():
    buf = ReplayBuffer(limit=5)
    buf.append_many([_make_experience(i) for i in range(5)])
    assert len(buf) == 5


# ── Index access ─────────────────────────────────────────────────────────────


def test_replay_buffer_index_access():
    buf = ReplayBuffer()
    items = [_make_experience(i) for i in range(3)]
    buf.append_many(items)
    assert buf[0] is items[0]
    assert buf[1] is items[1]
    assert buf[2] is items[2]


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_replay_buffer_append_empty_list_no_change():
    buf = ReplayBuffer()
    buf.append_many([_make_experience(0)])
    buf.append_many([])
    assert len(buf) == 1


def test_replay_buffer_clear_then_reuse():
    buf = ReplayBuffer(limit=3)
    buf.append_many([_make_experience(i) for i in range(5)])
    buf.clear()
    buf.append_many([_make_experience(10), _make_experience(11)])
    assert len(buf) == 2
    assert buf[0].metadata["seq"] == 10
