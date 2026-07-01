from graspo.core.graspo_parity import (
    GroupDecision,
    classify_group,
    group_advantages,
    is_invalid_group,
    is_uniform_partial_content,
    lower_median,
    replay_buffer_optimize_threshold,
    replay_ready,
)


def test_lower_median_matches_original_even_group_rule():
    assert lower_median([0.0, 0.2, 0.4, 1.0]) == 0.2


def test_first_group_perfect_skip_uses_lower_median():
    decision = classify_group(
        [1.0, 1.0, 1.0, 0.5],
        [1.0, 1.0, 1.0, 0.5],
        retry_count=0,
        rollout_max_retries=5,
    )

    assert decision.decision == GroupDecision.PERFECT_SKIP
    assert not decision.should_train


def test_retry_continues_until_rollout_max_retry_or_max_reward_reaches_threshold():
    retry = classify_group(
        [0.2, 0.2],
        [0.2, 0.2],
        retry_count=0,
        rollout_max_retries=1,
    )
    exhausted = classify_group(
        [0.2, 0.2],
        [0.2, 0.2],
        retry_count=1,
        rollout_max_retries=1,
    )

    assert retry.decision == GroupDecision.RETRY
    assert exhausted.decision == GroupDecision.INVALID


def test_no_right_preference_gap_trains_before_retry():
    decision = classify_group(
        [0.1, 0.2],
        [0.1, 0.2],
        retry_count=0,
        rollout_max_retries=5,
    )

    assert decision.decision == GroupDecision.TRAINABLE_NOT_CORRECT
    assert decision.should_train


def test_no_right_group_requires_max_above_lower_median_to_train():
    no_gap = classify_group(
        [0.1, 0.2, 0.2, 0.2],
        [0.0, 0.8, 0.8, 0.8],
        retry_count=5,
        rollout_max_retries=5,
    )
    has_gap = classify_group(
        [0.1, 0.2, 0.2, 0.3],
        [0.0, 0.8, 0.8, 0.9],
        retry_count=5,
        rollout_max_retries=5,
    )

    assert no_gap.reward_max_median_gap == 0.0
    assert not is_invalid_group([0.1, 0.2, 0.2, 0.2], [0.0, 0.8, 0.8, 0.8])
    assert no_gap.decision == GroupDecision.INVALID_NO_PREFERENCE_GAP
    assert not no_gap.should_train
    assert has_gap.reward_max_median_gap > 0.0
    assert has_gap.decision == GroupDecision.TRAINABLE_NOT_CORRECT
    assert has_gap.should_train


def test_no_right_gap_takes_priority_over_uniform_partial_invalid_filter():
    decision = classify_group(
        [0.1, 0.2, 0.2, 0.3],
        [0.8, 0.8, 0.8, 0.8],
        retry_count=5,
        rollout_max_retries=5,
    )

    assert is_uniform_partial_content([0.8, 0.8, 0.8, 0.8])
    assert decision.decision == GroupDecision.TRAINABLE_NOT_CORRECT
    assert decision.should_train


def test_invalid_group_matches_original_filters():
    assert is_invalid_group([0.0, 0.0], [0.0, 0.0])
    assert is_invalid_group([0.2, 0.8], [0.5, 0.5])
    assert is_uniform_partial_content([0.5, 0.5, 0.5])
    assert not is_uniform_partial_content([0.0, 0.0, 0.0])
    assert not is_uniform_partial_content([1.0, 1.0, 1.0])


def test_invalid_takes_priority_over_no_preference_gap_after_retry_exhausted():
    decision = classify_group(
        [0.2, 0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5, 0.5],
        retry_count=5,
        rollout_max_retries=5,
    )

    assert is_invalid_group([0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5])
    assert decision.decision == GroupDecision.INVALID


def test_trainable_max_correct_after_retry_success():
    decision = classify_group(
        [0.1, 1.0],
        [0.1, 1.0],
        retry_count=1,
        rollout_max_retries=5,
    )

    assert decision.decision == GroupDecision.TRAINABLE_MAX_CORRECT
    assert decision.should_train


def test_perfect_priority_applies_before_max_correct_after_retry():
    perfect = classify_group(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        retry_count=1,
        rollout_max_retries=5,
    )
    max_correct = classify_group(
        [0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 1.0],
        retry_count=1,
        rollout_max_retries=5,
    )

    assert perfect.decision == GroupDecision.PERFECT_SKIP
    assert not perfect.should_train
    assert max_correct.decision == GroupDecision.TRAINABLE_MAX_CORRECT
    assert max_correct.should_train


def test_group_advantages_matches_original_sample_std_formula():
    actual = group_advantages([0.0, 1.0])

    assert round(actual[0], 6) == -0.707107
    assert round(actual[1], 6) == 0.707107


def test_reject_unparseable_groups_retries_before_exhausted():
    """Defense line: unparseable completions trigger retry, then discard on exhaustion."""
    retry = classify_group(
        [0.1, 0.2],
        [0.1, 0.2],
        retry_count=0,
        rollout_max_retries=5,
        best_completion_has_parse_error=True,
        reject_unparseable_groups=True,
    )
    assert retry.decision == GroupDecision.RETRY

    exhausted = classify_group(
        [0.1, 0.2],
        [0.1, 0.2],
        retry_count=5,
        rollout_max_retries=5,
        best_completion_has_parse_error=True,
        reject_unparseable_groups=True,
    )
    assert exhausted.decision == GroupDecision.INVALID


def test_reject_unparseable_groups_when_false_trains_despite_parse_error():
    """When defense line is off, parse errors don't trigger retry/invalid."""
    decision = classify_group(
        [0.1, 0.2],
        [0.1, 0.2],
        retry_count=5,
        rollout_max_retries=5,
        best_completion_has_parse_error=True,
        reject_unparseable_groups=False,
    )
    # Falls through to normal decision logic (not RETRY/INVALID from parse error)
    assert decision.decision not in (GroupDecision.RETRY, GroupDecision.INVALID)


def test_reject_unparseable_groups_defaults_to_true():
    """Defense line is active by default (reject invalid data at boundary)."""
    decision = classify_group(
        [0.1, 0.2],
        [0.1, 0.2],
        retry_count=0,
        rollout_max_retries=5,
        best_completion_has_parse_error=True,
    )
    assert decision.decision == GroupDecision.RETRY


def test_replay_buffer_optimize_threshold_uses_completion_batch_times_rollout_group():
    assert (
        replay_buffer_optimize_threshold(
            optimize_prompt_batch_size=4,
            rollout_group_size=8,
        )
        == 32
    )
    assert replay_ready(
        replay_size=32,
        optimize_prompt_batch_size=4,
        rollout_group_size=8,
    )
    assert not replay_ready(
        replay_size=31,
        optimize_prompt_batch_size=4,
        rollout_group_size=8,
    )
