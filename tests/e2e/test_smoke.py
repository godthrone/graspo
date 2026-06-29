"""End-to-end smoke test — BADGE §11.1 (CPU-only pipeline validation)."""

from pathlib import Path

from graspo.core.completion import ParsedCompletion
from graspo.core.data import load_jsonl
from graspo.core.graspo_parity import (
    GroupDecision,
    classify_group,
    group_advantages,
    has_reward_variance,
    replay_ready,
)
from graspo.core.reward import GraspoReward, RewardConfig
from graspo.core.schema import GraspoConfig


# ── Full pipeline: YAML config → data → reward → decision ───────────────────


def test_smoke_config_load_from_yaml():
    """Config loads and validates from the example YAML file."""
    config_path = Path("config_example.yaml")
    assert config_path.exists(), "config_example.yaml not found"
    cfg = GraspoConfig.from_yaml(config_path)
    assert cfg.backend == "graspoflow"
    assert cfg.training.seed == 42
    assert cfg.training.rollout_group_size == 8


def test_smoke_data_load():
    """Sample JSONL loads into valid Sample objects."""
    samples = load_jsonl(Path("data/sample.jsonl"))
    assert len(samples) == 2
    for sample in samples:
        assert sample.messages
        assert sample.targets
        assert isinstance(sample.targets, list)
        assert len(sample.targets) >= 1
        output = sample.targets[0]["output"]
        assert "content" in output or "tool_calls" in output


def test_smoke_reward_on_sample_data():
    """Reward scoring runs on real sample data completions."""
    config = RewardConfig(check_json_markdown=True, content_reward_weight=100.0)
    reward_fn = GraspoReward(config)

    samples = load_jsonl(Path("data/sample.jsonl"))
    sample = samples[0]
    targets = sample.targets

    # A "correct" completion that should score well
    good_completion = '```json\n{"APN":"cmnet","fault_number":"13800138000"}\n```'
    result = reward_fn.score(good_completion, targets)
    assert result.reward >= 0.5, f"Good completion should score >= 0.5, got {result.reward}"
    assert result.all_right is True

    # A wrong completion should score lower
    wrong_completion = '```json\n{"APN":"wrong","fault_number":"wrong"}\n```'
    wrong_result = reward_fn.score(wrong_completion, targets)
    assert wrong_result.reward < result.reward, "Wrong completion should score lower"


def test_smoke_reward_poor_format_scores_lower():
    """Completions without JSON fences score lower."""
    config = RewardConfig(check_json_markdown=True)
    reward_fn = GraspoReward(config)

    samples = load_jsonl(Path("data/sample.jsonl"))
    targets = samples[0].targets

    good = '```json\n{"APN":"cmnet","fault_number":"13800138000"}\n```'
    no_fence = '{"APN":"cmnet","fault_number":"13800138000"}'  # missing ```

    good_result = reward_fn.score(good, targets)
    no_fence_result = reward_fn.score(no_fence, targets)
    assert no_fence_result.reward < good_result.reward, (
        f"Missing JSON fence should score lower: {no_fence_result.reward} >= {good_result.reward}"
    )


def test_smoke_reward_anti_useless_penalty():
    """Extra filler text reduces reward."""
    config = RewardConfig(check_json_markdown=True)
    reward_fn = GraspoReward(config)

    samples = load_jsonl(Path("data/sample.jsonl"))
    targets = samples[0].targets

    clean = '```json\n{"APN":"cmnet","fault_number":"13800138000"}\n```'
    verbose = (
        "Let me think about this carefully...\n\n"
        "I believe the answer should be...\n\n"
        '```json\n{"APN":"cmnet","fault_number":"13800138000"}\n```\n\n'
        "I hope this helps! Let me know if you need anything else."
    )

    clean_result = reward_fn.score(clean, targets)
    verbose_result = reward_fn.score(verbose, targets)
    # Verbose should score lower due to anti-useless penalty (same content)
    assert verbose_result.reward <= clean_result.reward, (
        f"Verbose should not score higher: {verbose_result.reward} > {clean_result.reward}"
    )


def test_smoke_reward_on_tool_call_sample():
    """Reward works on tool-call data."""
    config = RewardConfig(check_tool_call=True, check_json_markdown=False)
    reward_fn = GraspoReward(config)

    samples = load_jsonl(Path("data/sample_tool_call.jsonl"))
    assert len(samples) >= 1
    sample = samples[0]
    assert sample.expects_tool_calls

    # A correct tool call
    good = (
        '<tool_call>{"name":"query_device_status",'
        '"arguments":{"device_id":"OLT-17","panel_time":"2026-06-08T10:30:00+08:00"}}'
        '</tool_call>'
    )
    result = reward_fn.score_parsed(good, sample.targets, is_tool_call=True)
    assert result.reward >= 0, f"Tool call reward should be non-negative, got {result.reward}"


# ── Group classification pipeline ────────────────────────────────────────────


def test_smoke_group_classification_perfect_skip():
    """A group where all completions are perfect creates perfect_skip decision."""
    rewards = [1.0, 1.0, 1.0, 1.0]
    content_scores = [1.0, 1.0, 1.0, 1.0]
    decision = classify_group(
        rewards, content_scores,
        retry_count=0,
        rollout_max_retry_times=5,
        perfect_skip_reward_threshold=1.0,
        best_completion_has_parse_error=False,
    )
    assert decision.decision == GroupDecision.PERFECT_SKIP
    assert not decision.should_train


def test_smoke_group_classification_trainable():
    """A group with reward variance and no parse error is trainable."""
    rewards = [0.0, 0.3, 0.7, 1.0]
    content_scores = [0.0, 0.3, 0.7, 1.0]
    decision = classify_group(
        rewards, content_scores,
        retry_count=0,
        rollout_max_retry_times=5,
        perfect_skip_reward_threshold=1.0,
        best_completion_has_parse_error=False,
    )
    assert decision.should_train
    assert decision.decision in (
        GroupDecision.TRAINABLE_MAX_CORRECT,
        GroupDecision.TRAINABLE_NOT_CORRECT,
    )


def test_smoke_group_classification_reject_unparseable():
    """An unparseable best completion is rejected (defense line, §2.3)."""
    rewards = [0.0, 0.0, 0.3, 0.5]
    content_scores = [0.0, 0.0, 0.3, 0.5]
    decision = classify_group(
        rewards, content_scores,
        retry_count=0,
        rollout_max_retry_times=5,
        perfect_skip_reward_threshold=1.0,
        best_completion_has_parse_error=True,
        reject_unparseable_groups=True,
    )
    # Should retry (retry_count < max_retry_times)
    assert decision.should_retry


def test_smoke_group_classification_no_variance():
    """All identical rewards → invalid_no_preference_gap."""
    rewards = [0.5, 0.5, 0.5, 0.5]
    content_scores = [0.5, 0.5, 0.5, 0.5]
    decision = classify_group(
        rewards, content_scores,
        retry_count=0,
        rollout_max_retry_times=5,
        perfect_skip_reward_threshold=1.0,
        best_completion_has_parse_error=False,
    )
    assert not decision.should_train


def test_smoke_replay_ready():
    """replay_ready triggers when buffer has enough experience."""
    assert replay_ready(0, 8, 8) is False
    assert replay_ready(63, 8, 8) is False
    assert replay_ready(64, 8, 8) is True
    assert replay_ready(100, 8, 8) is True
