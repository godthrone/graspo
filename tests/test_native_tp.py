from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from graspo.backends.native_tp.logger import NativeRolloutLogger  # noqa: E402
from graspo.backends.native_tp.qwen_tp_adapter import QwenNativeTPAdapter  # noqa: E402
from graspo.backends.native_tp import qwen_tp_adapter as qwen_tp_adapter_module  # noqa: E402
from graspo.backends.native_tp.runtime import (  # noqa: E402
    NativeGeneration,
    assert_forbidden_runtime_modules_not_imported,
    validate_native_runtime_config,
)
from graspo.backends.native_tp.placement import build_placement_plan  # noqa: E402
from graspo.backends.native_tp.trainer import NativeTPGraspoTrainer  # noqa: E402
from graspo.core.schema import GraspoConfig  # noqa: E402


def test_training_defaults_are_long_run_safe():
    config = GraspoConfig()

    assert config.training.training_epoch_count == 100
    assert config.training.rollout_prompt_queue_batch_size == 1
    assert config.training.rollout_group_size == 8
    assert config.training.optimize_completion_batch_size == 4
    assert config.training.optimize_times_per_step == 4
    assert config.training.rollout_max_retry_times == 5
    assert config.training.policy_ratio_clip_eps == 0.2
    assert config.training.replay_buffer_optimize_threshold == 32
    assert config.training.max_new_tokens == 2048


def test_training_removed_aliases_are_rejected():
    with pytest.raises(ValueError, match="Removed training config field"):
        GraspoConfig.from_dict({"training": {"train_batch_size": 2}})


def test_data_field_aliases_are_rejected():
    with pytest.raises(ValueError, match="Removed data config field"):
        GraspoConfig.from_dict({"data": {"messages_field": "conversation"}})


def test_replay_buffer_optimize_threshold_is_derived():
    with pytest.raises(ValueError, match="derived"):
        GraspoConfig.from_dict({"training": {"replay_buffer_optimize_threshold": 32}})


def test_native_tp_config_parses_nested_backend_config():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {
                "native_tp": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "sequence_parallel": False,
                    "rollout_kv_cache_max_reserved_fraction": 0.65,
                    "empty_cache_after_rollout_split": True,
                }
            },
        }
    )

    assert config.native_tp.tensor_model_parallel_size == 2
    assert config.native_tp.rollout_kv_cache_max_reserved_fraction == 0.65
    assert config.native_tp.empty_cache_after_rollout_split is True
    validate_native_runtime_config(config)


def test_native_placement_config_accepts_pipeline_parallel():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {
                "native_tp": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 8,
                    "placement_strategy": "qwen36_pp8_static",
                    "pipeline_train_schedule": "simple",
                    "pipeline_max_inflight_microbatches": 4,
                }
            },
        }
    )

    validate_native_runtime_config(config)
    assert config.native_tp.pipeline_train_schedule == "simple"
    assert config.native_tp.pipeline_max_inflight_microbatches == 4


def test_native_placement_config_accepts_one_f_one_b_pipeline_schedule():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {
                "native_tp": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 8,
                    "placement_strategy": "qwen36_pp8_lm_head_only_final",
                    "train_micro_batch_size": 1,
                    "pipeline_train_schedule": "one_f_one_b",
                    "pipeline_max_inflight_microbatches": 4,
                }
            },
        }
    )

    validate_native_runtime_config(config)


def test_native_placement_config_rejects_invalid_pipeline_schedule():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {
                "native_tp": {
                    "pipeline_train_schedule": "sideways",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="pipeline_train_schedule"):
        validate_native_runtime_config(config)


def test_native_placement_config_rejects_one_f_one_b_without_pipeline_parallel():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {
                "native_tp": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "pipeline_train_schedule": "one_f_one_b",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="requires pipeline_model_parallel_size>1"):
        validate_native_runtime_config(config)


def test_native_placement_config_rejects_mixed_tp_pp_v1():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {
                "native_tp": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 4,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="pipeline_model_parallel_size>1"):
        validate_native_runtime_config(config)


def test_qwen36_static_pipeline_placement_covers_layers_once():
    plans = [
        build_placement_plan(
            strategy="qwen36_pp8_static",
            model_family="qwen3_5_text",
            num_hidden_layers=64,
            tp_size=1,
            pp_size=8,
            tp_rank=0,
            pp_rank=rank,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 16,
        )
        for rank in range(8)
    ]

    layers = [layer for plan in plans for layer in plan.local_layer_indices]

    assert sorted(layers) == list(range(64))
    assert plans[0].include_embeddings is True
    assert plans[0].include_lm_head is False
    assert plans[-1].include_embeddings is False
    assert plans[-1].include_lm_head is True
    assert all(plan.tp_size == 1 and plan.pp_size == 8 for plan in plans)
    assert len(plans[-1].local_layer_indices) < len(plans[3].local_layer_indices)
    assert len(plans[-1].local_layer_indices) <= 2


def test_qwen36_lm_head_only_final_pipeline_placement_keeps_final_stage_layerless():
    plans = [
        build_placement_plan(
            strategy="qwen36_pp8_lm_head_only_final",
            model_family="qwen3_5_text",
            num_hidden_layers=64,
            tp_size=1,
            pp_size=8,
            tp_rank=0,
            pp_rank=rank,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 16,
        )
        for rank in range(8)
    ]

    layers = [layer for plan in plans for layer in plan.local_layer_indices]

    assert sorted(layers) == list(range(64))
    assert plans[0].include_embeddings is True
    assert plans[0].include_lm_head is False
    assert plans[-1].include_embeddings is False
    assert plans[-1].include_lm_head is True
    assert plans[-1].local_layer_indices == ()
    assert all(len(plan.local_layer_indices) > 0 for plan in plans[:-1])


def test_native_tp_rejects_forbidden_framework_config():
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "backend_config": {"vllm_gpu_memory_utilization": 0.5},
        }
    )

    with pytest.raises(ValueError, match="forbids"):
        validate_native_runtime_config(config)


def test_rollout_prompt_queue_size_alias_is_rejected():
    with pytest.raises(ValueError, match="Removed training config field"):
        GraspoConfig.from_dict({"training": {"rollout_prompt_queue_size": 3}})


def test_native_tp_import_path_does_not_load_forbidden_frameworks():
    forbidden = ("nemo_rl", "vllm", "ray", "deepspeed", "accelerate", "transformer_engine", "apex")
    before = {name for name in forbidden if name in sys.modules}

    assert_forbidden_runtime_modules_not_imported()

    after = {name for name in forbidden if name in sys.modules}
    assert after == before


def test_native_rollout_logger_splits_readable_and_raw(tmp_path):
    logger = NativeRolloutLogger(tmp_path)
    logger.write_readable(
        {
            "prompt": "p",
            "completions": ["<think>x</think>```json\n{}\n```"],
            "rewards": [1.0],
            "content_scores": [1.0],
            "all_right": [True],
            "ground_truth": {"x": "ok"},
            "reward_details": [
                {
                    "raw_score": 231.0,
                    "max_score": 230.0,
                    "extracted": {"answer": "{}"},
                    "useless_text_length": 0,
                    "valid_extracted_json": True,
                }
            ],
            "generated_tokens": [8],
            "decision": "trainable_max_correct",
            "group_stats": {"count": 1},
        }
    )
    logger.write_raw(
        {"old_log_probs": torch.tensor([[0.1, 0.2]]), "sequences": torch.tensor([[1, 2]])}
    )

    readable = json.loads((tmp_path / "rollouts.readable.jsonl").read_text(encoding="utf-8"))
    raw = json.loads((tmp_path / "rollouts.raw.jsonl").read_text(encoding="utf-8"))
    assert "old_log_probs" not in readable
    assert readable["ground_truth"] == {"x": "ok"}
    assert readable["group_debug"]["likely_truncated_json_count"] == 0
    assert readable["completions"][0]["think"]["has_open"] is True
    assert readable["completions"][0]["json"]["has_markdown_json"] is True
    assert readable["completions"][0]["has_closing_json_fence"] is True
    assert readable["completions"][0]["raw_score"] == 231.0
    assert readable["completions"][0]["max_score"] == 230.0
    assert readable["completions"][0]["extracted"] == {"answer": "{}"}
    assert readable["completions"][0]["valid_extracted_json"] is True
    assert readable["completions"][0]["generated_tokens"] == 8
    assert raw["old_log_probs"][0] == pytest.approx([0.1, 0.2])


def test_readable_logger_flags_truncated_or_invalid_json(tmp_path):
    logger = NativeRolloutLogger(tmp_path)
    logger.write_readable(
        {
            "prompt": "p",
            "completions": [
                '```json\n{"x": ',
                "{}",
                '```json\n{"x": "ok"}\n```',
            ],
            "rewards": [0.01, 0.0, 1.0],
            "content_scores": [0.0, 0.0, 1.0],
            "all_right": [False, False, True],
            "reward_details": [
                {"valid_extracted_json": False},
                {"valid_extracted_json": None},
                {"valid_extracted_json": True},
            ],
            "decision": "retry",
            "group_stats": {"count": 3, "range": 1.0},
        }
    )

    readable = json.loads((tmp_path / "rollouts.readable.jsonl").read_text(encoding="utf-8"))

    assert readable["group_debug"]["missing_json_marker_count"] == 1
    assert readable["group_debug"]["unclosed_json_fence_count"] == 1
    assert readable["group_debug"]["invalid_extracted_json_count"] == 1
    assert readable["group_debug"]["likely_truncated_json_count"] == 1
    assert readable["completions"][0]["likely_truncated_json"] is True
    assert readable["completions"][1]["json"]["starts_with_object"] is True
    assert readable["completions"][2]["has_closing_json_fence"] is True


def test_production_configs_do_not_use_low_generation_caps():
    config_root = Path("configs")
    bad_values = {"max_new_tokens: 128", "max_new_tokens: 256", "max_new_tokens: 1024"}
    offenders: list[str] = []
    for path in config_root.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if any(value in text for value in bad_values):
            offenders.append(str(path))

    assert offenders == []


def test_production_configs_use_current_training_names():
    removed_keys = {
        "total_epochs",
        "group_size",
        "train_batch_size",
        "buffer_train_rounds",
        "max_retry",
        "clip_eps",
        "perfect_reward_threshold",
        "each_step_prompts_per_device",
    }
    offenders: list[str] = []
    for path in Path("configs").rglob("*.yaml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.strip().split(":", 1)[0]
            if key in removed_keys:
                offenders.append(f"{path}:{key}")

    assert offenders == []


def test_public_configs_are_complete_launch_configs():
    expected = {
        Path("configs/qwen3_8b_tp2.yaml"),
        Path("configs/qwen35_9b_mm_tp2.yaml"),
        Path("configs/qwen36_27b_pp8.yaml"),
    }
    actual = set(Path("configs").glob("*.yaml"))

    assert actual == expected
    assert not Path("configs/profiles").exists()
    assert not Path("configs/backends").exists()

    for path in sorted(actual):
        config = GraspoConfig.from_yaml(path)
        validate_native_runtime_config(config)
        assert config.backend == "native-tp"
        assert config.training.training_epoch_count == 100
        assert config.training.max_new_tokens == 2048
        assert config.lora.adapter_path is None
        assert config.lora.target_preset == "language_safe"
        assert config.export.final_formats == []
        assert config.launch.gpus


class FakeNativeRuntime:
    def __init__(self, *, primary: bool = True) -> None:
        self.generate_calls = 0
        self.train_batches = []
        self.saved = []
        self.saved_trainer_states = []
        self.loaded = []
        self.primary = primary

    def validate(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def generate_group(self, **kwargs):
        self.generate_calls += 1
        if self.generate_calls == 1:
            completions = [
                '```json\n{"x": "bad"}\n```',
                '```json\n{"x": "also_bad"}\n```',
            ]
        else:
            completions = [
                '```json\n{"x": "bad"}\n```',
                '```json\n{"x": "ok"}\n```',
            ]
        return NativeGeneration(
            sequences=torch.tensor([[1, 2, 3], [1, 2, 4]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.bool),
            action_mask=torch.tensor([[0, 1], [0, 1]], dtype=torch.bool),
            completions=completions,
            prompt_len=1,
        )

    def sequence_log_probs(self, sequences, attention_mask):
        return torch.tensor([[-0.5, -0.25], [-0.4, -0.2]])

    def train_batch(self, experiences, **kwargs):
        self.train_batches.append((experiences, kwargs))
        return {"optimized": True, "usable_experiences": len(experiences), "optimizer_steps": 1}

    def save_checkpoint(self, path, *, trainer_state=None):
        self.saved.append(str(path))
        self.saved_trainer_states.append(trainer_state)

    def load_checkpoint(self, path):
        self.loaded.append(str(path))
        return None

    def close(self) -> None:
        pass

    def is_primary(self) -> bool:
        return self.primary


class ScriptedGroupRuntime:
    def __init__(self, groups: list[list[str]], *, primary: bool = True) -> None:
        self.groups = groups
        self.generate_calls = 0
        self.message_keys = []
        self.train_batches = []
        self.saved = []
        self.saved_trainer_states = []
        self.loaded = []
        self.resume_state = None
        self.primary = primary

    def validate(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def generate_group(self, **kwargs):
        self.message_keys.append(_message_key(kwargs["messages"]))
        group_size = int(kwargs["rollout_group_size"])
        completions = self.groups[self.generate_calls]
        self.generate_calls += 1
        assert len(completions) == group_size
        return NativeGeneration(
            sequences=torch.arange(group_size * 3).view(group_size, 3),
            attention_mask=torch.ones((group_size, 3), dtype=torch.bool),
            action_mask=torch.tensor([[0, 1] for _ in range(group_size)], dtype=torch.bool),
            completions=completions,
            prompt_len=1,
        )

    def sequence_log_probs(self, sequences, attention_mask):
        return torch.full((sequences.shape[0], sequences.shape[1] - 1), -0.1)

    def train_batch(self, experiences, **kwargs):
        self.train_batches.append((experiences, kwargs))
        return {
            "optimized": True,
            "usable_experiences": len(experiences),
            "optimizer_steps": len(experiences),
            "lora_norm_delta": 0.1,
            "skipped_nonfinite": 0,
        }

    def save_checkpoint(self, path, *, trainer_state=None):
        self.saved.append(str(path))
        self.saved_trainer_states.append(trainer_state)

    def load_checkpoint(self, path):
        self.loaded.append(str(path))
        return self.resume_state

    def close(self) -> None:
        pass

    def is_primary(self) -> bool:
        return self.primary


class ScriptedQueuedRuntime(ScriptedGroupRuntime):
    def __init__(
        self, groups_by_prompt: dict[str, list[list[str]]], *, primary: bool = True
    ) -> None:
        super().__init__([], primary=primary)
        self.groups_by_prompt = groups_by_prompt
        self.generate_group_batches: list[list[str]] = []
        self.prompt_attempt_counts = {prompt: 0 for prompt in groups_by_prompt}

    def generate_groups(self, **kwargs):
        prompts = [_message_key(messages) for messages in kwargs["message_batches"]]
        self.generate_group_batches.append(prompts)
        group_size = int(kwargs["rollout_group_size"])
        generations = []
        for prompt in prompts:
            attempt_idx = self.prompt_attempt_counts[prompt]
            self.prompt_attempt_counts[prompt] = attempt_idx + 1
            completions = self.groups_by_prompt[prompt][attempt_idx]
            assert len(completions) == group_size
            generations.append(
                NativeGeneration(
                    sequences=torch.arange(group_size * 3).view(group_size, 3),
                    attention_mask=torch.ones((group_size, 3), dtype=torch.bool),
                    action_mask=torch.tensor([[0, 1] for _ in range(group_size)], dtype=torch.bool),
                    completions=completions,
                    prompt_len=1,
                    metadata={
                        "rollout_prompt_queue_batch_size": len(prompts),
                        "rollout_prompt_queue_effective_size": len(prompts),
                        "rollout_prompt_queue_fallback": False,
                        "rollout_generation_split_count": 1,
                        "decode_tokens": 2,
                    },
                )
            )
        return generations


def _ok_completion() -> str:
    return '```json\n{"x": "ok"}\n```'


def _bad_completion(value: str = "bad") -> str:
    return f'```json\n{{"x": "{value}"}}\n```'


def _mixed_group(group_size: int) -> list[str]:
    return [_bad_completion(str(idx)) for idx in range(group_size - 1)] + [_ok_completion()]


def _bad_group(group_size: int) -> list[str]:
    return [_bad_completion(str(idx)) for idx in range(group_size)]


def _message_key(messages: list[dict[str, object]]) -> str:
    return str(messages[-1]["content"])


def _no_preference_gap_group(group_size: int) -> list[str]:
    return ["{}"] + [_bad_completion(str(idx)) for idx in range(group_size - 1)]


def _write_train_data(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [{"role": "user", "content": f"p{idx}"}],
                    "ground_truth": {"x": "ok"},
                },
                ensure_ascii=False,
            )
            + "\n"
            for idx in range(count)
        ),
        encoding="utf-8",
    )


def _native_test_config(
    tmp_path,
    data: Path,
    *,
    rollout_group_size: int = 8,
    rollout_prompt_queue_batch_size: int = 1,
    optimize_completion_batch_size: int = 4,
    optimize_times_per_step: int = 1,
    rollout_max_retry_times: int = 1,
    max_steps: int = 1,
):
    return GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "data": {"train_path": str(data)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "training_epoch_count": 1,
                "rollout_group_size": rollout_group_size,
                "rollout_prompt_queue_batch_size": rollout_prompt_queue_batch_size,
                "optimize_completion_batch_size": optimize_completion_batch_size,
                "optimize_times_per_step": optimize_times_per_step,
                "rollout_max_retry_times": rollout_max_retry_times,
                "max_steps": max_steps,
                "save_steps": 1,
            },
            "backend_config": {"native_tp": {"tensor_model_parallel_size": 2}},
        }
    )


def test_native_trainer_retries_then_trains_with_fake_runtime(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "p"}], "ground_truth": {"x": "ok"}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "data": {"train_path": str(data)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "rollout_group_size": 2,
                "optimize_completion_batch_size": 1,
                "optimize_times_per_step": 1,
                "rollout_max_retry_times": 1,
                "max_steps": 1,
                "save_steps": 1,
            },
            "backend_config": {"native_tp": {"tensor_model_parallel_size": 2}},
        }
    )
    runtime = FakeNativeRuntime()
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer.train()

    assert runtime.generate_calls == 2
    assert len(runtime.train_batches) == 1
    experiences, kwargs = runtime.train_batches[0]
    assert len(experiences) == 2
    assert kwargs["optimize_times_per_step"] == 1
    assert (tmp_path / "out" / "rollouts.readable.jsonl").exists()
    raw_lines = (tmp_path / "out" / "rollouts.raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert "old_log_probs" in json.loads(raw_lines[-1])["raw"]


def test_four_trainable_groups_trigger_one_optimize_with_original_threshold(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 4)
    runtime = ScriptedGroupRuntime(
        [_mixed_group(8), _mixed_group(8), _mixed_group(8), _mixed_group(8)]
    )
    trainer = NativeTPGraspoTrainer(_native_test_config(tmp_path, data), runtime=runtime)

    trainer.train()

    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    train_step = next(event for event in stdout_events if event.get("event") == "train_step")
    assert set(train_step) >= {
        "timestamp",
        "run",
        "epoch",
        "batch",
        "optimize",
        "timing",
        "health",
        "elapsed_sec",
    }
    assert "reward_window" not in train_step
    assert "rank_metrics" not in json.dumps(train_step)
    assert '"prompt":' not in json.dumps(train_step)
    reward_batch = train_step["batch"]
    assert reward_batch["attempt_groups"] == 4
    assert reward_batch["completions"] == 32
    assert reward_batch["decisions"]["rollout_attempts"] == {
        "total": 4,
        "retry": 0,
        "terminal": 4,
    }
    assert reward_batch["decisions"]["terminal"]["total"] == 4
    assert reward_batch["decisions"]["trainable"]["max_correct"] == 4
    assert reward_batch["decisions"]["trainable"]["total"] == 4
    assert "retry_completion_count" not in json.dumps(reward_batch)
    assert "group_size" not in reward_batch
    assert train_step["epoch"]["decisions"]["trainable"]["max_correct"] == 4
    assert train_step["run"]["decisions"]["trainable"]["max_correct"] == 4
    assert train_step["optimize"]["replay_buffer_optimize_threshold"] == 32
    assert train_step["optimize"]["optimize_completion_batch_size"] == 4
    assert train_step["optimize"]["optimize_times_per_step"] == 1
    assert train_step["optimize"]["replay_buffer_trainable_completion_count"] == 32
    assert train_step["optimize"]["replay_buffer_trainable_group_count"] == 4.0
    assert set(train_step["timing"]) >= {
        "attempt_count",
        "rollout_sec",
        "reward_cpu_sec",
        "decision_sec",
        "old_logprob_sec",
        "replay_append_sec",
        "optimize_sec",
        "checkpoint_sec",
        "micro_batch_count",
        "total_observed_sec",
    }
    assert train_step["timing"]["attempt_count"] == 4
    assert train_step["timing"]["rollout_prompt_queue_batch_size"] == 1
    assert train_step["timing"]["rollout_prompt_queue_effective_size"] == 1
    assert train_step["timing"]["micro_batch_count"] >= 1

    batch_log = json.loads(
        (tmp_path / "out" / "train_batches.readable.jsonl").read_text(encoding="utf-8")
    )
    assert batch_log["batch"]["completions"] == 32
    assert batch_log["timing"]["attempt_count"] == 4
    assert batch_log["batch"]["decisions"]["trainable"]["max_correct"] == 4
    assert len(batch_log["attempts"]) == 4
    assert "completions" not in batch_log["attempts"][0]
    assert "prompt" not in batch_log["attempts"][0]
    rollout_log = json.loads(
        (tmp_path / "out" / "rollouts.readable.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "completion" in json.dumps(rollout_log["completions"])
    timing_events = [
        json.loads(line)
        for line in (tmp_path / "out" / "timing_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["phase"] for event in timing_events].count("rollout_attempt") == 4
    assert timing_events[-1]["phase"] == "optimize"
    assert set(timing_events[0]) >= {
        "timestamp",
        "elapsed_sec",
        "phase",
        "duration_sec",
        "step",
        "epoch",
    }
    assert set(timing_events[0]["details"]) >= {
        "rollout_sec",
        "reward_cpu_sec",
        "decision_sec",
        "old_logprob_sec",
        "replay_append_sec",
        "completion_count",
        "sequence_len",
    }
    assert '"prompt":' not in json.dumps(timing_events)
    checkpoint_event = next(
        event for event in stdout_events if event.get("event") == "checkpoint_saved"
    )
    assert checkpoint_event["checkpoint_save_sec"] >= 0.0


def test_rollout_prompt_queue_batch_size_batches_multiple_prompts_without_changing_threshold(
    tmp_path, capsys
):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 4)
    runtime = ScriptedQueuedRuntime(
        {
            "p0": [_mixed_group(8)],
            "p1": [_mixed_group(8)],
            "p2": [_mixed_group(8)],
            "p3": [_mixed_group(8)],
        }
    )
    config = _native_test_config(tmp_path, data, rollout_prompt_queue_batch_size=2)
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer.train()

    assert [len(batch) for batch in runtime.generate_group_batches] == [2, 2]
    assert sorted(prompt for batch in runtime.generate_group_batches for prompt in batch) == [
        "p0",
        "p1",
        "p2",
        "p3",
    ]
    assert len(runtime.train_batches) == 1
    experiences, _ = runtime.train_batches[0]
    assert len(experiences) == 32
    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    train_step = next(event for event in stdout_events if event.get("event") == "train_step")
    assert train_step["timing"]["rollout_prompt_queue_batch_size"] == 2
    assert train_step["timing"]["rollout_prompt_queue_effective_size"] == 2
    assert train_step["batch"]["decisions"]["trainable"]["total"] == 4
    assert train_step["batch"]["attempt_groups"] == 4


def test_rollout_prompt_queue_retries_only_unfinished_prompt(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 2)
    runtime = ScriptedQueuedRuntime(
        {
            "p0": [_bad_group(2), _mixed_group(2)],
            "p1": [_mixed_group(2)],
        }
    )
    config = _native_test_config(
        tmp_path,
        data,
        rollout_group_size=2,
        rollout_prompt_queue_batch_size=2,
        optimize_completion_batch_size=2,
        rollout_max_retry_times=1,
    )
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer.train()

    assert len(runtime.generate_group_batches) == 2
    assert sorted(runtime.generate_group_batches[0]) == ["p0", "p1"]
    assert runtime.generate_group_batches[1] == ["p0"]
    batch_log = json.loads(
        (tmp_path / "out" / "train_batches.readable.jsonl").read_text(encoding="utf-8")
    )
    assert batch_log["batch"]["decisions"]["rollout_attempts"] == {
        "total": 3,
        "retry": 1,
        "terminal": 2,
    }
    assert batch_log["batch"]["decisions"]["trainable"]["total"] == 2


def test_three_trainable_groups_wait_for_more_replay_items(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 3)
    runtime = ScriptedGroupRuntime([_mixed_group(8), _mixed_group(8), _mixed_group(8)])
    trainer = NativeTPGraspoTrainer(
        _native_test_config(tmp_path, data, max_steps=-1),
        runtime=runtime,
    )

    trainer.train()

    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    train_step = next(event for event in stdout_events if event.get("event") == "train_step")
    assert train_step["optimize"]["force_flush"] is True
    assert train_step["optimize"]["replay_buffer_trainable_completion_count"] == 24


def test_no_preference_gap_group_is_logged_invalid_and_not_trained(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 1)
    runtime = ScriptedGroupRuntime([_no_preference_gap_group(4)])
    config = _native_test_config(
        tmp_path,
        data,
        rollout_group_size=4,
        optimize_completion_batch_size=1,
        rollout_max_retry_times=0,
        max_steps=-1,
    )
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer.train()

    assert runtime.train_batches == []
    rollout_log = json.loads(
        (tmp_path / "out" / "rollouts.readable.jsonl").read_text(encoding="utf-8")
    )
    assert rollout_log["decision"] == "invalid_no_preference_gap"
    assert rollout_log["invalid_reason"] == "no_preference_gap"
    assert rollout_log["reward_max_median_gap"] == 0.0


def test_native_trainer_shuffles_prompt_order_each_epoch_on_cpu(tmp_path):
    data = tmp_path / "train.jsonl"
    sample_count = 6
    _write_train_data(data, sample_count)
    runtime = ScriptedGroupRuntime(
        [[_ok_completion(), _ok_completion()] for _ in range(sample_count * 2)]
    )
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "data": {"train_path": str(data)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "seed": 123,
                "training_epoch_count": 2,
                "rollout_group_size": 2,
                "optimize_completion_batch_size": 1,
                "rollout_max_retry_times": 0,
                "max_steps": -1,
            },
            "backend_config": {"native_tp": {"tensor_model_parallel_size": 2}},
        }
    )
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer.train()

    expected_epoch_0 = [f"p{idx}" for idx in range(sample_count)]
    random.Random(123).shuffle(expected_epoch_0)
    expected_epoch_1 = [f"p{idx}" for idx in range(sample_count)]
    random.Random(124).shuffle(expected_epoch_1)
    assert runtime.message_keys[:sample_count] == expected_epoch_0
    assert runtime.message_keys[sample_count:] == expected_epoch_1
    assert expected_epoch_0 != expected_epoch_1


def test_native_trainer_resumes_from_trainer_state_without_repeating_samples(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    sample_count = 6
    _write_train_data(data, sample_count)
    checkpoint_dir = tmp_path / "step_5"
    checkpoint_dir.mkdir()
    runtime = ScriptedGroupRuntime([_mixed_group(2)])
    runtime.resume_state = {
        "format": "graspo-native-tp-trainer-state",
        "global_step": 5,
        "sample_index": 2,
        "run_stats": {
            "total_groups": 2,
            "perfect_skipped": 0,
            "retries": 0,
            "invalid": 0,
            "invalid_no_preference_gap": 0,
            "trainable": 2,
            "trainable_max_correct": 2,
            "trainable_not_correct": 0,
            "optimized_steps": 5,
        },
        "epoch_stats": {
            "epoch": 0,
            "samples_seen": 2,
            "attempt_groups": 2,
            "completion_count": 4,
            "perfect_skipped": 0,
            "retries": 0,
            "invalid": 0,
            "invalid_no_preference_gap": 0,
            "trainable": 2,
            "trainable_max_correct": 2,
            "trainable_not_correct": 0,
            "reward_mean_sum": 1.0,
            "content_mean_sum": 1.0,
            "best_reward": 1.0,
        },
    }
    config = GraspoConfig.from_dict(
        {
            "backend": "native-tp",
            "data": {"train_path": str(data)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "seed": 123,
                "training_epoch_count": 1,
                "rollout_group_size": 2,
                "optimize_completion_batch_size": 1,
                "optimize_times_per_step": 1,
                "rollout_max_retry_times": 0,
                "max_steps": 6,
                "save_steps": 1,
                "resume_from_checkpoint": str(checkpoint_dir),
            },
            "backend_config": {"native_tp": {"tensor_model_parallel_size": 2}},
        }
    )
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer.train()

    expected_epoch = [f"p{idx}" for idx in range(sample_count)]
    random.Random(123).shuffle(expected_epoch)
    assert runtime.loaded == [str(checkpoint_dir)]
    assert runtime.message_keys == [expected_epoch[2]]
    assert runtime.saved_trainer_states[-1]["global_step"] == 6
    assert runtime.saved_trainer_states[-1]["epoch_stats"]["samples_seen"] == 3
    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    resumed = next(event for event in stdout_events if event.get("event") == "checkpoint_resumed")
    assert resumed["global_step"] == 5


def test_checkpoint_resume_requires_current_trainer_state(tmp_path):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 1)
    checkpoint_dir = tmp_path / "step_1"
    checkpoint_dir.mkdir()
    runtime = ScriptedGroupRuntime([_mixed_group(2)])
    runtime.resume_state = None
    config = _native_test_config(tmp_path, data, rollout_group_size=2)
    config.training.resume_from_checkpoint = str(checkpoint_dir)
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    with pytest.raises(RuntimeError, match="trainer_state"):
        trainer.train()


def test_train_batch_log_counts_group8_retry_once(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 4)
    runtime = ScriptedGroupRuntime(
        [_bad_group(8), _mixed_group(8), _mixed_group(8), _mixed_group(8), _mixed_group(8)]
    )
    trainer = NativeTPGraspoTrainer(_native_test_config(tmp_path, data), runtime=runtime)

    trainer.train()

    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    run_start = next(event for event in stdout_events if event.get("event") == "run_start")
    assert run_start["config"]["rollout_group_size"] == 8
    assert run_start["config"]["optimize_completion_batch_size"] == 4
    assert run_start["config"]["replay_buffer_optimize_threshold"] == 32
    train_step = next(event for event in stdout_events if event.get("event") == "train_step")
    reward_batch = train_step["batch"]
    assert reward_batch["attempt_groups"] == 5
    assert reward_batch["completions"] == 40
    assert reward_batch["decisions"]["rollout_attempts"] == {
        "total": 5,
        "retry": 1,
        "terminal": 4,
    }
    assert reward_batch["decisions"]["terminal"]["total"] == 4
    assert reward_batch["decisions"]["trainable"]["max_correct"] == 4
    assert reward_batch["decisions"]["trainable"]["total"] == 4
    assert reward_batch["reward"]["max_median_gap_mean"] >= 0.0


def test_train_batch_log_counts_batch2_with_one_retry(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 2)
    runtime = ScriptedGroupRuntime([_bad_group(8), _mixed_group(8), _mixed_group(8)])
    trainer = NativeTPGraspoTrainer(
        _native_test_config(
            tmp_path,
            data,
            rollout_group_size=8,
            optimize_completion_batch_size=2,
        ),
        runtime=runtime,
    )

    trainer.train()

    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    train_step = next(event for event in stdout_events if event.get("event") == "train_step")
    reward_batch = train_step["batch"]
    assert reward_batch["attempt_groups"] == 3
    assert reward_batch["completions"] == 24
    assert reward_batch["decisions"]["rollout_attempts"] == {
        "total": 3,
        "retry": 1,
        "terminal": 2,
    }
    assert reward_batch["decisions"]["trainable"]["total"] == 2
    assert reward_batch["decisions"]["terminal"]["total"] == 2


def test_native_trainer_global_stdout_is_rank0_only(capsys):
    config = GraspoConfig()
    runtime = FakeNativeRuntime(primary=False)
    trainer = NativeTPGraspoTrainer(config, runtime=runtime)

    trainer._print_json({"event": "train_step"})

    assert capsys.readouterr().out == ""


def test_qwen_adapter_writes_rank_memory_event(tmp_path):
    config = GraspoConfig.from_dict({"training": {"output_dir": str(tmp_path)}})
    adapter = QwenNativeTPAdapter(config)

    adapter._emit_rank_memory_event("unit_test", {"marker": "ok"})

    payload = json.loads((tmp_path / "rank_metrics.rank_00000.jsonl").read_text(encoding="utf-8"))
    assert payload["event"] == "rank_memory"
    assert payload["phase"] == "unit_test"
    assert payload["marker"] == "ok"
    assert set(payload["memory"]) >= {
        "allocated_mib",
        "reserved_mib",
        "max_allocated_mib",
        "max_reserved_mib",
    }


def test_qwen_adapter_close_destroys_process_group(monkeypatch):
    config = GraspoConfig()
    adapter = QwenNativeTPAdapter(config)
    calls: list[str] = []

    monkeypatch.setattr(qwen_tp_adapter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(qwen_tp_adapter_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen_tp_adapter_module.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(
        qwen_tp_adapter_module.dist, "destroy_process_group", lambda: calls.append("destroy")
    )

    adapter.close()

    assert calls == ["barrier", "destroy"]
