from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from graspo.backends.megatron_native.logger import NativeRolloutLogger  # noqa: E402
from graspo.backends.megatron_native.qwen_tp_adapter import QwenMegatronNativeAdapter  # noqa: E402
from graspo.backends.megatron_native import qwen_tp_adapter as qwen_tp_adapter_module  # noqa: E402
from graspo.backends.megatron_native.runtime import (  # noqa: E402
    NativeGeneration,
    assert_forbidden_runtime_modules_not_imported,
    validate_native_runtime_config,
)
from graspo.backends.megatron_native.trainer import MegatronNativeGraspoTrainer  # noqa: E402
from graspo.core.schema import GraspoConfig  # noqa: E402


def test_training_defaults_are_long_run_safe():
    config = GraspoConfig()

    assert config.training.training_epoch_count == 100
    assert config.training.rollout_group_size == 8
    assert config.training.optimize_completion_batch_size == 4
    assert config.training.optimize_times_per_step == 4
    assert config.training.rollout_max_retry_times == 5
    assert config.training.policy_ratio_clip_eps == 0.2
    assert config.training.replay_buffer_optimize_threshold == 32
    assert config.training.max_new_tokens == 2048


def test_training_legacy_aliases_normalize_to_canonical_names():
    config = GraspoConfig.from_dict(
        {
            "training": {
                "total_epochs": 7,
                "group_size": 3,
                "train_batch_size": 2,
                "buffer_train_rounds": 5,
                "max_retry": 4,
                "clip_eps": 0.3,
                "perfect_reward_threshold": 0.9,
            }
        }
    )

    assert config.training.training_epoch_count == 7
    assert config.training.rollout_group_size == 3
    assert config.training.optimize_completion_batch_size == 2
    assert config.training.optimize_times_per_step == 5
    assert config.training.rollout_max_retry_times == 4
    assert config.training.policy_ratio_clip_eps == 0.3
    assert config.training.perfect_skip_reward_threshold == 0.9
    assert config.training.replay_buffer_optimize_threshold == 6
    assert set(config.training.legacy_config_aliases) >= {"train_batch_size", "buffer_train_rounds"}


def test_replay_buffer_optimize_threshold_is_derived():
    with pytest.raises(ValueError, match="derived"):
        GraspoConfig.from_dict({"training": {"replay_buffer_optimize_threshold": 32}})


def test_megatron_native_config_parses_nested_backend_config():
    config = GraspoConfig.from_dict(
        {
            "backend": "megatron-native",
            "backend_config": {
                "megatron_native": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "sequence_parallel": False,
                    "rollout_kv_cache_max_reserved_fraction": 0.65,
                    "empty_cache_after_rollout_split": True,
                }
            },
        }
    )

    assert config.megatron_native.tensor_model_parallel_size == 2
    assert config.megatron_native.rollout_kv_cache_max_reserved_fraction == 0.65
    assert config.megatron_native.empty_cache_after_rollout_split is True
    validate_native_runtime_config(config)


def test_megatron_native_rejects_forbidden_framework_config():
    config = GraspoConfig.from_dict(
        {
            "backend": "megatron-native",
            "backend_config": {"vllm_gpu_memory_utilization": 0.5},
        }
    )

    with pytest.raises(ValueError, match="forbids"):
        validate_native_runtime_config(config)


def test_megatron_native_import_path_does_not_load_forbidden_frameworks():
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
    logger.write_raw({"old_log_probs": torch.tensor([[0.1, 0.2]]), "sequences": torch.tensor([[1, 2]])})

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
                "```json\n{\"x\": ",
                "{}",
                "```json\n{\"x\": \"ok\"}\n```",
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


def test_production_configs_use_canonical_training_names():
    legacy_keys = {
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
            if key in legacy_keys:
                offenders.append(f"{path}:{key}")

    assert offenders == []


def test_megatron_native_reproduction_profiles_use_original_lora_targets():
    offenders: list[str] = []
    for path in Path("configs/profiles").glob("qwen3_8b_megatron_native_tp2*.yaml"):
        config = GraspoConfig.from_yaml(path)
        if config.lora.target_modules != ["q_proj", "v_proj"]:
            offenders.append(str(path))

    assert offenders == []


class FakeNativeRuntime:
    def __init__(self, *, primary: bool = True) -> None:
        self.generate_calls = 0
        self.train_batches = []
        self.saved = []
        self.primary = primary

    def validate(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def generate_group(self, **kwargs):
        self.generate_calls += 1
        if self.generate_calls == 1:
            completions = [
                "```json\n{\"x\": \"bad\"}\n```",
                "```json\n{\"x\": \"also_bad\"}\n```",
            ]
        else:
            completions = [
                "```json\n{\"x\": \"bad\"}\n```",
                "```json\n{\"x\": \"ok\"}\n```",
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

    def save_checkpoint(self, path):
        self.saved.append(str(path))

    def close(self) -> None:
        pass

    def is_primary(self) -> bool:
        return self.primary


class ScriptedGroupRuntime:
    def __init__(self, groups: list[list[str]], *, primary: bool = True) -> None:
        self.groups = groups
        self.generate_calls = 0
        self.prompts = []
        self.train_batches = []
        self.saved = []
        self.primary = primary

    def validate(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def generate_group(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
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

    def save_checkpoint(self, path):
        self.saved.append(str(path))

    def close(self) -> None:
        pass

    def is_primary(self) -> bool:
        return self.primary


def _ok_completion() -> str:
    return '```json\n{"x": "ok"}\n```'


def _bad_completion(value: str = "bad") -> str:
    return f'```json\n{{"x": "{value}"}}\n```'


def _mixed_group(group_size: int) -> list[str]:
    return [_bad_completion(str(idx)) for idx in range(group_size - 1)] + [_ok_completion()]


def _bad_group(group_size: int) -> list[str]:
    return [_bad_completion(str(idx)) for idx in range(group_size)]


def _no_preference_gap_group(group_size: int) -> list[str]:
    return ["{}"] + [_bad_completion(str(idx)) for idx in range(group_size - 1)]


def _write_train_data(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps({"prompt": f"p{idx}", "ground_truth": {"x": "ok"}}, ensure_ascii=False) + "\n"
            for idx in range(count)
        ),
        encoding="utf-8",
    )


def _native_test_config(
    tmp_path,
    data: Path,
    *,
    rollout_group_size: int = 8,
    optimize_completion_batch_size: int = 4,
    optimize_times_per_step: int = 1,
    rollout_max_retry_times: int = 1,
    max_steps: int = 1,
):
    return GraspoConfig.from_dict(
        {
            "backend": "megatron-native",
            "data": {"train_path": str(data)},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "training_epoch_count": 1,
                "rollout_group_size": rollout_group_size,
                "optimize_completion_batch_size": optimize_completion_batch_size,
                "optimize_times_per_step": optimize_times_per_step,
                "rollout_max_retry_times": rollout_max_retry_times,
                "max_steps": max_steps,
                "save_steps": 1,
            },
            "backend_config": {"megatron_native": {"tensor_model_parallel_size": 2}},
        }
    )


def test_native_trainer_retries_then_trains_with_fake_runtime(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps({"prompt": "p", "ground_truth": {"x": "ok"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = GraspoConfig.from_dict(
        {
            "backend": "megatron-native",
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
            "backend_config": {"megatron_native": {"tensor_model_parallel_size": 2}},
        }
    )
    runtime = FakeNativeRuntime()
    trainer = MegatronNativeGraspoTrainer(config, runtime=runtime)

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
    runtime = ScriptedGroupRuntime([_mixed_group(8), _mixed_group(8), _mixed_group(8), _mixed_group(8)])
    trainer = MegatronNativeGraspoTrainer(_native_test_config(tmp_path, data), runtime=runtime)

    trainer.train()

    stdout_events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    train_step = next(event for event in stdout_events if event.get("event") == "train_step")
    assert set(train_step) >= {"timestamp", "run", "epoch", "batch", "optimize", "health", "elapsed_sec"}
    assert "reward_window" not in train_step
    assert "rank_metrics" not in json.dumps(train_step)
    assert "prompt" not in json.dumps(train_step)
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

    batch_log = json.loads((tmp_path / "out" / "train_batches.readable.jsonl").read_text(encoding="utf-8"))
    assert batch_log["batch"]["completions"] == 32
    assert batch_log["batch"]["decisions"]["trainable"]["max_correct"] == 4
    assert len(batch_log["attempts"]) == 4
    assert "completions" not in batch_log["attempts"][0]
    assert "prompt" not in batch_log["attempts"][0]
    rollout_log = json.loads(
        (tmp_path / "out" / "rollouts.readable.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "completion" in json.dumps(rollout_log["completions"])


def test_three_trainable_groups_wait_for_more_replay_items(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 3)
    runtime = ScriptedGroupRuntime([_mixed_group(8), _mixed_group(8), _mixed_group(8)])
    trainer = MegatronNativeGraspoTrainer(
        _native_test_config(tmp_path, data, max_steps=-1),
        runtime=runtime,
    )

    trainer.train()

    stdout_events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
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
    trainer = MegatronNativeGraspoTrainer(config, runtime=runtime)

    trainer.train()

    assert runtime.train_batches == []
    rollout_log = json.loads((tmp_path / "out" / "rollouts.readable.jsonl").read_text(encoding="utf-8"))
    assert rollout_log["decision"] == "invalid_no_preference_gap"
    assert rollout_log["invalid_reason"] == "no_preference_gap"
    assert rollout_log["reward_max_median_gap"] == 0.0


def test_native_trainer_shuffles_prompt_order_each_epoch_on_cpu(tmp_path):
    data = tmp_path / "train.jsonl"
    sample_count = 6
    _write_train_data(data, sample_count)
    runtime = ScriptedGroupRuntime([[_ok_completion(), _ok_completion()] for _ in range(sample_count * 2)])
    config = GraspoConfig.from_dict(
        {
            "backend": "megatron-native",
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
            "backend_config": {"megatron_native": {"tensor_model_parallel_size": 2}},
        }
    )
    trainer = MegatronNativeGraspoTrainer(config, runtime=runtime)

    trainer.train()

    expected_epoch_0 = [f"p{idx}" for idx in range(sample_count)]
    random.Random(123).shuffle(expected_epoch_0)
    expected_epoch_1 = [f"p{idx}" for idx in range(sample_count)]
    random.Random(124).shuffle(expected_epoch_1)
    assert runtime.prompts[:sample_count] == expected_epoch_0
    assert runtime.prompts[sample_count:] == expected_epoch_1
    assert expected_epoch_0 != expected_epoch_1


def test_train_batch_log_counts_group8_retry_once(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    _write_train_data(data, 4)
    runtime = ScriptedGroupRuntime(
        [_bad_group(8), _mixed_group(8), _mixed_group(8), _mixed_group(8), _mixed_group(8)]
    )
    trainer = MegatronNativeGraspoTrainer(_native_test_config(tmp_path, data), runtime=runtime)

    trainer.train()

    stdout_events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
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
    trainer = MegatronNativeGraspoTrainer(
        _native_test_config(
            tmp_path,
            data,
            rollout_group_size=8,
            optimize_completion_batch_size=2,
        ),
        runtime=runtime,
    )

    trainer.train()

    stdout_events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
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
    trainer = MegatronNativeGraspoTrainer(config, runtime=runtime)

    trainer._print_json({"event": "train_step"})

    assert capsys.readouterr().out == ""


def test_qwen_adapter_writes_rank_memory_event(tmp_path):
    config = GraspoConfig.from_dict({"training": {"output_dir": str(tmp_path)}})
    adapter = QwenMegatronNativeAdapter(config)

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
    adapter = QwenMegatronNativeAdapter(config)
    calls: list[str] = []

    monkeypatch.setattr(qwen_tp_adapter_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(qwen_tp_adapter_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen_tp_adapter_module.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(qwen_tp_adapter_module.dist, "destroy_process_group", lambda: calls.append("destroy"))

    adapter.close()

    assert calls == ["barrier", "destroy"]
