import importlib.util
import json
from pathlib import Path

from graspo.core.schema import GraspoConfig, Sample

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_native_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_native_checkpoint", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_samples = _MODULE.evaluate_samples


class _Generation:
    def __init__(self, completions: list[str]) -> None:
        self.completions = completions


class _EvalRuntime:
    def __init__(self, completions: list[str]) -> None:
        self.completions = completions

    def is_primary(self) -> bool:
        return True

    def generate_sample_groups(self, **kwargs):
        assert len(kwargs["samples"]) == 1
        return [_Generation(self.completions)]

    def generate_groups(self, **kwargs):
        assert len(kwargs["message_batches"]) == 1
        return [_Generation(self.completions)]


def test_evaluate_samples_scores_groups_and_scrubs_media_paths(tmp_path: Path) -> None:
    config = GraspoConfig()
    config.training.rollout_group_size = 2
    sample = Sample(
        messages=[{"role": "user", "content": "read the panel"}],
        targets=[{"id": "ok", "output": {"content": {"status": "ok"}}}],
        metadata={"source": "synthetic"},
        media=[{"type": "image", "path": "/private/panel.png"}],
    )
    perfect = '```json\n{"status":"ok"}\n```'
    partial = '```json\n{"status":"fail"}\n```'

    summary = evaluate_samples(
        _EvalRuntime([perfect, partial]),
        config,
        [sample],
        tmp_path,
        checkpoint="/checkpoints/final",
    )

    assert summary["count"] == 1
    assert summary["completion_count"] == 2
    assert summary["reward_mean"] > 0
    assert summary["reward_range_mean"] > 0
    assert summary["checkpoint"] == "/checkpoints/final"

    rows = [
        json.loads(line)
        for line in (tmp_path / "completions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["metadata"]["media"] == [{"type": "image"}]
    assert rows[0]["targets"] == [{"id": "ok", "output": {"content": {"status": "ok"}}}]
    assert rows[0]["matched_target_id"] == "ok"
    assert "/private/panel.png" not in (tmp_path / "completions.jsonl").read_text(encoding="utf-8")
