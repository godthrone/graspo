import json

from graspo.sft.data import load_mixed_sft_samples


def test_load_mixed_sft_samples(tmp_path):
    hard_path = tmp_path / "hard.jsonl"
    anchor_path = tmp_path / "anchor.jsonl"
    hard_path.write_text(
        json.dumps(
            {
                "sample_type": "hard",
                "messages": [{"role": "user", "content": "extract"}],
                "target": '{"field":"value"}',
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    anchor_path.write_text(
        json.dumps(
            {
                "sample_type": "anchor",
                "messages": [{"role": "user", "content": "general question"}],
                "teacher_answer": "general answer",
                "teacher_model": "base",
                "anchor_meta": {"domain": "general"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_mixed_sft_samples(hard_path, anchor_path)

    assert [sample.sample_type for sample in samples] == ["hard", "anchor"]
    assert samples[0].target == '{"field":"value"}'
    assert samples[1].target == "general answer"
