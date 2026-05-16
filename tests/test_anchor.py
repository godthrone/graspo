import json

from graspo.anchor import (
    AnchorGenerationConfig,
    AnsweredAnchor,
    filter_answered_anchors,
    generate_anchor_prompts,
    load_ontology,
    split_answered_anchors,
)


def test_anchor_generation_is_deterministic(tmp_path):
    knowledge_path = tmp_path / "knowledge.json"
    language_path = tmp_path / "language.json"
    knowledge_path.write_text(json.dumps({"domain": {"topic": ["leaf"]}}), encoding="utf-8")
    language_path.write_text(json.dumps({"style": ["concise", "formal"]}), encoding="utf-8")

    knowledge = load_ontology(knowledge_path)
    language = load_ontology(language_path)
    config = AnchorGenerationConfig(count=4, seed=7, languages=["English"], task_types=["qa"])

    first = generate_anchor_prompts(knowledge, language, config)
    second = generate_anchor_prompts(knowledge, language, config)

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 4
    assert first[0].anchor_meta["task_type"] == "qa"


def test_anchor_filter_and_split():
    anchors = [
        AnsweredAnchor(
            id="a",
            messages=[{"role": "user", "content": "question a"}],
            teacher_answer="a useful answer",
            teacher_model="base",
        ),
        AnsweredAnchor(
            id="b",
            messages=[{"role": "user", "content": "question b"}],
            teacher_answer="",
            teacher_model="base",
        ),
        AnsweredAnchor(
            id="c",
            messages=[{"role": "user", "content": "question a"}],
            teacher_answer="another answer",
            teacher_model="base",
        ),
    ]

    kept, stats = filter_answered_anchors(anchors, min_answer_chars=4)
    assert [item.id for item in kept] == ["a"]
    assert stats.dropped_empty == 1
    assert stats.dropped_duplicate_prompt == 1

    train, eval_items = split_answered_anchors(kept, eval_ratio=0.5, seed=1)
    assert len(train) + len(eval_items) == 1
