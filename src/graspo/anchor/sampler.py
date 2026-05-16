from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from graspo.anchor.bank import AnchorPrompt
from graspo.anchor.ontology import Ontology


DEFAULT_TASK_TYPES = (
    "qa",
    "explanation",
    "reasoning",
    "coding",
    "translation",
    "summarization",
    "tool_use",
)


@dataclass(slots=True)
class AnchorGenerationConfig:
    count: int = 100
    seed: int = 42
    languages: list[str] = field(default_factory=lambda: ["English", "简体中文"])
    task_types: list[str] = field(default_factory=lambda: list(DEFAULT_TASK_TYPES))
    language_features_per_prompt: int = 2


def _group_by_top_level(leaves: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for leaf in leaves:
        grouped.setdefault(str(leaf.get("top_level", "default")), []).append(leaf)
    return grouped


def build_anchor_prompt(
    knowledge_leaf: dict[str, Any],
    language_features: list[dict[str, Any]],
    language: str,
    task_type: str,
) -> str:
    features = "; ".join(feature["full_path"] for feature in language_features)
    domain = knowledge_leaf["full_path"]
    if language.lower().startswith("english"):
        return (
            f"Create a natural single-turn user request in {language}. "
            f"The request should involve the domain '{domain}', follow these language features: "
            f"{features}, and represent the task type '{task_type}'. "
            "Only write the user's request, without an answer."
        )
    return (
        f"请用{language}写一个自然的单轮用户请求。"
        f"请求应涉及知识领域“{domain}”，体现这些语言特征：{features}，"
        f"任务类型为“{task_type}”。只写用户请求，不要写答案。"
    )


def generate_anchor_prompts(
    knowledge: Ontology,
    language: Ontology,
    config: AnchorGenerationConfig,
) -> list[AnchorPrompt]:
    if not knowledge.leaves:
        raise ValueError("knowledge ontology has no leaves")
    if not language.leaves:
        raise ValueError("language ontology has no leaves")

    rng = random.Random(config.seed)
    grouped = _group_by_top_level(knowledge.leaves)
    domain_keys = sorted(grouped)
    prompts: list[AnchorPrompt] = []

    for idx in range(config.count):
        domain_key = domain_keys[idx % len(domain_keys)]
        knowledge_leaf = rng.choice(grouped[domain_key])
        k = min(config.language_features_per_prompt, len(language.leaves))
        features = rng.sample(language.leaves, k)
        output_language = rng.choice(config.languages)
        task_type = config.task_types[idx % len(config.task_types)]
        prompt = build_anchor_prompt(knowledge_leaf, features, output_language, task_type)
        meta = {
            "language": output_language,
            "knowledge_domain": knowledge_leaf["full_path"],
            "knowledge_top_level": domain_key,
            "language_features": [feature["full_path"] for feature in features],
            "task_type": task_type,
            "source": "anchor_generator",
            "seed": config.seed,
        }
        prompts.append(AnchorPrompt.from_prompt(prompt=prompt, meta=meta, salt=str(config.seed)))

    return prompts

