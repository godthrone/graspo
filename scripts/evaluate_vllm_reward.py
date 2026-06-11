#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from graspo.backends.native_tp.qwen_tp_adapter import _parse_qwen_tool_completion
from graspo.core.completion import ParsedCompletion
from graspo.core.data import load_jsonl
from graspo.core.reward import GraspoReward
from graspo.core.schema import GraspoConfig, Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a vLLM OpenAI-compatible chat endpoint with GRASPO reward."
    )
    parser.add_argument("--config", required=True, help="GRASPO YAML config for reward settings.")
    parser.add_argument("--data", required=True, help="Evaluation JSONL path.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL.")
    parser.add_argument("--model", required=True, help="Model id to send to vLLM.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL results.")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit.")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chat-template-kwargs",
        default=None,
        help="Optional JSON object passed through extra_body.chat_template_kwargs.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GraspoConfig.from_yaml(args.config)
    if args.temperature is not None:
        config.training.temperature = args.temperature
    if args.top_p is not None:
        config.training.top_p = args.top_p
    if args.max_tokens is not None:
        config.training.max_new_tokens = args.max_tokens
    chat_template_kwargs = (
        json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    )
    samples = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate(
        config=config,
        samples=samples,
        data_path=Path(args.data),
        base_url=args.base_url,
        model=args.model,
        output_dir=output_dir,
        samples_per_prompt=max(1, int(args.samples_per_prompt)),
        seed=int(args.seed),
        chat_template_kwargs=chat_template_kwargs,
        timeout=float(args.timeout),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def evaluate(
    *,
    config: GraspoConfig,
    samples: list[Sample],
    data_path: Path,
    base_url: str,
    model: str,
    output_dir: Path,
    samples_per_prompt: int,
    seed: int,
    chat_template_kwargs: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    reward = GraspoReward(config.reward)
    results_path = output_dir / "completions.jsonl"
    completion_count = 0
    all_right_count = 0
    reward_values: list[float] = []
    content_values: list[float] = []
    started_at = time.monotonic()
    with results_path.open("w", encoding="utf-8") as handle:
        for sample_index, sample in enumerate(samples):
            messages = _messages_for_openai(sample.messages, base_dir=data_path.parent)
            for trial in range(samples_per_prompt):
                response = _chat_completion(
                    base_url=base_url,
                    payload={
                        "model": model,
                        "messages": messages,
                        "tools": sample.tools,
                        "temperature": config.training.temperature,
                        "top_p": config.training.top_p,
                        "max_tokens": config.training.max_new_tokens,
                        "seed": seed + trial,
                        **(
                            {"chat_template_kwargs": chat_template_kwargs}
                            if chat_template_kwargs
                            else {}
                        ),
                    },
                    timeout=timeout,
                )
                choice = response["choices"][0]
                message = choice.get("message") or {}
                completion = str(message.get("content") or "")
                parsed = _parsed_vllm_message(message, completion, sample=sample)
                result = reward.score_parsed(
                    parsed, sample.ground_truth, is_tool_call=bool(sample.tools)
                )
                completion_count += 1
                all_right_count += int(result.all_right)
                reward_values.append(float(result.reward))
                content_values.append(float(result.content_score))
                handle.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "completion_index": trial,
                            "reward": result.reward,
                            "content_score": result.content_score,
                            "all_right": result.all_right,
                            "parsed_tool_calls": parsed.tool_calls,
                            "parser_name": parsed.parser_name,
                            "parser_errors": parsed.parse_errors,
                            "completion": completion,
                            "ground_truth": sample.ground_truth,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return {
        "count": len(samples),
        "completion_count": completion_count,
        "reward_mean": _mean(reward_values),
        "content_mean": _mean(content_values),
        "all_right_rate": all_right_count / completion_count if completion_count else 0.0,
        "elapsed_sec": time.monotonic() - started_at,
        "base_url": base_url,
        "model": model,
    }


def _messages_for_openai(messages: list[dict[str, Any]], *, base_dir: Path) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [
                _content_part_for_openai(part, base_dir=base_dir) for part in content
            ]
        converted.append(item)
    return converted


def _content_part_for_openai(part: Any, *, base_dir: Path) -> Any:
    if not isinstance(part, dict):
        return part
    part_type = str(part.get("type") or "").lower()
    if part_type not in {"image", "image_url"}:
        return dict(part)
    path_value = part.get("image") or part.get("path") or part.get("url")
    if isinstance(part.get("image_url"), dict):
        path_value = part["image_url"].get("url") or path_value
    if not isinstance(path_value, str):
        return dict(part)
    if path_value.startswith(("data:", "http://", "https://")):
        url = path_value
    else:
        image_path = Path(path_value)
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def _chat_completion(*, base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {exc.code}: {body}") from exc


def _parsed_vllm_message(
    message: dict[str, Any],
    completion: str,
    *,
    sample: Sample,
) -> ParsedCompletion:
    tool_calls = message.get("tool_calls") or []
    parsed_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if isinstance(name, str) and isinstance(arguments, dict):
            parsed_calls.append({"name": name, "arguments": arguments})
    if parsed_calls:
        return ParsedCompletion(
            raw_text=completion,
            think_text="",
            tool_calls=parsed_calls,
            answer_text=completion,
            parser_name="vllm_tool_calls",
            parse_errors=[],
            extra_text=completion,
        )
    return _parse_qwen_tool_completion(completion, expect_tool_calls=bool(sample.tools))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
