#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STATUSES = ("normal", "warning", "critical", "maintenance")
PRIORITIES = ("low", "medium", "high", "urgent")
DEVICES = ("PUMP", "VALVE", "CTRL", "MOTOR", "SENSOR", "COMP")
ACTIONS = {
    "normal": "log_only",
    "warning": "inspect",
    "critical": "dispatch",
    "maintenance": "schedule_service",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic/de-identified GRASPO multimodal ticket images and JSONL."
    )
    parser.add_argument("--output-dir", required=True, help="Directory for images and JSONL files.")
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--eval-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train = [
        _make_record(image_dir=image_dir, rng=rng, index=idx, split="train")
        for idx in range(args.train_count)
    ]
    eval_rows = [
        _make_record(image_dir=image_dir, rng=rng, index=idx, split="eval")
        for idx in range(args.eval_count)
    ]
    _write_jsonl(train, output_dir / "train.jsonl")
    _write_jsonl(eval_rows, output_dir / "eval.jsonl")
    manifest = {
        "format": "graspo-synthetic-multimodal-ticket-v1",
        "train_count": len(train),
        "eval_count": len(eval_rows),
        "image_dir": str(image_dir),
        "seed": args.seed,
        "task": "Extract structured JSON from a synthetic ticket/device panel/form screenshot.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _make_record(*, image_dir: Path, rng: random.Random, index: int, split: str) -> dict[str, object]:
    status = rng.choice(STATUSES)
    priority = _priority_for_status(status, rng)
    temperature = rng.randint(22, 98)
    pressure = rng.randint(80, 260)
    device_id = f"{rng.choice(DEVICES)}-{rng.randint(100, 999)}"
    ticket_id = f"{split.upper()}-{index:04d}"
    zone = f"Z{rng.randint(1, 9)}"
    shift = rng.choice(("day", "night"))
    action = ACTIONS[status]
    expected = {
        "ticket_id": ticket_id,
        "device_id": device_id,
        "zone": zone,
        "status": status,
        "priority": priority,
        "temperature_c": temperature,
        "pressure_kpa": pressure,
        "shift": shift,
        "recommended_action": action,
    }
    image_path = image_dir / f"{split}_{index:04d}.png"
    _draw_ticket_image(image_path, expected)
    prompt = (
        "Read the attached ticket/device panel screenshot and extract the fields into strict JSON. "
        "Return only a fenced ```json block with keys: ticket_id, device_id, zone, status, "
        "priority, temperature_c, pressure_kpa, shift, recommended_action."
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": "```json\n" + json.dumps(expected, separators=(",", ":")) + "\n```"},
        ],
        "prompt": prompt,
        "image": str(image_path),
        "ground_truth": expected,
        "metadata": {"split": split, "synthetic": True, "task_family": "ticket_panel"},
    }


def _priority_for_status(status: str, rng: random.Random) -> str:
    if status == "critical":
        return rng.choice(("high", "urgent"))
    if status == "warning":
        return rng.choice(("medium", "high"))
    if status == "maintenance":
        return rng.choice(("low", "medium"))
    return rng.choice(("low", "medium"))


def _draw_ticket_image(path: Path, values: dict[str, object]) -> None:
    image = Image.new("RGB", (960, 640), (244, 247, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    draw.rectangle((30, 30, 930, 610), fill=(255, 255, 255), outline=(36, 58, 86), width=3)
    draw.rectangle((30, 30, 930, 96), fill=(36, 58, 86))
    draw.text((54, 54), "MAINTENANCE TICKET / DEVICE PANEL", fill=(255, 255, 255), font=title_font)
    rows = [
        ("Ticket ID", values["ticket_id"]),
        ("Device ID", values["device_id"]),
        ("Zone", values["zone"]),
        ("Status", values["status"]),
        ("Priority", values["priority"]),
        ("Temperature C", values["temperature_c"]),
        ("Pressure kPa", values["pressure_kpa"]),
        ("Shift", values["shift"]),
        ("Recommended Action", values["recommended_action"]),
    ]
    y = 130
    for idx, (label, value) in enumerate(rows):
        row_fill = (238, 243, 248) if idx % 2 == 0 else (250, 252, 254)
        draw.rectangle((70, y - 12, 890, y + 38), fill=row_fill, outline=(198, 207, 216))
        draw.text((94, y), str(label), fill=(42, 52, 62), font=font)
        draw.text((420, y), str(value), fill=(12, 26, 41), font=font)
        y += 54
    status = str(values["status"])
    color = {
        "normal": (42, 157, 88),
        "warning": (225, 153, 33),
        "critical": (207, 54, 54),
        "maintenance": (54, 102, 207),
    }[status]
    draw.rectangle((720, 120, 890, 178), fill=color)
    draw.text((742, 140), status.upper(), fill=(255, 255, 255), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
