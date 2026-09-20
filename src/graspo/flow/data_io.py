"""JSONL 文件 I/O：加载训练数据、写入样本。

设施层（flow）负责文件系统操作，算法层（ripple）负责数据变换。
"""

import json
from pathlib import Path

from graspo.core.schema import Sample
from graspo.ripple.data import sample_from_record


def load_jsonl(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                samples.append(sample_from_record(record))
            except Exception as exc:
                raise ValueError(f"invalid JSONL record at {path}:{line_no}: {exc}") from exc
    return samples


def write_jsonl(samples: list[Sample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")
