"""graspo / ARD JSONL → ms-swift 数据集（SFT 与 GRPO 两种形态）。

**职责边界**

- **本模块**：读 graspo 训练 JSONL（原生 graspo 样本，或 ARD v3 ``anchor_bank.jsonl``
  记录），经 E1 已签收的 ``GraspoToMsSwiftAdapter`` 归一到 graspo 样本契约，再产出
  ms-swift 能直接消费的行；写盘到本次运行的隔离目录。
- **不负责**：ms-swift 参数映射（``_config_mapping.py``）、训练循环与算法注入
  （``trainer.py`` / ``sft_trainer.py``）、奖励打分（``reward.py``）。

**两种形态（同一份输入数据，两种用法——对应决策 D4「SFT 与 RL 共享基座」）**

- SFT：``{"messages": [system?, user, assistant=<target text>]}``。
  target 文本由 ``ripple`` 的 ``build_sft_target_text`` 生成——**与 native SFT 用的是
  同一个函数**（宪法 §1.3：算法只实现一次）。
- GRPO：``{"messages": [system?, user], "targets": <JSON 字符串>}``。
  提示词交给 ms-swift 采样，``targets`` 作为额外列透传给奖励函数（ms-swift 会把
  数据集列原样喂给 reward，见 ``GRPO 奖励适配器`` ``reward.py``）。

**输入格式自适应**：ARD v3 记录的 ``targets[i].output.content`` 是 **str**，graspo 原生
样本是 **dict**。本模块显式判别二者（``content`` 的类型），不做"猜字段"的隐式约定；
非法记录在边界处被拒（§2.3）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from graspo.flow.msswift._config_mapping import Stage
from graspo.flow.msswift.reward import GRASPO_TARGETS_COLUMN

#: ms-swift SFT / GRPO 行统一的对话列名（ms-swift AutoPreprocessor 的标准键）。
MESSAGES_KEY = "messages"

#: ms-swift 侧的**数据集列**：chat template 参数（如 ``enable_thinking``）。
#: T1 复验（4.5.3）：它不是命令行参数，而是 per-sample 列
#: （``rl_core/data.py::OnPolicySample.to_template_dict`` 只消费这一列）。
CHAT_TEMPLATE_KWARGS_KEY = "chat_template_kwargs"


def _first_target_output(record: dict[str, Any]) -> dict[str, Any] | None:
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        return None
    first = targets[0]
    if not isinstance(first, dict):
        return None
    output = first.get("output")
    return output if isinstance(output, dict) else None


def is_ard_record(record: dict[str, Any]) -> bool:
    """ARD v3 记录判据：``targets[0].output.content`` 是 **str**（graspo 原生为 dict）。"""
    output = _first_target_output(record)
    return output is not None and isinstance(output.get("content"), str)


def load_graspo_samples(path: str | Path) -> list[Any]:
    """读取训练 JSONL 并归一到 graspo ``Sample`` 契约（ARD 记录先过适配器）。

    Raises:
        ArdContractError: ARD 记录不符合共享基座契约。
        ValueError: JSONL 解析失败或记录格式非法（带行号定位）。
    """
    from graspo.core.schema import Sample
    from graspo.flow.msswift.adapter import GraspoToMsSwiftAdapter
    from graspo.ripple.data import sample_from_record

    adapter = GraspoToMsSwiftAdapter()
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: record must be a JSON object")
            try:
                normalized = adapter.convert_sample(record) if is_ard_record(record) else record
                samples.append(sample_from_record(normalized))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not samples:
        raise ValueError(f"no training samples found in {path}")
    return samples


def _chat_template_column(config: Any) -> dict[str, Any]:
    """``model.chat_template_kwargs`` → ms-swift 数据集列（空则不加列）。

    ms-swift 把 chat template 参数当**数据集列**消费（per-sample），因此这里逐行带上，
    语义与 graspo 的全局配置等价（同一份值应用到每一行）。
    """
    kwargs = dict(getattr(config.model, "chat_template_kwargs", None) or {})
    return {CHAT_TEMPLATE_KWARGS_KEY: kwargs} if kwargs else {}


def build_sft_rows(samples: list[Any], *, config: Any = None) -> list[dict[str, Any]]:
    """graspo 样本 → ms-swift SFT 行（messages 末条为 assistant 目标文本）。"""
    from graspo.ripple.parsing.xml import build_sft_target_text

    extra_columns = _chat_template_column(config) if config is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not sample.targets:
            raise ValueError(f"samples[{index}] has no targets; cannot build an SFT row")
        target_text = build_sft_target_text(sample.targets[0].get("output") or {})
        if not target_text.strip():
            raise ValueError(
                f"samples[{index}] produced an empty SFT target text "
                "(check targets[0].output.content / tool_calls)"
            )
        rows.append(
            {
                MESSAGES_KEY: [*sample.messages, {"role": "assistant", "content": target_text}],
                **extra_columns,
            }
        )
    return rows


def build_grpo_rows(samples: list[Any], *, config: Any = None) -> list[dict[str, Any]]:
    """graspo 样本 → ms-swift GRPO 行（提示词 + ``targets`` 奖励列）。

    GRPO 只喂提示词：completion 由 ms-swift 现场采样（这正是 RL 与 SFT 的区别）。
    ``targets`` 以 JSON 字符串承载，避免 ms-swift 数据集层对嵌套结构做非预期展开。
    """
    extra_columns = _chat_template_column(config) if config is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not sample.messages:
            raise ValueError(f"samples[{index}] has no messages; cannot build a GRPO prompt")
        if not sample.targets:
            raise ValueError(
                f"samples[{index}] has no targets; the graspo reward needs them "
                "(refusing to build a GRPO row that would train on a constant reward)"
            )
        rows.append(
            {
                MESSAGES_KEY: list(sample.messages),
                GRASPO_TARGETS_COLUMN: json.dumps(sample.targets, ensure_ascii=False),
                **extra_columns,
            }
        )
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> int:
    """把行写成 JSONL；返回写入条数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def prepare_ms_swift_dataset(
    config: Any,
    *,
    stage: Stage,
    work_dir: str | Path,
) -> str:
    """``config.data.train_path`` → ms-swift 数据集文件路径（本次运行的隔离目录内）。

    Args:
        config: ``GraspoConfig`` 实例（只用 ``data.train_path``）。
        stage: ``"sft"`` 或 ``"rlhf"``——决定行形态。
        work_dir: 本次运行的隔离目录（数据集是运行产物，不写回源码树，§8.5）。

    Returns:
        写好的 JSONL 路径。

    Raises:
        SystemExit: 输入文件不存在 / 无样本（边界校验，失败即退出，不静默用空数据训练）。
    """
    source = Path(str(config.data.train_path or ""))
    if not source.is_file():
        raise SystemExit(f"data.train_path does not exist: {source}")

    samples = load_graspo_samples(source)
    rows = (
        build_sft_rows(samples, config=config)
        if stage == "sft"
        else build_grpo_rows(samples, config=config)
    )
    target = Path(work_dir) / f"ms_swift_{stage}.jsonl"
    write_rows(target, rows)
    return str(target)
