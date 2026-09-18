"""checkpoint → merged-hf 导出：识别 checkpoint 形态并合并 LoRA。

**职责**：把一个"评测要用的模型目录"准备好——可能是 base 模型（直接用）、
native GRASPO checkpoint（走既有 ``graspo export``）、或 ms-swift 产出的
PEFT/LoRA 目录（**本模块补齐的缺口**，用 peft + transformers 合并）。

**本文件不负责**：起 vLLM（调用方）、评测（`evaluate.py`）、锁卡（`guard.py`）。

**为什么需要这个模块**

现状缺口（实测确证）：``graspo export`` 只认 native 的 ``graspoflow-lora`` 格式
（``flow/lora/lora_io.py::export_from_checkpoint``），而 SFT 走 ms-swift 后端时
产物是标准的 HuggingFace PEFT 目录（``adapter_config.json`` + ``adapter_model.safetensors``）。
没有这座桥，SFT 的 checkpoint 无法进入评测链路。

**checkpoint 形态识别是纯函数**

:func:`classify_checkpoint` 只读目录结构，不 import torch、不加载权重——可在
无 GPU 环境下单测（宪法 §1.3）。真正的合并（:func:`merge_peft_checkpoint`）
才 import torch/peft/transformers，且必须在锁卡的 GPU/CPU 进程里跑。

**为什么合并必须显式声明 base 模型**

PEFT 合并需要知道"往哪个底座上合并"。adapter 目录里的
``adapter_config.json.base_model_name_or_path`` 可能指向训练机上才有的路径。
因此本模块要求调用方**显式**传入 base 模型路径，并校验 adapter 记录的底座与
它一致（不一致即拒绝——防止拿错底座合出一个假模型）。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

#: merged-hf 产物必须有的文件（判断"导出成功"的判据，与 v3 脚本一致）。
MERGED_REQUIRED_FILES = ("config.json",)

#: HF 权重文件候选名。任一存在即认为权重完整。
WEIGHT_FILE_CANDIDATES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
)


class ExportError(RuntimeError):
    """导出/识别失败。调用方应终止，不要拿半成品去评测。"""


class CheckpointKind(StrEnum):
    """checkpoint 形态。"""

    BASE_MODEL = "base-model"
    """本身就是可服务模型（有 config.json + 权重），无需合并。"""

    GRASPO_NATIVE = "graspo-native"
    """native GRASPO checkpoint（graspoflow-lora 分片格式），走 ``graspo export``。"""

    PEFT_ADAPTER = "peft-adapter"
    """标准 HF PEFT/LoRA adapter 目录（ms-swift 产物），需合并到 base。"""


@dataclass(slots=True)
class CheckpointClassification:
    """识别结果 + 证据。``evidence`` 记录判据，便于人工复核。"""

    kind: CheckpointKind
    path: Path
    evidence: tuple[str, ...]


def classify_checkpoint(path: str | Path) -> CheckpointClassification:
    """按目录内容识别 checkpoint 形态（纯文件系统检查，不加载权重）。

    判定顺序（先到先得）：
    1. ``config.json`` 且存在权重文件 → base 模型（可直接服务）。
    2. ``adapter_config.json`` → PEFT adapter（需合并）。
    3. 存在 ``*.safetensors`` 分片且无 ``config.json`` → 视为 native GRASPO
       checkpoint（native 导出路径自己会做完整校验，这里不重复判定细节）。

    Args:
        path: checkpoint 目录，或 HF 产出的 ``checkpoint-<step>`` 目录。

    Returns:
        ``CheckpointClassification``。

    Raises:
        ExportError: 路径不存在或无法识别为以上任一形态。
    """
    root = Path(path)
    if not root.is_dir():
        raise ExportError(f"checkpoint directory not found: {root}")

    evidence: list[str] = []
    has_config = (root / "config.json").is_file()
    weight = next((name for name in WEIGHT_FILE_CANDIDATES if (root / name).is_file()), None)
    adapter_config = root / "adapter_config.json"
    safetensors = sorted(root.glob("*.safetensors"))

    if has_config and weight:
        evidence.append(f"found config.json and {weight}")
        return CheckpointClassification(CheckpointKind.BASE_MODEL, root, tuple(evidence))
    if adapter_config.is_file():
        evidence.append("found adapter_config.json (no config.json) → PEFT/LoRA adapter")
        return CheckpointClassification(CheckpointKind.PEFT_ADAPTER, root, tuple(evidence))
    if safetensors and not has_config:
        names = ", ".join(item.name for item in safetensors[:3])
        evidence.append(f"found safetensors shards without config.json: {names}")
        return CheckpointClassification(CheckpointKind.GRASPO_NATIVE, root, tuple(evidence))

    raise ExportError(
        f"cannot classify checkpoint at {root}: no config.json+weights, no adapter_config.json, "
        f"no *.safetensors (entries: {sorted(item.name for item in root.iterdir())[:10]})"
    )


def resolve_peft_adapter_dir(checkpoint_dir: str | Path) -> Path | None:
    """在 ms-swift 的输出目录里向下找到真正的 adapter 目录。

    ms-swift 的 layout 随版本变化，因此**不从版本号推断**，而是做有界深度搜索
    （最多 3 层，避免扫全盘）：找到含 ``adapter_config.json`` 的目录。
    找到多个时报错（歧义无法自动裁决，交给人工指定具体目录）。

    Args:
        checkpoint_dir: 用户给的 checkpoint 路径（可能是输出根目录，也可能是
            具体的 ``checkpoint-500``）。

    Returns:
        含 ``adapter_config.json`` 的目录；找不到时返回 ``None``。

    Raises:
        ExportError: 找到多个候选（歧义）。
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    if (root / "adapter_config.json").is_file():
        return root

    found: list[Path] = []
    for depth in (1, 2, 3):
        pattern = "/".join(["*"] * depth) + "/adapter_config.json"
        found.extend(sorted(item.parent for item in root.glob(pattern)))
        if found:
            break
    distinct = sorted({item.resolve() for item in found})
    if not distinct:
        return None
    if len(distinct) > 1:
        listing = "\n  ".join(str(item) for item in distinct)
        raise ExportError(
            f"multiple PEFT adapters found under {root} — specify one explicitly:\n  {listing}"
        )
    return distinct[0]


def read_adapter_base(adapter_dir: str | Path) -> str | None:
    """读 adapter_config.json 里记录的底座模型路径。缺失/无该字段返回 ``None``。"""
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{config_path} is not valid JSON: {exc}") from None
    base = payload.get("base_model_name_or_path")
    return str(base) if base else None


def assert_base_matches(adapter_dir: str | Path, base_model_path: str | Path) -> None:
    """校验 adapter 记录的底座与调用方给出的底座一致。

    不一致或 adapter 未记录底座时**拒绝**（fail-closed）：拿错底座合并会
    产出一个既不是 A 也不是 B 的模型，而评测结果看起来"很正常"——这是最危险的
    一类静默错误。

    Raises:
        ExportError: 不一致或无法确认。
    """
    recorded = read_adapter_base(adapter_dir)
    if recorded is None:
        raise ExportError(
            f"{adapter_dir}/adapter_config.json does not record base_model_name_or_path; "
            "cannot verify the merge base — add it or pass the adapter explicitly"
        )
    if Path(recorded).name != Path(str(base_model_path)).name:
        raise ExportError(
            f"adapter base mismatch: adapter records {recorded!r}, "
            f"caller provided {str(base_model_path)!r}"
        )


def existing_merged_output(output_dir: str | Path) -> bool:
    """判断输出目录是否已有一份完整的 merged-hf 产物（避免重复合并）。"""
    root = Path(output_dir)
    if not all((root / name).is_file() for name in MERGED_REQUIRED_FILES):
        return False
    return any((root / name).is_file() for name in WEIGHT_FILE_CANDIDATES)


def prepare_output_directory(output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """准备合并输出目录。

    防呆（宪法 §2.4）：目录非空且未显式 ``overwrite`` 时拒绝——不覆盖已有产物；
    显式覆盖时先逐项列出将删除的内容，再用具体路径逐个删除（不用通配符）。

    Args:
        output_dir: 目标目录。
        overwrite: 是否允许覆盖已有非空目录（默认 False = 预授权退路未开启）。

    Returns:
        目标目录 Path。

    Raises:
        ExportError: 目录非空且未允许覆盖。
    """
    root = Path(output_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return root
    entries = sorted(root.iterdir())
    if not entries:
        return root
    if not overwrite:
        raise ExportError(
            f"output directory {root} is not empty ({len(entries)} entries); "
            "refusing to overwrite — set allow_overwrite explicitly to accept losing it"
        )
    # 显式列出 → 逐项删除（§2.4：先看得见，再动手；严禁通配符）
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return root


def merge_peft_checkpoint(
    adapter_dir: str | Path,
    base_model_path: str | Path,
    output_dir: str | Path,
    *,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    allow_overwrite: bool = False,
) -> Path:
    """把 PEFT/LoRA adapter 合并进 base 模型，落成一份可被 vLLM 服务的 HF 目录。

    这是**设施层**动作：import torch / peft / transformers，加载权重，写盘。
    必须在锁卡的进程/容器里调用；本轮（工作包边界）不实际执行。

    Args:
        adapter_dir: 含 ``adapter_config.json`` 的 PEFT 目录。
        base_model_path: 底座模型路径。必须与 adapter 记录一致。
        output_dir: 合并产物落盘目录。
        trust_remote_code: 是否允许底座模型自带远程代码。
        torch_dtype: 合并后的权重 dtype（字符串，避免在签名里 import torch）。
        allow_overwrite: 是否允许覆盖已有非空输出目录。

    Returns:
        输出目录 Path。

    Raises:
        ExportError: 前置校验失败，或合并后缺少必需文件。
    """
    # 先做文件系统层的前置校验，再导入重设施——顺序很重要：路径/底座不一致这类
    # 用户级错误应该在"加载权重"之前就被报出来，而不是先花几分钟起 torch 才发现。
    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").is_file():
        raise ExportError(f"{adapter_path} is not a PEFT adapter (no adapter_config.json)")
    assert_base_matches(adapter_path, base_model_path)

    import torch  # noqa: PLC0415  设施层延迟导入：保证模块可在无 torch 环境被导入解析
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # noqa: PLC0415

    destination = prepare_output_directory(output_dir, overwrite=allow_overwrite)

    dtype = getattr(torch, torch_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
    merged.save_pretrained(str(destination), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(str(destination))
    # 多模态底座（Qwen3.5 系列）需要把 processor 一并带过去，否则 vLLM 起不来。
    try:
        processor = AutoProcessor.from_pretrained(
            str(base_model_path), trust_remote_code=trust_remote_code
        )
        processor.save_pretrained(str(destination))
    except (OSError, ValueError) as exc:
        # 透明退路（§3.2）：纯文本底座没有 processor，属正常情况，但要留痕。
        print(f"[export] no processor copied from base ({type(exc).__name__}: {exc})")

    if not existing_merged_output(destination):
        raise ExportError(
            f"merge finished but {destination} lacks a complete merged-hf artifact "
            f"(looked for {MERGED_REQUIRED_FILES} + one of {WEIGHT_FILE_CANDIDATES})"
        )
    return destination
