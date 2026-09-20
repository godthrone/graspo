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

**★ 加载器类别必须按底座实际配置判定（本项目实测事故）**

底座分两类，**语言层在模型树里的位置不同**，必须用对应的加载器类别：

* **多模态**（Qwen3.5 全系，本项目所有档）：``Qwen3_5ForConditionalGeneration``，
  语言塔挂在 ``model.language_model`` 下（全名 ``model.language_model.layers.{i}...``），
  另有 ``model.visual`` 视觉塔。加载器 = 多模态 auto 类（``AutoModelForImageTextToText`` /
  ``AutoModel``），它们按 ``config.architectures`` 解析到 ``...ForConditionalGeneration``。
* **纯文本**：``...ForCausalLM``，语言层直接挂在 ``model.layers.{i}...``。

历史缺陷（缺陷 D，已修）：本模块**固定**用 ``AutoModelForCausalLM``（纯文本类）加载
多模态底座。纯文本类**能**把权重读进来（`_keys_to_ignore_on_load_unexpected` 会静默
忽略 ``^model.visual.*``），但 ``save_pretrained`` 落出的 ``config.json`` 变成
``architectures=['Qwen3_5ForCausalLM'] / model_type=qwen3_5_text / 无 vision_config``
⇒ **产物在、退出码为 0、视觉塔已丢**，vLLM 直接拒绝
（``TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig``）；若被吞掉异常，
就是在评一个丢掉视觉塔的错模型，准确率是**假数字**（宪法 §3.4：静默降级到更差的模型
是"坏的退路"）。

所以现在：**加载器类别由 :func:`detect_multimodal_base` 按底座 ``config.json``
自动判定**（不硬编码模型名/类别），并在加载后 :func:`_assert_loader_class_matches_base`
复验类别真的对得上（§2.3 边界校验 + §3.4 拒绝静默降级）。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

#: merged-hf 产物必须有的文件（判断"导出成功"的判据，与 v3 脚本一致）。
MERGED_REQUIRED_FILES = ("config.json",)

#: 底座 ``config.json`` 文件名（加载器类别判据的唯一真相源，§1.4）。
BASE_CONFIG_FILENAME = "config.json"

#: 底座 config 里声明视觉塔的字段。多重判据**任一命中即算多模态**，不猜（§2.2 显式）。
#:
#: * ``has_vision_config`` —— 本项目既有约定（``flow/adapters/models/common/model_builders.py``
#:   把 ``"vision_config" in config`` 写进该字段；``flow/trainer/preflight.py`` 也读它）。
#: * ``vision_config`` —— HF 多模态 config 的标准字段（``Qwen3_5Config.vision_config``）。
#: * ``image_token_id`` / ``video_token_id`` —— 多模态 config 的输入侧标记。
#: * ``architectures`` 含 ``ConditionalGeneration`` —— 多模态架构名的共同后缀。
BASE_MULTIMODAL_FLAG_KEYS = (
    "has_vision_config",
    "vision_config",
    "image_token_id",
    "video_token_id",
)
BASE_MULTIMODAL_ARCHITECTURE_MARKERS = ("ConditionalGeneration",)

#: HF 权重文件候选名。任一存在即认为权重完整。
WEIGHT_FILE_CANDIDATES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
)

#: 多模态底座的"语言层命名空间"。
#:
#: Qwen3.5 是**多模态**底座：`Qwen3_5ForConditionalGeneration.model`（= `Qwen3_5Model`）
#: 把语言塔挂成 `model.language_model = Qwen3_5TextModel(...)`，所以语言层真实全名是
#: ``model.language_model.layers.{i}.self_attn.q_proj``。
#: 而 `AutoModelForCausalLM` 命中的是**纯文本**类 `Qwen3_5ForCausalLM`
#: （`self.model = Qwen3_5TextModel(...)`），语言层全名是
#: ``model.layers.{i}.self_attn.q_proj`` —— **没有**这棵子树。
#:
#: ms-swift 在**多模态**路径上训练，因此它写进 `adapter_config.json` 的
#: `target_modules` 带 `model.language_model.` 前缀。前缀本身没错；错的是
#: 本模块用纯文本类去装它 ⇒ peft 的正则一条都匹配不上 ⇒ `ValueError`。
#:
#: 与 `graspo.flow.adapters.models.common.model_builders` 的 `key_prefix` 同值，
#: 但那支模块 import torch（设施层），本模块必须保持"无 torch 也能 import"（§1.3），
#: 所以这里独立声明一份常量，不做跨层 import。
MULTIMODAL_LANGUAGE_MODEL_PREFIX = "model.language_model."

#: 同一前缀在 **regex 形态** 下的转义写法。真实 ms-swift / peft 0.19.1 产物写的是
#: 正则字符串（实测：
#: ``.local/.../task-r3-effect/evidence/light_evidence/T013/adapter_config.json``）：
#: ``^(model\\.language_model(?=\\.).*\\....)$``。
MULTIMODAL_LANGUAGE_MODEL_PREFIX_REGEX = r"model\.language_model"


def normalize_language_model_targets(targets: str | Iterable[str]) -> str | list[str]:
    """把"多模态语言层命名空间"的 target_modules 归一到纯文本模型的命名空间。

    ★ **只在纯文本加载器下使用**（缺陷 D 复核结论，见 :func:`_resolve_injection_targets`）：
    多模态加载器（``model.language_model.*``）下 adapter 的目标名**本来就是对的**，
    调用本函数会**放宽**正则而不是改名（``^(model(?=\\.).*\\.(...))$`` 对
    ``model.language_model.*`` 与 ``model.layers.*`` **都** fullmatch）⇒ 多模态路径
    **禁止**调用它。本函数保留是为了纯文本加载器路径（语言层挂 ``model.layers.*``）。

    纯函数（不 import torch / peft），可在无 GPU 环境单测。

    纯文本类 ``...ForCausalLM`` 的语言层挂在 ``model.layers.{i}`` 下，**没有**
    ``model.language_model`` 子树，所以 adapter 里带该前缀的目标必须剥掉前缀才能被
    peft 的正则匹配上。

    两种形态都要处理（**保持入参形态**，不改语义）：

    * **regex 字符串**（真实 T013 产物形态）—— 只把前缀 ``model\\.language_model``
      换成 ``model``，其余正则结构（`^`、`(?=\\.)`、`.*`、分支顺序）**一字不动**：
      ``^(model\\.language_model(?=\\.).*\\.(X|Y))$`` → ``^(model(?=\\.).*\\.(X|Y))$``
    * **列表 / 集合 / 裸字符串** —— 剥掉字面前缀 ``model.language_model.``。

    不含该前缀的入参**原样返回**（既有可用路径不受影响）。

    Args:
        targets: `adapter_config.json` 里的 `target_modules`（regex 字符串或列表/集合）。

    Returns:
        与入参**同形态**的归一结果（str 进 str 出，list 进 list 出）。

    Raises:
        ExportError: 入参既非 str 也非 list/tuple/set —— 不做猜测式转换（§2.3）。
    """
    if isinstance(targets, str):
        # 顺序很重要：先试 regex 转义形态（更具体），再试字面形态。
        return targets.replace(MULTIMODAL_LANGUAGE_MODEL_PREFIX_REGEX, "model").replace(
            MULTIMODAL_LANGUAGE_MODEL_PREFIX, "model."
        )

    if not isinstance(targets, (list, tuple, set, frozenset)):
        raise ExportError(
            f"unsupported target_modules type {type(targets).__name__}: {targets!r}; "
            "expected str / list / set"
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for item in targets:
        candidate = str(item)
        if candidate.startswith(MULTIMODAL_LANGUAGE_MODEL_PREFIX):
            candidate = "model." + candidate[len(MULTIMODAL_LANGUAGE_MODEL_PREFIX) :]
        if candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)
    return normalized


class ExportError(RuntimeError):
    """导出/识别失败。调用方应终止，不要拿半成品去评测。"""


@dataclass(slots=True)
class BaseModelShape:
    """底座形态判据（纯读 ``config.json``，不 import torch）。"""

    path: Path
    multimodal: bool
    """``True`` = 多模态底座（有视觉塔），必须用多模态加载器类别。"""
    evidence: tuple[str, ...]
    """命中的判据（逐条列出），便于人工复核与事故复盘。"""


def read_base_config(base_model_path: str | Path) -> dict[str, Any]:
    """读底座 ``config.json``（纯 JSON，不 import torch）。

    这是加载器类别判据的**唯一真相源**（§1.4）——不解析模型名、不硬编码类别。

    Raises:
        ExportError: 底座目录不存在，或缺 ``config.json``，或不是合法 JSON 对象。
    """
    root = Path(base_model_path)
    if not root.is_dir():
        raise ExportError(f"base model directory not found: {root}")
    config_path = root / BASE_CONFIG_FILENAME
    if not config_path.is_file():
        raise ExportError(
            f"{config_path} not found — cannot determine the loader class for the base "
            "model; refusing to guess a class (a wrong class silently drops the vision tower)"
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{config_path} is not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise ExportError(f"{config_path}: expected a JSON object, got {type(payload).__name__}")
    return payload


def is_multimodal_base_config(config: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """判断一份底座 config 是否声明了视觉塔，并给出**命中判据**（纯函数）。

    多重判据任一命中即算多模态；全部未命中即纯文本。判据见
    :data:`BASE_MULTIMODAL_FLAG_KEYS` / :data:`BASE_MULTIMODAL_ARCHITECTURE_MARKERS`。

    Args:
        config: 底座 ``config.json`` 解析出的 dict。

    Returns:
        ``(是否多模态, 命中判据元组)`` —— 判据逐条形如 ``"vision_config present"``，
        既是回显也是证据（宪法 §2.2 显式即防呆）。
    """
    evidence: list[str] = []

    for key in BASE_MULTIMODAL_FLAG_KEYS:
        value = config.get(key)
        # 注意用 is not None / 显式真值判断：`image_token_id = 0` 是合法 id，不是"未提供"（§2.2）。
        if value is None:
            continue
        if key == "has_vision_config":
            if bool(value):
                evidence.append(f"{key}={value!r}")
            continue
        evidence.append(f"{key} present")

    architectures = config.get("architectures")
    if isinstance(architectures, (list, tuple)):
        for name in architectures:
            text = str(name)
            if any(marker in text for marker in BASE_MULTIMODAL_ARCHITECTURE_MARKERS):
                evidence.append(f"architectures contains {text!r}")

    return bool(evidence), tuple(evidence)


def detect_multimodal_base(base_model_path: str | Path) -> BaseModelShape:
    """判定底座形态（纯文件系统 + JSON，无 torch ⇒ 可无 GPU 单测，§1.3）。

    加载器类别**必须**由它决定，不得硬编码（缺陷 D 的根因就是硬编码了纯文本类）。

    Raises:
        ExportError: 底座目录/配置缺失或不可解析（fail-closed，不猜）。
    """
    root = Path(base_model_path)
    config = read_base_config(root)
    multimodal, evidence = is_multimodal_base_config(config)
    if not evidence:
        evidence = ("no vision marker found in the base config → text-only base",)
    return BaseModelShape(path=root, multimodal=multimodal, evidence=evidence)


#: 多模态底座按序尝试的 auto 类名（transformers 5.12.1 均存在；任一可解析出
#: ``...ForConditionalGeneration`` 即可，选谁由 ``config.architectures`` 决定）。
MULTIMODAL_AUTO_MODEL_CLASS_NAMES = ("AutoModelForImageTextToText", "AutoModel")

#: 纯文本底座用的 auto 类名。
TEXT_AUTO_MODEL_CLASS_NAME = "AutoModelForCausalLM"


def _resolve_auto_model_class(transformers_module: Any, *, multimodal: bool) -> tuple[str, Any]:
    """按底座形态挑加载器类别，返回 ``(类名, 类对象)``。

    **不硬编码类别**：候选名按序尝试，第一个真的存在于当前 transformers 里的胜出。
    多模态底座的类最终由 ``config.architectures`` 决定（``AutoModelForImageTextToText``
    会解析到 ``Qwen3_5ForConditionalGeneration``）。

    Raises:
        ExportError: 候选类在当前 transformers 里一个都不存在（不猜、不回落，
            fail-closed —— 回落成纯文本类正是缺陷 D）。
    """
    candidates = MULTIMODAL_AUTO_MODEL_CLASS_NAMES if multimodal else (TEXT_AUTO_MODEL_CLASS_NAME,)
    for name in candidates:
        cls = getattr(transformers_module, name, None)
        if cls is not None:
            return name, cls
    raise ExportError(
        f"none of the loader classes {list(candidates)} exist in the installed "
        f"transformers ({getattr(transformers_module, '__version__', 'unknown')}); "
        "refusing to fall back to a text-only loader (that silently drops the vision tower)"
    )


def _assert_loader_class_matches_base(base: Any, shape: BaseModelShape) -> None:
    """复验"装进内存的模型"与底座形态一致（§3.4：拒绝静默降级）。

    只看**结构**，不看模型名：多模态底座装进来的模型必须有 ``model.language_model``
     子树（语言塔的多模态挂法）；纯文本底座反之。

    Raises:
        ExportError: 结构对不上 ⇒ 产物会丢掉视觉塔，必须当场停，不许继续合并。
    """
    model = getattr(base, "model", None)
    has_language_model = getattr(model, "language_model", None) is not None

    if shape.multimodal and not has_language_model:
        raise ExportError(
            f"base {shape.path} is multimodal (evidence: {'; '.join(shape.evidence)}) but "
            f"the loaded object {type(base).__name__!r} has no 'model.language_model' "
            "sub-tree — the vision tower would be dropped and the eval numbers would be fake; "
            "refusing to continue"
        )
    if not shape.multimodal and has_language_model:
        raise ExportError(
            f"base {shape.path} looks text-only but the loaded object "
            f"{type(base).__name__!r} exposes 'model.language_model' — the class was chosen "
            "for the wrong shape; refusing to continue"
        )


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


def read_adapter_targets(adapter_dir: str | Path) -> str | list[str] | None:
    """读 `adapter_config.json` 的 `target_modules`（**保持原形态**）。缺失返回 ``None``。

    形态很重要：peft 对 **str** 走 ``re.fullmatch`` 正则匹配，对 **list** 走
    ``key.endswith(f".{item}")`` 后缀匹配（实测 peft 0.18/0.19 均如此）。
    混形态会静默改变匹配语义，所以这里原样透传。
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{config_path} is not valid JSON: {exc}") from None
    targets = payload.get("target_modules")
    if targets is None:
        return None
    if isinstance(targets, str):
        return targets
    if isinstance(targets, (list, tuple, set, frozenset)):
        return [str(item) for item in targets]
    raise ExportError(
        f"{config_path}: unsupported target_modules type {type(targets).__name__}; "
        "expected str / list / set"
    )


def targets_use_language_model_namespace(targets: str | Iterable[str] | None) -> bool:
    """判断 adapter 的 ``target_modules`` 是否写在多模态语言层命名空间里（纯函数）。

    逐字检查 ``model.language_model``（正则转义形态与字面形态都算）是否出现在任一
    目标里。用于"多模态加载器 + 多模态目标名"的一致性断言（§2.3 边界校验）。
    """
    if targets is None:
        return False
    items = [targets] if isinstance(targets, str) else [str(item) for item in targets]
    return any(
        MULTIMODAL_LANGUAGE_MODEL_PREFIX_REGEX in item or MULTIMODAL_LANGUAGE_MODEL_PREFIX in item
        for item in items
    )


def _resolve_injection_targets(
    adapter_dir: str | Path,
    *,
    multimodal: bool = False,
) -> tuple[Any, str | list[str] | None]:
    """算出注入 peft 时应当使用的 `target_modules`（按加载器类别决定是否归一命名空间）。

    **归一化只在纯文本加载器下成立**（缺陷 D 复核结论）：

    * ``multimodal=False``（纯文本类 ``AutoModelForCausalLM``，语言层挂 ``model.layers.*``）：
      ms-swift 多模态产物写的 ``model.language_model.*`` 一条都匹配不上 ⇒ 必须剥前缀，
      否则 peft 报 ``Target modules ... not found in the base model``（保留既有可用路径）。
    * ``multimodal=True``（多模态类，语言层挂 ``model.language_model.*``）：
      adapter 的目标名**本来就是对的**，**必须原样保留** —— 归一化此时不但多余，
      而且会**放宽**成 ``^(model(?=\\.).*\\.(...))$``（既匹配 ``model.language_model.*``
      也匹配 ``model.layers.*``）并在层名匹配宽的正则下波及 ``model.visual.*`` 之外的
      意外模块（实测：`re.fullmatch` 结果见交付报告）。所以这里**拒绝归一**，
      并在目标名与多模态命名空间不一致时 fail-closed，而不是替 adapter "猜"一个名字。

    为什么必须走"显式构造 PeftConfig"而不是给 ``PeftModel.from_pretrained``
    多传一个 ``target_modules=`` 关键字：peft 0.18 的
    ``PeftConfig.from_pretrained`` 第 262 行是
    ``kwargs = {**class_kwargs, **loaded_attributes}`` —— **adapter 文件里的值
    覆盖调用方传的值**，所以关键字覆盖是无效的（实测确证：报错里出现的仍是
    多模态前缀）。必须先把配置对象读出来、按需改掉 `target_modules`，再作为
    ``config=`` 传进去。

    Args:
        adapter_dir: 含 ``adapter_config.json`` 的 PEFT 目录。
        multimodal: 底座是否多模态（决定是否做命名空间归一化）。

    Returns:
        (配置, 预览用目标列表)。adapter 没有 `adapter_config.json` 时返回
        ``(None, None)``，由调用方交回 peft 自行报错（保持原报错形态）。

    Raises:
        ExportError: 多模态加载器下，adapter 的目标名不在 ``model.language_model.*``
            命名空间里（说明 adapter 与底座不是同一条路径训出来的，不许硬凑）。
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.is_file():
        return None, None

    from peft import PeftConfig  # noqa: PLC0415  设施层延迟导入

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    targets = read_adapter_targets(adapter_dir)
    if targets is None:
        return peft_config, None

    original = targets if isinstance(targets, str) else list(targets)

    if multimodal:
        # 多模态加载器：目标名已经是正确的命名空间，**不做任何改写**。
        if not targets_use_language_model_namespace(targets):
            raise ExportError(
                f"{config_path}: base is multimodal but target_modules {original!r} do not "
                f"reference {MULTIMODAL_LANGUAGE_MODEL_PREFIX!r} — the adapter was not "
                "trained on this (multimodal) module namespace; refusing to rewrite the "
                "targets and merge a wrong-shaped model"
            )
        print(
            f"[export] keeping the original target_modules for the multimodal loader "
            f"(namespace {MULTIMODAL_LANGUAGE_MODEL_PREFIX!r} is correct as-is)"
        )
        return peft_config, original

    normalized = normalize_language_model_targets(targets)
    if normalized != original:
        print(
            f"[export] normalized target_modules for the text loader "
            f"(dropped prefix {MULTIMODAL_LANGUAGE_MODEL_PREFIX!r}): {normalized}"
        )
        peft_config.target_modules = normalized
    return peft_config, normalized


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

    # ★ 加载器类别必须按底座实际配置判定（缺陷 D 根因：这里曾硬编码 AutoModelForCausalLM）。
    # 纯读 config.json，无重设施 ⇒ 判错也在"花几分钟起 torch"之前就暴露。
    shape = detect_multimodal_base(base_model_path)

    import torch  # noqa: PLC0415  设施层延迟导入：保证模块可在无 torch 环境被导入解析
    import transformers  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoProcessor, AutoTokenizer  # noqa: PLC0415

    destination = prepare_output_directory(output_dir, overwrite=allow_overwrite)

    loader_name, loader_cls = _resolve_auto_model_class(transformers, multimodal=shape.multimodal)
    print(
        f"[export] base {shape.path} is "
        f"{'multimodal' if shape.multimodal else 'text-only'} "
        f"({' ; '.join(shape.evidence)}) → loader class {loader_name}"
    )

    dtype = getattr(torch, torch_dtype)
    base = loader_cls.from_pretrained(
        str(base_model_path),
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="cpu",
    )
    # 复验装进来的模型与底座形态一致：不一致 ⇒ 产物会丢视觉塔，当场停（§3.4 拒绝静默降级）。
    _assert_loader_class_matches_base(base, shape)

    merged = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        config=_resolve_injection_targets(adapter_path, multimodal=shape.multimodal)[0],
    ).merge_and_unload()
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
