"""导出域的数据模型（pydantic，全部 ``extra="forbid"``）。

**纯声明**：本模块零 ``graspo`` 内部 import、模块级无 ``torch``/``peft``/
``transformers``/``vllm`` —— 因此可在不装 torch 的环境里**按文件路径加载执行**
（§1.3 计算与设施分离；C2 的被测模块之一）。

**本文件不负责**：不做 IO、不加载权重、不含任何形态的检测或合并逻辑。
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

#: 形态检测器：给定目录，返回 (是否命中, 命中证据)。纯文件系统，禁 import torch。
Detector = Callable[[Path], "tuple[bool, tuple[str, ...]]"]


class CheckpointForm(StrEnum):
    """checkpoint 形态。**这是形态词汇表的唯一真相源**（§1.4）。

    现状曾有三套并存需人工同步（``CheckpointKind`` / ``FORM_CHANNELS`` /
    ``FORM_MARKERS``）；本枚举是它们的唯一继承者。
    """

    HF_BASE_MODEL = "hf-base-model"
    PEFT_ADAPTER = "peft-adapter"
    NATIVE_LORA_RANK = "native-lora-rank"
    NATIVE_FULL_PARAM_RANK = "native-full-param-rank"


class ExportTarget(StrEnum):
    """目标产物。取值与既有 config 字面量逐字兼容（``core/schema.py`` 的 ``"peft-adapter"``）。"""

    MERGED_HF = "merged-hf"
    PEFT_ADAPTER = "peft-adapter"

    @classmethod
    def parse(cls, value: str) -> ExportTarget:
        """把 config 里的字符串收紧为枚举；未知值 ⇒ fail-closed 并列出可选值。"""
        try:
            return cls(str(value))
        except ValueError:
            known = ", ".join(sorted(item.value for item in cls))
            raise ValueError(
                f"unsupported export target {value!r}; known targets: {known}"
            ) from None


class UnsupportedForm(BaseModel):
    """显式声明某形态**当前不可导出**，并给出可操作的理由（§17.7：状态必须显式）。

    ``detector`` 让该形态仍能被 :mod:`graspo.export.classify` **识别**（从而给出
    "已识别但暂不支持"的可操作错误），而不是退化成"未知形态"。它必须是纯
    文件系统检查（无 torch）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    form: CheckpointForm
    reason: str
    detector: Detector | None = None


class ShardRef(BaseModel):
    """一个输入分片的身份：**只存 basename + 内容摘要**，不存宿主绝对路径（§10.1/§16.2）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dp_rank: int
    tp_rank: int
    pp_rank: int
    sha256: str


class PlacementRecord(BaseModel):
    """PP 放置计划快照（``manifest.placement`` / payload ``placement`` 的映射）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    placement_strategy: str = ""
    model_family: str = ""
    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    local_layer_indices: list[int] = []
    include_embeddings: bool = False
    include_lm_head: bool = False


class NativeTopology(BaseModel):
    """native ``manifest.json`` 的映射。**拓扑的唯一真相源**（不得从文件名反推，V3）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["native-lora", "native-full-param"]
    tuner_type: str = ""
    dp_size: int
    tp_size: int
    pp_size: int
    world_size: int | None = None
    checkpoint_type: str | None = None
    has_trainer_state: bool = False
    placement: PlacementRecord | None = None
    lora_target_signature: dict[str, Any] = {}


class ReplicaCheck(BaseModel):
    """DP 副本唯一性判定的**证据**（dp>1 时必填，且写进产物清单）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dp_size: int
    identical: bool
    compared_keys: int
    evidence: list[str]


class ShardPlan(BaseModel):
    """本次导出将要消费什么（执行前可审阅 —— §2.4 操作防呆的正向应用）。

    ``source_dir`` 只存在于内存（供 ``_load`` 定位分片），**不写进产物清单**
    （产物清单只记 basename + sha256）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: CheckpointForm
    target: ExportTarget
    checkpoint_name: str
    source_dir: str
    topology: NativeTopology | None = None
    consumed_shards: list[ShardRef] = []
    replica_check: ReplicaCheck | None = None
    base_model_path: str | None = None
    expected_output_keys: list[str] = []


class ArtifactManifest(BaseModel):
    """落盘的产物清单（``export_manifest.json``）：把"输入 → 产物"的绑定变成机器可判定事实。

    ⛔ 只记 **basename** 与 ``sha256``，不记宿主绝对路径（§10.1 / §16.2 / T17）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "graspo-export-manifest-1"
    form: CheckpointForm
    target: ExportTarget
    checkpoint_name: str
    consumed_shards: list[ShardRef]
    base_model_name: str | None = None
    replica_check: ReplicaCheck | None = None
    produced_files: list[str]
    produced_keys: int
    #: 产物里**显式从基座复制**的键（ckpt 未含、经声明后从基座取；⛔ 不静默丢/猜）。
    base_copied_keys: list[str] = []
    #: 由多个 native 键**融合**而成的 HF 键 → 其 native 来源键（顺序即拼接顺序）。
    mapping_fusions: dict[str, list[str]] = {}
    #: 发生 dtype 回落的键 → ``"<native>-><base>"``（如 ``"bfloat16->float32"``）。
    dtype_casts: dict[str, str] = {}


class CheckpointClassification(BaseModel):
    """形态识别结果 + 命中证据（保留现状"证据化"风格，便于人工复核）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form: CheckpointForm
    path: str
    evidence: tuple[str, ...]


class ExportResult(BaseModel):
    """一次导出调用的返回值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    destination: str
    target: ExportTarget
    manifest_path: str | None = None
    produced_files: list[str] = []
    consumed_shards: list[ShardRef] = []


__all__ = [
    "ArtifactManifest",
    "CheckpointClassification",
    "CheckpointForm",
    "Detector",
    "ExportResult",
    "ExportTarget",
    "NativeTopology",
    "PlacementRecord",
    "ReplicaCheck",
    "ShardPlan",
    "ShardRef",
    "UnsupportedForm",
]
