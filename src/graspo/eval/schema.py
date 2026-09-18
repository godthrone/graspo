"""评测产物契约（evaluation artifact contract）。

**职责**：定义评测运行落盘产物（`eval_report.json`）的 pydantic 模型。
产物必须自包含到"一个独立的人/AI 只读这一个文件就能复现本次评测结论"：
模型标识、数据集标识、温度、样本数、有效样本数、口径版本、逐样本明细、
时间戳、环境指纹——一样都不能少（工作包"产物契约"硬要求）。

**本文件不负责**：计算准确率（`criteria.py`）、读数据（`dataset.py`）、
发请求（`vllm_client.py`）。这里只有数据结构 + 校验规则。

**为什么逐样本明细要放进报告**：聚合结论必须可被独立重算。只存一个
"acc=0.53"的数字，事后无法验证分母、无法审计哪些样本被判错、无法切换口径
（如剔除图像重叠子集）重算。明细是复现的最小充分信息。
"""

from __future__ import annotations

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: 产物 schema 版本。字段增删或语义变更即递增。
EVAL_ARTIFACT_SCHEMA_VERSION = "graspo-eval-report-1"

#: 评测目标模型在本链路中的角色名。base 是 GRASPO Δ 的基线锚点（用户已拍板）。
EvalRole = Literal["base", "after", "sft", "other"]


class EvalModel(BaseModel):
    """被评测模型的标识。``path`` 是运行时路径（机器相关信息，留在 .local/产物内）。"""

    model_config = ConfigDict(extra="forbid")

    served_name: str
    role: EvalRole
    path: str
    base_model_path: str | None = None
    checkpoint_path: str | None = None
    export_format: str | None = None


class EvalDataset(BaseModel):
    """被评测数据集的标识与规模。``sha256`` 是文件内容哈希，锁定数据版本。"""

    model_config = ConfigDict(extra="forbid")

    path: str
    split: str
    sha256: str
    sample_count_total: int
    sample_count_requested: int


class EvalDecoding(BaseModel):
    """解码参数。**temperature 恒为 0**，不接受调用方覆盖（见 vllm_client）。"""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.0
    top_p: float
    max_tokens: int
    enable_thinking: bool
    seed: int | None = None

    def model_post_init(self, __context: Any) -> None:
        """防线：温度必须是 0。非 0 直接拒绝，不给"手滑改成 0.1"留后门。"""
        if self.temperature != 0.0:
            raise ValueError(
                f"temperature must be exactly 0.0 (got {self.temperature}); "
                "the v3 0.1 setting produced ±1.6pp run-to-run jitter, "
                "same order as the pass margin — locked to 0 by user ruling"
            )


class EvalCriteria(BaseModel):
    """记录本次评测使用的准确率口径版本，供事后比对。"""

    model_config = ConfigDict(extra="forbid")

    version: str
    source: str
    definition: str = "all_right = (pred tool name AND pred action_type) exact-match gt"


class EvalSummary(BaseModel):
    """聚合结论。分母口径：``valid = correct + incorrect``（error 样本两边都不计）。"""

    model_config = ConfigDict(extra="forbid")

    sample_count_total: int
    sample_count_valid: int
    sample_count_error: int
    correct: int
    incorrect: int
    accuracy: float
    accuracy_percent: float
    #: 按 gt 工具名汇总的 [正确数, 有效数]，供诊断。
    by_tool: dict[str, list[int]] = Field(default_factory=dict)
    #: 按 gt 动作方向汇总的 [正确数, 有效数]，供诊断。
    by_action: dict[str, list[int]] = Field(default_factory=dict)
    #: **重叠分析是否真的执行过**。这是与下面两个数值字段正交的溯源信息：
    #: `False` = 没跑（未提供训练集）；`True` = 跑了。**必须单独存在**，因为
    #: "跑了但重叠集为空" 与 "根本没跑" 在数值上无法区分——前者会给出
    #: `accuracy_percent_excluding_overlap == accuracy_percent` 且
    #: `overlap_excluded_count == 0`，与"未跑"的 `None` 相比虽可区分，但一旦有人
    #: 只看数值就分不清"重叠为零"和"没做这个分析"。布尔字段让意图无从误读。
    overlap_analysis_performed: bool = False
    #: 图像重叠子集口径：剔除图像集是训练样本子集的测试样本后的准确率。
    #: `None` = **未计算**（overlap_analysis_performed 为 False 时必为 None）。
    #: 见 report.md「图像重叠」节。
    accuracy_percent_excluding_overlap: float | None = None
    #: 被剔除的样本数（仅当上一字段非 None 时有意义）。
    overlap_excluded_count: int | None = None

    def model_post_init(self, __context: Any) -> None:
        """契约自检：分析未执行 ⇒ 两个数值字段必须都是 ``None``（§2.1 契约即防呆）。

        反向不强制：分析执行过但重叠集为空时，字段是 ``0`` / ``0.0`` 而同主口径，
        这是**合法的**——那正是"做了且结果为零"。用布尔字段而非数值来区分。
        """
        if not self.overlap_analysis_performed:
            if self.accuracy_percent_excluding_overlap is not None:
                raise ValueError(
                    "overlap_analysis_performed=False but "
                    "accuracy_percent_excluding_overlap is set; an unperformed analysis "
                    "must not report a number"
                )
            if self.overlap_excluded_count is not None:
                raise ValueError(
                    "overlap_analysis_performed=False but overlap_excluded_count is set"
                )


class EvalTrainSubset(BaseModel):
    """本次评测用于**重叠分析**的训练子集标识（宪法 §1.4 单一真相源）。

    为什么必须记下来：不同档位的评测会取训练集的不同切片（如"train 前 100 条"
    vs "train 前 20 条"）。那个切片决定了"哪些测试样本被判为图像完全重叠"，
    进而决定了剔除口径的数值。只记 shape 不记切片，事后无法解释两次评测为什么
    给出不同的 `accuracy_percent_excluding_overlap`，也无法判断两个 Δ 是否可比。

    ``sample_count`` 是**实际使用的切片长度**（可能与 sha256 对应的整份文件不同）。
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    sample_count: int


class EvalEnvironmentPointer(BaseModel):
    """环境指纹的**指针**而非内容：GPU 可见性 + 外部指纹文件路径。

    指纹（驱动版本、镜像 digest 等）体积大且含机器信息，落盘在同目录的
    `environment.json` 里；报告只存路径与可见卡，保持报告主体可读。
    """

    model_config = ConfigDict(extra="forbid")

    visible_gpus: list[int]
    gpu_count: int
    fingerprint_path: str


class EvalReport(BaseModel):
    """一次评测的完整产物（落盘为 `eval_report.json`）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVAL_ARTIFACT_SCHEMA_VERSION
    run_id: str
    started_at: str
    finished_at: str
    elapsed_sec: float
    model: EvalModel
    dataset: EvalDataset
    #: 用于重叠分析的训练子集标识；`None` = 本次未做重叠分析（与
    #: `summary.overlap_analysis_performed` 保持一致）。
    train_subset: EvalTrainSubset | None = None
    decoding: EvalDecoding
    criteria: EvalCriteria
    environment: EvalEnvironmentPointer
    summary: EvalSummary
    #: 逐样本明细：报告的全部聚合结论都可由它重算（口径版本见 criteria.version）。
    samples: list[dict[str, Any]]


def utc_now_iso() -> str:
    """当前 UTC 时间戳（ISO 8601，秒精度）。产物时间统一 UTC，避免时区歧义。"""
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


def make_run_id(role: str) -> str:
    """生成运行标识：角色 + UTC 时间戳，保证同目录多次运行不互相覆盖。"""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{role}-{stamp}"
