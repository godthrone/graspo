"""导出流程骨架（模板方法）与钩子契约。

**职责**：定死一次导出的**顺序**（规划 → 校验规划 → 加载 → 校验分片 → 组装 →
落点防呆 → 落盘 → 回读自校验 → 写产物清单），并统一实现安全关键的**拓扑完备性 /
DP 副本唯一性**判定与**回读自校验**编排。

**不负责**：不知道任何具体形态；不自己做逐键比对（在 :mod:`graspo.export.verify`）；
:meth:`Exporter._verify` 只做**编排与委托**，⛔ 不复制比对逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from graspo.export import paths, verify
from graspo.export.errors import ExportError
from graspo.export.schema import (
    ArtifactManifest,
    CheckpointClassification,
    CheckpointForm,
    ExportResult,
    ExportTarget,
    ReplicaCheck,
    ShardPlan,
    ShardRef,
)

#: 产物清单文件名（生成物，⛔ 禁止手工编辑）。
MANIFEST_NAME = "export_manifest.json"

#: 目标产物必需的落盘文件（V14）。
_REQUIRED_FILES: dict[ExportTarget, tuple[str, ...]] = {
    ExportTarget.MERGED_HF: ("config.json",),
    ExportTarget.PEFT_ADAPTER: ("adapter_config.json", "adapter_model.safetensors"),
}


@dataclass(slots=True)
class LoadedShards:
    """``_load`` 的返回值：已加载的 payload、分片身份（basename+sha256）与磁盘路径。

    ``paths`` 只用于内存内定位；**产物清单只消费** ``shard_refs``（§10.1/§16.2）。
    """

    payloads: list[dict[str, Any]]
    shard_refs: list[ShardRef]
    paths: list[Path] = field(default_factory=list)


class Exporter(ABC):
    """一种 checkpoint 形态 → 一种目标产物的导出器。

    公开方法 :meth:`export` 定死流程顺序，子类**不得**覆盖（覆盖即 `TypeError`）；
    子类只实现 ``_`` 前缀钩子。
    """

    #: 本导出器负责的形态（注册键，§9.3）。
    form: ClassVar[CheckpointForm]
    #: 本导出器支持的目标产物。
    supported_targets: ClassVar[frozenset[ExportTarget]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "export" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override Exporter.export — the template method "
                "defines the mandatory export order; implement the underscore hooks instead"
            )

    # ── 公开模板方法：定顺序，子类不得覆盖 ─────────────────────────────────
    def export(
        self,
        config: Any,
        classification: CheckpointClassification,
        *,
        base_model_path: str | None = None,
    ) -> ExportResult:
        """流程骨架：规划 → 校验规划 → 加载 → 校验分片 → 组装 → 落点防呆 → 落盘 →
        回读自校验 → 写产物清单。

        ``base_model_path`` 是基座路径的**唯一入口**（V10）：由调用方从
        ``config.model.model_path`` 传入，⛔ 不回落读 checkpoint 内嵌 config。
        """
        plan = self._plan(config, classification, base_model_path=base_model_path)
        self._assert_plan_is_sound(plan)

        # "已是可服务模型"形态的显式回执（无写盘、无合并）：由基类控制流程，
        # 子类只声明意图（⛔ 子类仍不得覆盖 export）。
        if self._is_passthrough(plan):
            return self._passthrough_result(plan, classification)

        loaded = self._load(plan)
        replica_check = self._assert_shards_are_sound(plan, loaded)
        plan = plan.model_copy(
            update={
                "replica_check": replica_check,
                "consumed_shards": list(loaded.shard_refs),
            }
        )

        assembled = self._assemble(loaded, plan)

        # 落点防呆在**任何写盘之前**（V12），准备动作在 `_write` 之前（V11）。
        destination = self._prepare_destination(plan, config, classification)

        written = self._write(destination, assembled, plan)
        manifest = self._verify(destination, plan, assembled, written)

        # 只有全部自校验通过后才落产物清单（V15：失败不留"半成品 + 成功记录"）。
        manifest_path = destination / MANIFEST_NAME
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")

        return ExportResult(
            destination=str(destination),
            target=plan.target,
            manifest_path=str(manifest_path),
            produced_files=list(written),
            consumed_shards=list(plan.consumed_shards),
        )

    # ── 形态检测钩子（纯 FS，禁 import torch）───────────────────────────
    @classmethod
    @abstractmethod
    def _matches(cls, root: Path) -> tuple[bool, tuple[str, ...]]:
        """本导出器是否认这个目录，并给出**命中证据**（逐条，便于人工复核）。"""

    # ── 规划与校验 ─────────────────────────────────────────────────────
    @abstractmethod
    def _plan(
        self,
        config: Any,
        classification: CheckpointClassification,
        *,
        base_model_path: str | None = None,
    ) -> ShardPlan:
        """算出本次要消费什么（可读 JSON/manifest，禁 import torch）。"""

    def _assert_plan_is_sound(self, plan: ShardPlan) -> None:
        """执行前的规划级校验（默认：目标必须被本导出器支持）。

        ``plan.topology is None``（非 native 形态）时不做拓扑检查。
        """
        if plan.target not in self.supported_targets:
            options = ", ".join(sorted(item.value for item in self.supported_targets)) or "(none)"
            raise ExportError(
                f"{type(self).__name__} does not support target {plan.target.value!r}; "
                f"supported: {options}"
            )
        if plan.topology is not None:
            for name, size in (
                ("dp_size", plan.topology.dp_size),
                ("tp_size", plan.topology.tp_size),
                ("pp_size", plan.topology.pp_size),
            ):
                if int(size) < 1:
                    raise ExportError(
                        f"manifest {name}={size} is invalid for {plan.checkpoint_name}; "
                        "topology must be positive (topology truth source = manifest.json)"
                    )

    # ── 设施钩子（允许延迟 import torch/peft/transformers）──────────────
    def _is_passthrough(self, plan: ShardPlan) -> bool:
        """该 (形态, 目标) 是否**无需写盘**（默认否）。

        F1（HF 基座 → merged-hf）是唯一的 true：产物就是它自己，导出退化为
        "显式回执"（§1.3）。它必须走基类流程，⛔ 不代表子类可以覆盖 ``export``。
        """
        return False

    def _passthrough_result(
        self, plan: ShardPlan, classification: CheckpointClassification
    ) -> ExportResult:
        """``_is_passthrough`` 为真时的回执（默认：未实现 ⇒ fail-closed）。"""
        raise ExportError(
            f"{type(self).__name__} declares a passthrough for {plan.target.value} but does "
            "not implement _passthrough_result (fail-closed)"
        )

    @abstractmethod
    def _load(self, plan: ShardPlan) -> LoadedShards:
        """读分片/基座到内存。唯一允许 import 重型依赖的钩子之一。"""

    @abstractmethod
    def _assemble(self, loaded: LoadedShards, plan: ShardPlan) -> Any:
        """纯张量变换：拼装/合并出待落盘的张量集合。无 IO 副作用。"""

    @abstractmethod
    def _write(self, destination: Path, assembled: Any, plan: ShardPlan) -> tuple[str, ...]:
        """落盘产物文件，返回写出文件的相对名列表。唯一允许写盘的钩子。"""

    # ── 分片级安全校验（基类统一实现，禁止各形态各写一份）────────────────
    def _replica_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        """DP 副本比对的可比对象（默认 = native ``lora_state_dict``）。"""
        state = payload.get("lora_state_dict")
        return dict(state) if isinstance(state, dict) else {}

    def _assert_shards_are_sound(
        self, plan: ShardPlan, loaded: LoadedShards
    ) -> ReplicaCheck | None:
        """分片级 fail-closed 校验（默认实现）：

        - **V5 分片完备性**：实际 ``(dp,tp,pp)`` 三元组集合 == 全笛卡尔积，不重不漏，
          且分片数 == ``dp*tp*pp``；
        - **V7 DP 副本唯一性**：``dp>1`` 时逐键比对（键集合 + 数值 +
          非张量元数据）；一致 ⇒ 通过并留证，任何分歧 ⇒ fail-closed。
          ★ 数值谓词是 ``Tensor.equal``（位模式等价：``-0.0 == +0.0`` 为真、``NaN != NaN``），
          语义边界见 :func:`graspo.export.verify.states_identical` 的说明。

        Returns:
            ``dp>1`` 时返回 :class:`ReplicaCheck` 证据；否则 ``None``。
        """
        topology = plan.topology
        if topology is None:
            return None

        triples = [(ref.dp_rank, ref.tp_rank, ref.pp_rank) for ref in loaded.shard_refs]
        expected = {
            (dp, tp, pp)
            for dp in range(topology.dp_size)
            for tp in range(topology.tp_size)
            for pp in range(topology.pp_size)
        }
        seen: set[tuple[int, int, int]] = set()
        duplicates: list[tuple[int, int, int]] = []
        for triple in triples:
            if triple in seen:
                duplicates.append(triple)
            seen.add(triple)
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        if duplicates or missing or unexpected or len(triples) != len(expected):
            raise ExportError(
                f"shard set is incomplete or inconsistent for {plan.checkpoint_name}: "
                f"expected {len(expected)} shards = "
                f"dp({topology.dp_size}) x tp({topology.tp_size}) x pp({topology.pp_size}), "
                f"found {len(triples)}; missing={missing[:5]} "
                f"unexpected={unexpected[:5]} duplicates={duplicates[:5]} — refusing to merge "
                "(topology truth source = manifest.json; triples read from payload fields)"
            )

        return self._check_replica_uniqueness(plan, loaded)

    def _check_replica_uniqueness(
        self, plan: ShardPlan, loaded: LoadedShards
    ) -> ReplicaCheck | None:
        topology = plan.topology
        assert topology is not None
        if topology.dp_size <= 1:
            return None

        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for ref, payload in zip(loaded.shard_refs, loaded.payloads, strict=True):
            groups.setdefault((ref.tp_rank, ref.pp_rank), []).append(payload)

        evidence: list[str] = []
        compared_keys = 0
        for position in sorted(groups):
            replicas = groups[position]
            if len(replicas) != topology.dp_size:
                raise ExportError(
                    f"DP replica count mismatch at (tp,pp)={position} for "
                    f"{plan.checkpoint_name}: expected {topology.dp_size}, found {len(replicas)}"
                )
            reference = self._replica_state(replicas[0])
            reference_meta = verify.canonical_json(replicas[0].get("lora_tensor_metadata"))
            for index, replica in enumerate(replicas[1:], start=1):
                candidate = self._replica_state(replica)
                identical, detail, compared = verify.states_identical(reference, candidate)
                compared_keys = compared
                if not identical:
                    raise ExportError(
                        f"DP replicas at (tp,pp)={position} are NOT identical for "
                        f"{plan.checkpoint_name}: replica[0] vs replica[{index}] diverge at "
                        f"{detail}. Each DP rank is effectively a different model (distinct LoRA "
                        "initialisation), so there is no unique correct result; silently picking "
                        "one replica is forbidden (P1). Refusing to export — fix the training-side "
                        "seed synchronisation or export a dp=1 checkpoint."
                    )
                if verify.canonical_json(replica.get("lora_tensor_metadata")) != reference_meta:
                    raise ExportError(
                        f"DP replicas at (tp,pp)={position} have different lora_tensor_metadata "
                        f"for {plan.checkpoint_name}; refusing to export (fail-closed)."
                    )
            evidence.append(
                f"(tp,pp)={position}: {topology.dp_size} DP replicas identical over "
                f"{len(reference)} keys (digest={verify.state_digest(reference)[:16]})"
            )

        return ReplicaCheck(
            dp_size=topology.dp_size,
            identical=True,
            compared_keys=compared_keys,
            evidence=evidence,
        )

    # ── 落点防呆 ───────────────────────────────────────────────────────
    def _prepare_destination(
        self,
        plan: ShardPlan,
        config: Any,
        classification: CheckpointClassification,
    ) -> Path:
        """V12 重叠校验 → V11 非空防呆（预授权后逐项删除）。"""
        output = getattr(config, "export_output", "") or ""
        if not output:
            raise ExportError(
                "export.export_output is required (output location comes from config)"
            )
        paths.assert_destination_allowed(
            output,
            checkpoint_dir=classification.path,
            base_model_path=plan.base_model_path,
        )
        overwrite = bool(getattr(config, "allow_overwrite", False))
        return paths.prepare_output_dir(output, overwrite=overwrite)

    # ── 自校验：基类统一实现，子类只补差异 ─────────────────────────────
    def _required_files(self, target: ExportTarget) -> tuple[str, ...]:
        return _REQUIRED_FILES.get(target, ())

    def _verify_produced_state(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ) -> int:
        """回读产物的形态钩子：返回产物张量**键数**，自校验失败即抛 ``ExportError``。

        默认实现不做回读（返回 0）；具体形态在 ``exporters/*`` 覆盖。
        """
        return 0

    def _verify_extra(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ) -> None:
        """可选的额外校验（默认空实现）。"""

    def _verify(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ) -> ArtifactManifest:
        """回读产物 → 必需文件齐备 → 形态级键集合守恒/有限性 → 生成 :class:`ArtifactManifest`。

        校验不通过 ⇒ 抛 ``ExportError``，**且不得留下** ``export_manifest.json``
        （清单由 :meth:`export` 在本方法成功之后才写）。
        """
        for name in self._required_files(plan.target):
            if not (destination / name).exists():
                raise ExportError(
                    f"export self-check failed: required artifact {name!r} is missing from "
                    f"{destination} (target={plan.target.value})"
                )
        produced_keys = int(self._verify_produced_state(destination, plan, assembled, written))
        self._verify_extra(destination, plan, assembled, written)

        base_name = Path(plan.base_model_path).name if plan.base_model_path else None
        return ArtifactManifest(
            form=plan.form,
            target=plan.target,
            checkpoint_name=plan.checkpoint_name,
            consumed_shards=list(plan.consumed_shards),
            base_model_name=base_name,
            replica_check=plan.replica_check,
            produced_files=list(written),
            produced_keys=produced_keys,
        )


__all__ = ["Exporter", "LoadedShards", "MANIFEST_NAME"]
