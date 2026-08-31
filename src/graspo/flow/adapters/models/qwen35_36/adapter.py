"""Qwen3.5/3.6 adapter — hybrid attention + visual tower + multimodal.

Class-directory pattern: model loading/construction lives in this file;
generation/training/logprobs methods live in generation.py / training.py /
logprobs.py. External users import only the class name and never see the
internal split.
"""

from pathlib import Path
from typing import Any

from graspo.core.lora import resolve_lora_target_modules
from graspo.flow.adapters.models.common.model_builders import build_native_qwen_model
from graspo.flow.adapters.models.qwen35_36.generation import _Qwen35GenerationMethods
from graspo.flow.adapters.models.qwen35_36.logprobs import _Qwen35LogprobsMethods
from graspo.flow.adapters.models.qwen35_36.pipeline_forward import _Qwen35PipelineForwardMethods
from graspo.flow.adapters.models.qwen35_36.training import _Qwen35TrainingMethods
from graspo.flow.adapters.models.qwen35_36.training_sft import _Qwen35SFTTrainingMethods
from graspo.flow.adapters.transformer_adapter import TransformerAdapter
from graspo.flow.lora.lora_helpers import native_qwen_lora_available_targets
from graspo.flow.lora.lora_io import load_peft_adapter_into_native_model
from graspo.flow.parallel.placement_plan import (
    build_placement_plan,
)
from graspo.flow.parallel.tensor_utils import (
    SafetensorIndex,
    _resolve_dtype,
)
from graspo.ripple.parsing.completion import ParsedCompletion
from graspo.ripple.parsing.qwen_tool_parser import parse_qwen_tool_completion


class Qwen35Adapter(  # type: ignore[misc]  # mixin 组合点的多基类签名兼容检查是 mypy 已知误报区；各 mixin 签名已与 ABC 对齐（keyword-only + **kwargs），运行时由测试验证
    _Qwen35GenerationMethods,
    _Qwen35TrainingMethods,
    _Qwen35SFTTrainingMethods,
    _Qwen35LogprobsMethods,
    _Qwen35PipelineForwardMethods,
    TransformerAdapter,
):
    """Qwen3.5/3.6 adapter for GraspoFlow.

    Supports hybrid attention, visual tower, multimodal rollout, and
    TP-only / PP / TP+PP training.

    使用 mixin 组合：_Qwen35GenerationMethods（生成/rollout）、
    _Qwen35TrainingMethods（RL 训练/优化）、
    _Qwen35SFTTrainingMethods（SFT 训练）、
    _Qwen35LogprobsMethods（log 概率）。
    """

    completion_parser_name = "qwen_tool_call"

    def _load_model(self, hf_config: Any, model_path: Path) -> None:
        torch_dtype = _resolve_dtype(self.config.model.torch_dtype)
        loader = SafetensorIndex(model_path)
        lora_targets = resolve_lora_target_modules(
            self.config.lora.target_modules or (self.config.lora.target_preset,),
            available=native_qwen_lora_available_targets(hf_config),
        )
        self.placement = build_placement_plan(
            strategy=self.config.graspoflow.placement_strategy,
            model_family=hf_config.family,
            num_hidden_layers=int(hf_config.num_hidden_layers),
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            layer_types=list(getattr(hf_config, "layer_types", []) or []),
            manual_ranges=[list(r) for r in self.config.graspoflow.layer_ranges]
            if self.config.graspoflow.layer_ranges is not None
            else None,
        )
        self.model = build_native_qwen_model(
            hf_config=hf_config,
            loader=loader,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            use_sp=bool(self.config.graspoflow.sequence_parallel),
            placement=self.placement,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            lora_targets=set(lora_targets.resolved),
            gradient_checkpointing=bool(self.config.model.gradient_checkpointing),
            torch_dtype=torch_dtype,
            device=self.device,
        )
        if self.model is None:
            raise RuntimeError("model not loaded; call setup() first")
        missing_lora_targets = sorted(
            target
            for target in set(lora_targets.resolved) - set(self.model.enabled_lora_target_names())
            if not (target.startswith("visual.") and getattr(self.model, "visual", None) is None)
        )
        if missing_lora_targets:
            raise ValueError(
                "Resolved LoRA target(s) are not implemented by this model yet: "
                + ", ".join(missing_lora_targets)
            )
        self.model.train(False)
        if self.config.lora.adapter_path:
            load_peft_adapter_into_native_model(
                self.model,
                self.config.lora.adapter_path,
                base_model_path=str(model_path),
            )

    def parse_completion(self, completion: str, sample: Any | None = None) -> ParsedCompletion:
        return parse_qwen_tool_completion(
            completion,
            expect_tool_calls=bool(getattr(sample, "expects_tool_calls", False)),
            tools=getattr(sample, "tools", None),
        )

    def _build_ops(self) -> None:
        """Build pipeline operators (RL training).  SFT path is a no-op."""
        pass
