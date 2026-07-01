"""Qwen3.5/3.6 adapter — hybrid attention + visual tower + multimodal.

采用类改目录模式（宪法 8.3）：模型加载/构建驻留在本文件，
生成/训练/logprobs 方法分布在 generation.py / training.py / logprobs.py 中。
外部使用者只 import 类名，完全不感知内部拆分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graspo.backends.graspoflow.lora_helpers import native_qwen_lora_available_targets
from graspo.backends.graspoflow.lora_io import load_peft_adapter_into_native_model
from graspo.backends.graspoflow.models.qwen3.model import (
    build_native_qwen_model,
)
from graspo.backends.graspoflow.models.qwen35_36.generation import _Qwen35GenerationMethods
from graspo.backends.graspoflow.models.qwen35_36.logprobs import _Qwen35LogprobsMethods
from graspo.backends.graspoflow.models.qwen35_36.ops import build_qwen35_ops
from graspo.backends.graspoflow.models.qwen35_36.training import _Qwen35TrainingMethods
from graspo.backends.graspoflow.placement import (
    build_placement_plan,
)
from graspo.backends.graspoflow.tensor_utils import (
    SafetensorIndex,
    _resolve_dtype,
)
from graspo.backends.graspoflow.tool_parser import parse_qwen_tool_completion
from graspo.backends.graspoflow.transformer_adapter import TransformerAdapter
from graspo.core.completion import ParsedCompletion
from graspo.trainer.lora import resolve_lora_target_modules


class Qwen35Adapter(
    _Qwen35GenerationMethods,
    _Qwen35TrainingMethods,
    _Qwen35LogprobsMethods,
    TransformerAdapter,
):
    """Qwen3.5/3.6 adapter for GraspoFlow.

    Supports hybrid attention, visual tower, multimodal rollout, and
    TP-only / PP / TP+PP training.

    使用 mixin 组合：_Qwen35GenerationMethods（生成/rollout）、
    _Qwen35TrainingMethods（训练/优化）、_Qwen35LogprobsMethods（log 概率）。
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
            placement=self.placement,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            lora_targets=set(lora_targets.resolved),
            gradient_checkpointing=bool(self.config.model.gradient_checkpointing),
            torch_dtype=torch_dtype,
            device=self.device,
        )
        assert self.model is not None
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

    def _build_ops(self) -> None:
        self._ops = build_qwen35_ops(
            model=self.model,
            tp_state=self.tp_state,
            tp_size=self.tp_size,
        )

    def parse_completion(self, completion: str, sample: Any | None = None) -> ParsedCompletion:
        return parse_qwen_tool_completion(
            completion,
            expect_tool_calls=bool(getattr(sample, "expects_tool_calls", False)),
            tools=getattr(sample, "tools", None),
        )
