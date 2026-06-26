from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import torch

from graspo.core.completion import ParsedCompletion
from graspo.backends.native_tp.runtime import NativeGeneration


class BaseNativeTPAdapter(ABC):
    """Abstract base for model-specific native tensor-parallel adapters.

    Each model family (Qwen, Llama, etc.) should subclass this and implement
    all abstract methods.  Common cross-model utilities live here so they
    are available to every adapter without code duplication.

    The ``NativeTPRuntime`` protocol already defines the public interface that
    the trainer calls.  This ABC mirrors that contract and adds shared
    concrete helpers.
    """

    # ------------------------------------------------------------------
    # Subclass responsibility
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Initialise distributed state, load model, build LoRA, create optimizer."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Tear down distributed state and free GPU resources."""
        ...

    @abstractmethod
    def is_primary(self) -> bool:
        """Return ``True`` on the rank that should perform logging / I/O."""
        ...

    @abstractmethod
    def generate_groups(
        self,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None],
        *,
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[NativeGeneration]:
        """Generate *rollout_group_size* completions per message batch."""
        ...

    @abstractmethod
    def generate_sample_groups(
        self,
        samples: list[Any],
        *,
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[NativeGeneration]:
        """Generate completions for multimodal :class:`Sample` objects."""
        ...

    @abstractmethod
    def sequence_log_probs(
        self,
        sequences: list[list[int]] | torch.Tensor,
        attention_mask: list[list[int]] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return per-token log-probabilities under the current policy."""
        ...

    @abstractmethod
    def train_batch(
        self,
        experiences: list[Any],
        *,
        optimizer_steps: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one or more optimizer steps over a batch of experiences.

        Returns
        -------
        dict
            Training metrics (loss, grad-norm, learning-rate, ...).
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist a recoverable training checkpoint."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> dict[str, Any] | None:
        """Restore trainer state from a native checkpoint."""
        ...

    @abstractmethod
    def parse_completion(self, completion: str, sample: Any | None = None) -> ParsedCompletion:
        """Convert raw model output into a canonical :class:`ParsedCompletion`."""
        ...

    @abstractmethod
    def format_messages(
        self,
        messages: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply the model's chat-template to *messages* and return the prompt string."""
        ...

    # ------------------------------------------------------------------
    # Shared concrete helpers
    # ------------------------------------------------------------------

    def _require_ready(self) -> None:
        if getattr(self, "model", None) is None or getattr(self, "tokenizer", None) is None:
            raise RuntimeError(f"{type(self).__name__} is not set up")

    def _print_rank0(self, payload: dict[str, Any]) -> None:
        if self.is_primary():
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _sync_timing(self) -> None:
        cfg = getattr(self, "config", None)
        dev = getattr(self, "device", None)
        if (
            cfg is not None
            and bool(getattr(cfg, "native_tp", None) and cfg.native_tp.synchronize_cuda_timing)
            and dev is not None
            and dev.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(dev)

    def _is_pipeline_parallel(self) -> bool:
        placement = getattr(self, "placement", None)
        return bool(placement is not None and placement.is_pipeline)

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        import torch

        return getattr(torch, name) if isinstance(name, str) else name
