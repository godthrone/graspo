"""Layer 2 — BaseGraspoFlowAdapter: abstract protocol for all model adapters.

Every model-family adapter must implement this interface.  The Trainer and
Runtime only see this ABC, so they are completely model-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from graspo.backends.graspoflow.runtime import NativeGeneration
from graspo.core.completion import ParsedCompletion


class BaseGraspoFlowAdapter(ABC):
    """Abstract base for model-specific GraspoFlow adapters.

    Each model family (Qwen3, Qwen3.5/3.6, DeepSeek, …) should subclass this
    and implement all abstract methods.  Common cross-model utilities live in
    ``TransformerAdapter``.
    """

    # ── Subclass responsibility ─────────────────────────────────────────────

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
        """Generate *rollout_group_size* completions per message batch.

        Subclasses may accept additional keyword arguments via ``**kwargs``:
        ``max_prompt_length``, ``temperature``, ``top_p``, ``samples``,
        ``tool_batches``, and backend-specific generation parameters.
        """
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
        """Generate completions for multimodal :class:`Sample` objects.

        Subclasses may accept additional keyword arguments via ``**kwargs``:
        ``max_prompt_length``, ``temperature``, ``top_p``,
        and backend-specific multimodal generation parameters.
        """
        ...

    @abstractmethod
    def sequence_log_probs(
        self,
        sequences: list[list[int]] | torch.Tensor,
        attention_mask: list[list[int]] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return per-token log-probabilities under the current policy.

        Subclasses may accept additional keyword arguments via ``**kwargs``:
        ``metadata`` (multimodal context), ``multimodal_inputs``, and
        backend-specific forwarding parameters.
        """
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

        Subclasses may accept additional keyword arguments via ``**kwargs``:
        ``policy_ratio_clip_eps``, ``optimize_iterations_per_step``,
        ``max_grad_norm``, ``sft_batches`` (for SFT mode),
        and backend-specific training parameters.
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist a recoverable training checkpoint."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> dict[str, Any] | None:
        """Restore trainer state from a checkpoint."""
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
