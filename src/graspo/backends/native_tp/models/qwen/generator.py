from __future__ import annotations


class QwenGeneratorMixin:
    """Generation and rollout methods for Qwen-family adapters.

    Requires the host class to provide ``self.model``, ``self.tokenizer``,
    ``self.config``, ``self.device``, ``self.placement``, and helpers from
    ``QwenEncodingMixin``.
    """

    # Populated by extracting methods from QwenNativeTPAdapter.
    # Currently implemented directly in adapter.py; methods will be moved here
    # incrementally in follow-up PRs.
    pass
