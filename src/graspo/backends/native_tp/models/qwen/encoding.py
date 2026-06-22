from __future__ import annotations


class QwenEncodingMixin:
    """Tokenization and multimodal-encoding methods for Qwen-family adapters.

    Requires the host class to provide ``self.tokenizer``, ``self.processor``,
    ``self.config``, and ``self.device``.
    """

    # Populated by extracting methods from QwenNativeTPAdapter.
    # Currently implemented directly in adapter.py; methods will be moved here
    # incrementally in follow-up PRs.
    pass
