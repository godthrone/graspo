from __future__ import annotations


class QwenCheckpointMixin:
    """Checkpoint save/load methods for Qwen-family adapters.

    Requires the host class to provide ``self.model``, ``self.optimizer``,
    ``self.config``, ``self.placement``, and ``self._print_rank0``.
    """

    # Populated by extracting methods from QwenNativeTPAdapter.
    # Currently implemented directly in adapter.py; methods will be moved here
    # incrementally in follow-up PRs.
    pass
