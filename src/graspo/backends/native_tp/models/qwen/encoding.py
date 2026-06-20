from __future__ import annotations

import json
from typing import Any

import torch

from graspo.backends.native_tp.multimodal import (
    _messages_from_multimodal_row,
    _processor_chat_messages,
    _tools_from_multimodal_row,
    _tools_for_chat_template,
)
from graspo.backends.native_tp.models.qwen.config import NativeQwenConfig


class QwenEncodingMixin:
    """Tokenization and multimodal-encoding methods for Qwen-family adapters.

    Requires the host class to provide ``self.tokenizer``, ``self.processor``,
    ``self.config``, and ``self.device``.
    """

    # Populated by extracting methods from QwenNativeTPAdapter.
    # Currently implemented directly in adapter.py; methods will be moved here
    # incrementally in follow-up PRs.
    pass
