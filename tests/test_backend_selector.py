from __future__ import annotations

import pytest

from graspo.backends.selector import select_backend
from graspo.core.schema import GraspoConfig


def test_backend_selection_is_native_tp_only():
    selection = select_backend(GraspoConfig())

    assert selection.name == "native-tp"


@pytest.mark.parametrize("backend", ["auto", "hf-reference", "megatron-vllm"])
def test_backend_rejects_removed_names(backend):
    config = GraspoConfig()

    with pytest.raises(ValueError, match="only native-tp"):
        select_backend(config, requested=backend)
