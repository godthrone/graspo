from __future__ import annotations

import pytest

from graspo.backends.selector import select_backend
from graspo.core.schema import GraspoConfig


def test_backend_selection_defaults_to_graspoflow():
    selection = select_backend(GraspoConfig())

    assert selection.name == "graspoflow"


@pytest.mark.parametrize("backend", ["auto", "hf-reference", "megatron-vllm", "native-tp"])
def test_backend_rejects_removed_names(backend):
    config = GraspoConfig()

    with pytest.raises(ValueError, match="only supports 'graspoflow'"):
        select_backend(config, requested=backend)
