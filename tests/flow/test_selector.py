"""后端选择器（graspo.flow.selector）的单元测试。"""

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.selector import select_backend


def test_backend_selection_defaults_to_graspoflow():
    selection = select_backend(GraspoConfig())

    assert selection.name == "graspoflow"


@pytest.mark.parametrize("backend", ["auto", "hf-reference", "megatron-vllm", "native-tp"])
def test_backend_rejects_removed_names(backend):
    config = GraspoConfig()

    with pytest.raises(ValueError, match="only supports 'graspoflow'"):
        select_backend(config, requested=backend)
