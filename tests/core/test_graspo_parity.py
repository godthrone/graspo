import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from graspo.core.graspo_parity import is_uniform_partial_content, lower_median  # noqa: E402


def test_lower_median_matches_torch_median_for_even_group():
    values = [0.0, 0.2, 0.4, 1.0]

    assert lower_median(values) == pytest.approx(torch.tensor(values).median().item())
    assert lower_median(values) == 0.2


def test_uniform_partial_content_matches_original_invalid_filter():
    assert is_uniform_partial_content([0.5, 0.5, 0.5])
    assert not is_uniform_partial_content([0.0, 0.0, 0.0])
    assert not is_uniform_partial_content([1.0, 1.0, 1.0])
    assert not is_uniform_partial_content([0.25, 0.5, 0.25])
