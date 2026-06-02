import pytest


torch = pytest.importorskip("torch")

from graspo.trainer.lora import detect_lora_target_modules  # noqa: E402


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)


def test_detect_lora_targets():
    targets = detect_lora_target_modules(TinyAttention())

    assert targets == ["k_proj", "o_proj", "q_proj", "v_proj"]


def test_detect_lora_targets_fails_clearly():
    with pytest.raises(ValueError, match="Could not auto-detect"):
        detect_lora_target_modules(torch.nn.Sequential(torch.nn.ReLU()))
