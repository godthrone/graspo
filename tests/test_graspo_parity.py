import pytest


torch = pytest.importorskip("torch")

from graspo.core.graspo_parity import is_uniform_partial_content, lower_median  # noqa: E402
from graspo.trainer.generation import generate_group  # noqa: E402
from graspo.trainer.loss import GRASPOLoss  # noqa: E402


def test_lower_median_matches_torch_median_for_even_group():
    values = [0.0, 0.2, 0.4, 1.0]

    assert lower_median(values) == pytest.approx(torch.tensor(values).median().item())
    assert lower_median(values) == 0.2


def test_uniform_partial_content_matches_original_invalid_filter():
    assert is_uniform_partial_content([0.5, 0.5, 0.5])
    assert not is_uniform_partial_content([0.0, 0.0, 0.0])
    assert not is_uniform_partial_content([1.0, 1.0, 1.0])
    assert not is_uniform_partial_content([0.25, 0.5, 0.25])


def test_ppo_clip_loss_matches_original_formula():
    log_probs = torch.tensor([[0.0, 0.2, -0.1], [0.1, -0.3, 0.4]])
    old_log_probs = torch.tensor([[-0.1, 0.1, -0.1], [0.0, -0.1, 0.2]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-0.5, -0.5, -0.5]])
    action_mask = torch.tensor([[True, True, False], [True, False, True]])

    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(0.8, 1.2) * advantages
    expected = -torch.min(surr1, surr2)
    expected = ((expected * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)).mean()

    actual = GRASPOLoss(policy_ratio_clip_eps=0.2)(log_probs, old_log_probs, advantages, action_mask)

    assert torch.allclose(actual, expected)


class FakeConfig:
    use_cache = False


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()
        self.outputs = [
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.tensor([[1, 2, 3, 6]]),
        ]

    def generate(self, **kwargs):
        return self.outputs.pop(0)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    chat_template = None

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def batch_decode(self, sequences, skip_special_tokens):
        return ["decoded"] * sequences.shape[0]


def test_generate_group_action_mask_tracks_completion_after_padding():
    sequences, attention_mask, action_mask, completions, prompt_len = generate_group(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        prompt="q",
        group_size=2,
        device=torch.device("cpu"),
        max_new_tokens=4,
        max_prompt_length=16,
        temperature=1.0,
        top_p=1.0,
        synced_gpus=False,
        chat_template_kwargs=None,
    )

    assert prompt_len == 3
    assert sequences.tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 6, 0]]
    assert attention_mask.tolist() == [[True, True, True, True, True], [True, True, True, True, False]]
    assert action_mask.tolist() == [[False, False, True, True], [False, False, True, False]]
    assert completions == ["decoded", "decoded"]
