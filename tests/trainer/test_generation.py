import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from graspo.trainer.generation import generate_group, render_messages  # noqa: E402


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


class FakeChatTokenizer:
    chat_template = "template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def test_render_messages_preserves_multiturn_messages_and_template_kwargs():
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]

    rendered = render_messages(
        tokenizer,
        messages,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert rendered == "rendered"
    assert tokenizer.calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]


def test_generate_group_action_mask_tracks_completion_after_padding():
    sequences, attention_mask, action_mask, completions, prompt_len = generate_group(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        messages=[{"role": "user", "content": "q"}],
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
    assert attention_mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, True, False],
    ]
    assert action_mask.tolist() == [[False, False, True, True], [False, False, True, False]]
    assert completions == ["decoded", "decoded"]
