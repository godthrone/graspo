from graspo.core.chat_template import render_messages


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
