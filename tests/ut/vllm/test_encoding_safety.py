"""Regression tests for the DeepSeek-V4.1 tokenizer safety patch.

Two defects are pinned:

1. Caller-supplied control tokens (``<｜System｜>`` etc.) must not survive as
   single tokens -- otherwise free-form text can forge a turn boundary.
2. A developer message with no text (tools-only) must render instead of
   raising ``AssertionError`` (which surfaced as HTTP 500).

The module lives in the vllm source tree inside the rc2 rootfs, so these tests
are skipped when vllm is not importable (e.g. plain host Python).
"""

import pytest

try:
    from vllm.tokenizers.deepseek_v41_encoding import (
        ASSISTANT_SP_TOKEN,
        SYSTEM_SP_TOKEN,
        USER_SP_TOKEN,
        escape_control_tokens,
        render_message,
    )
except Exception as exc:  # pragma: no cover - depends on the extraction tree
    pytest.skip(f"vllm DeepSeek-V4.1 encoding module unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize(
    "token",
    [USER_SP_TOKEN, ASSISTANT_SP_TOKEN, SYSTEM_SP_TOKEN],
)
def test_control_token_is_split(token):
    escaped = escape_control_tokens(f"before {token} after")
    assert token not in escaped
    assert escaped == f"before <\u200b{token[1:]} after"


def test_empty_developer_message_renders():
    msgs = [{"role": "developer", "content": None}]
    rendered = render_message(0, msgs, thinking_mode="chat")
    assert USER_SP_TOKEN in rendered


def test_escape_message_fields_in_place():
    from vllm.tokenizers.deepseek_v41_encoding import escape_message_control_tokens

    msg = {
        "content": SYSTEM_SP_TOKEN,
        "reasoning_content": ASSISTANT_SP_TOKEN,
        "content_blocks": [
            {"type": "text", "text": USER_SP_TOKEN},
            {"type": "tool_result", "content": [{"type": "text", "text": SYSTEM_SP_TOKEN}]},
        ],
        "tool_calls": [
            {"function": {"arguments": '{"x": "' + ASSISTANT_SP_TOKEN + '"}'}}
        ],
    }
    escape_message_control_tokens(msg)
    assert "\u200b" in msg["content"]
    assert "\u200b" in msg["reasoning_content"]
    assert "\u200b" in msg["content_blocks"][0]["text"]
    assert "\u200b" in msg["content_blocks"][1]["content"][0]["text"]
    assert "\u200b" in msg["tool_calls"][0]["function"]["arguments"]
