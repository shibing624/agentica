# -*- coding: utf-8 -*-
"""Wire-payload allowlist guards for both provider paths (Reasonix ModelMessages port).

The OpenAI path serialises through ``Message.to_model_dict()`` — an include-list,
so a new local field is safe by default. The Anthropic path builds blocks by
reading ``content``/``images`` attributes directly and passes dict-typed content
blocks through as-is; for it these tests are the only tripwire against a
"dump the whole message" regression. A Message fully loaded with sentinel local
fields must leak none of them through either path.
"""
import asyncio
import json

import pytest

from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message, MessageReferences
from agentica.model.openai.chat import OpenAIChat

# OpenAI top-level allowlist — mirrors Message.to_model_dict()'s include set.
OPENAI_WIRE_KEYS = {"role", "content", "audio", "name", "tool_call_id", "tool_calls", "reasoning_content"}
ANTHROPIC_WIRE_KEYS = {"role", "content"}

# Sentinel strings live exclusively in local-only fields; finding any of them
# in a wire payload means a field escaped its boundary.
SENTINELS = (
    "SENTINEL_METRICS_x7q",
    "SENTINEL_REFERENCES_x7q",
    "SENTINEL_PROVIDER_DATA_x7q",
    "SENTINEL_COMPRESSED_x7q",
    "SENTINEL_THINKING_x7q",
    "SENTINEL_FINISH_x7q",
    "SENTINEL_TOOLNAME_x7q",
    "SENTINEL_TOOLARGS_x7q",
    "SENTINEL_REDACTED_x7q",
)


def _loaded_message(role: str = "user") -> Message:
    """A message whose local-only fields all carry unique sentinels."""
    return Message(
        role=role,
        content="hello world",
        metrics={"tokens": "SENTINEL_METRICS_x7q"},
        references=MessageReferences(query="SENTINEL_REFERENCES_x7q"),
        provider_data={"blob": "SENTINEL_PROVIDER_DATA_x7q"},
        compressed_content="SENTINEL_COMPRESSED_x7q",
        thinking="SENTINEL_THINKING_x7q",
        redacted_thinking="SENTINEL_REDACTED_x7q",
        finish_reason="SENTINEL_FINISH_x7q",
        tool_name="SENTINEL_TOOLNAME_x7q",
        tool_args={"arg": "SENTINEL_TOOLARGS_x7q"},
    )


def _assert_clean(payload_text: str, allowed_keys: set, wire_dicts: list):
    for sentinel in SENTINELS:
        assert sentinel not in payload_text, f"local field leaked into wire payload: {sentinel}"
    for d in wire_dicts:
        extra = set(d.keys()) - allowed_keys
        assert not extra, f"unexpected wire keys {extra} in {d!r}"


class TestOpenAIWireAllowlist:
    def test_loaded_message_serialises_via_allowlist(self):
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        payload = model.format_message(_loaded_message())
        _assert_clean(json.dumps(payload, ensure_ascii=False), OPENAI_WIRE_KEYS, [payload])

    def test_assistant_message_with_tool_calls_clean(self):
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        msg = _loaded_message(role="assistant")
        msg.tool_calls = [{"id": "call_1", "type": "function",
                           "function": {"name": "lookup_city", "arguments": "{}"}}]
        payload = model.format_message(msg)
        _assert_clean(json.dumps(payload, ensure_ascii=False), OPENAI_WIRE_KEYS, [payload])


class TestAnthropicWireAllowlist:
    def test_loaded_message_clean(self):
        model = Claude(id="claude-sonnet-4-5", api_key="fake_anthropic_key")
        chat_messages, _system = asyncio.run(model.format_messages([_loaded_message()]))
        _assert_clean(json.dumps(chat_messages, ensure_ascii=False), ANTHROPIC_WIRE_KEYS, chat_messages)

    def test_block_typed_content_passthrough_is_intentional(self):
        """Dict content blocks (tool_use/thinking) pass through by design — the
        one surface the allowlist cannot cover. Pin that the passthrough works
        (legit blocks survive) while local fields still stay off the wire."""
        model = Claude(id="claude-sonnet-4-5", api_key="fake_anthropic_key")
        msg = _loaded_message(role="assistant")
        msg.content = [
            {"type": "thinking", "thinking": "legit chain-of-thought", "signature": "sig"},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup_city", "input": {"name": "Beijing"}},
        ]
        chat_messages, _system = asyncio.run(model.format_messages([msg]))
        assert chat_messages, "block-typed assistant message dropped"
        text = json.dumps(chat_messages, ensure_ascii=False)
        assert "legit chain-of-thought" in text, "thinking block must survive (Anthropic wire needs it)"
        assert "toolu_1" in text, "tool_use block must survive"
        for sentinel in SENTINELS:
            assert sentinel not in text, f"local field leaked into wire payload: {sentinel}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
