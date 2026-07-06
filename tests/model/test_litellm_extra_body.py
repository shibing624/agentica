# -*- coding: utf-8 -*-
"""
Tests for LiteLLMChat.extra_body field.

`extra_body` is a vendor-specific parameter passthrough (e.g. vLLM
`top_k_per_token`, Together `repetition_penalty`). Explicit dataclass fields
must always win over `extra_body` to avoid silent overrides.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

try:
    from litellm import completion as _litellm_completion  # noqa: F401
    _LITELLM_OK = True
except Exception:
    _LITELLM_OK = False


@unittest.skipUnless(_LITELLM_OK, "litellm not installed")
class TestLiteLLMExtraBody(unittest.TestCase):
    def test_extra_body_passes_through(self):
        from agentica.model.litellm.chat import LiteLLMChat

        model = LiteLLMChat(
            id="vllm/foo",
            extra_body={"top_k_per_token": 5, "repetition_penalty": 1.05},
        )
        params = model.request_kwargs

        self.assertEqual(params.get("top_k_per_token"), 5)
        self.assertEqual(params.get("repetition_penalty"), 1.05)

    def test_explicit_field_wins_over_extra_body(self):
        from agentica.model.litellm.chat import LiteLLMChat

        model = LiteLLMChat(
            id="anthropic/claude-x",
            reasoning_effort="high",
            extra_body={"reasoning_effort": "low", "novel_param": "ok"},
        )
        params = model.request_kwargs

        self.assertEqual(params["reasoning_effort"], "high")
        self.assertEqual(params["novel_param"], "ok")

    def test_extra_body_none_is_noop(self):
        from agentica.model.litellm.chat import LiteLLMChat

        model = LiteLLMChat(id="openai/gpt-x")
        params = model.request_kwargs
        self.assertNotIn("top_k_per_token", params)


if __name__ == "__main__":
    unittest.main()
