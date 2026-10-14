# -*- coding: utf-8 -*-
"""Output-token cap: user config > catalog ceiling > gateway default.

Without a cap in the request the gateway picks one: 20000 on Venus for
claude-opus-5-5, which a high-effort thinking turn spent entirely on reasoning
(delegate a034fa4c died there). The catalog ceiling is models.dev
``limit.output``; it describes the model, not the deployment, so a deployment
that rejects it must not lose the request.
"""
import asyncio
import unittest
from unittest import mock

import httpx
import openai

from agentica.model.cost import _parse_catalog
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.openai.responses import OpenAIResponses
from agentica.runner import Runner
from agentica.model.loop_state import LoopState

CEILING = "agentica.model.openai.chat.get_model_max_output_tokens"


def _bad_request(text: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "http://gateway/v1/chat/completions"))
    return openai.BadRequestError(text, response=response, body=None)


class TestCatalogKeepsOutputLimit(unittest.TestCase):
    def test_limit_output_is_parsed(self):
        catalog = {
            "anthropic": {
                "models": {
                    "claude-opus-5-5": {
                        "cost": {"input": 4, "output": 20},
                        "limit": {"context": 1000000, "output": 128000},
                    }
                }
            }
        }
        entry = _parse_catalog(catalog)["claude-opus-5-5"]
        self.assertEqual(entry["max_output_tokens"], 128000)
        self.assertEqual(entry["context_window"], 1000000)


class TestDefaultCap(unittest.TestCase):
    def test_catalog_ceiling_used_when_nothing_configured(self):
        with mock.patch(CEILING, return_value=128000):
            model = OpenAIChat(id="claude-opus-5-5", api_key="k")
        self.assertEqual(model.request_kwargs["max_tokens"], 128000)

    def test_user_max_tokens_wins(self):
        with mock.patch(CEILING, return_value=128000):
            model = OpenAIChat(id="claude-opus-5-5", api_key="k", max_tokens=8000)
        self.assertEqual(model.request_kwargs["max_tokens"], 8000)

    def test_user_max_completion_tokens_is_not_doubled_up(self):
        with mock.patch(CEILING, return_value=128000):
            model = OpenAIChat(id="gpt-5.5", api_key="k", max_completion_tokens=4096)
        kwargs = model.request_kwargs
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["max_completion_tokens"], 4096)

    def test_unknown_model_sends_no_cap(self):
        with mock.patch(CEILING, return_value=None):
            model = OpenAIChat(id="in-house-model", api_key="k")
        self.assertNotIn("max_tokens", model.request_kwargs)

    def test_responses_wire_uses_max_output_tokens(self):
        with mock.patch(CEILING, return_value=128000):
            model = OpenAIResponses(id="gpt-5.5", api_key="k")
        self.assertEqual(model.request_kwargs["max_output_tokens"], 128000)


class TestRejectedCatalogCap(unittest.TestCase):
    """A deployment smaller than the model (vLLM ``max_model_len=163840``)."""

    def _model(self, **kw):
        with mock.patch(CEILING, return_value=393216):
            return OpenAIChat(id="DeepSeek-V4-Flash", api_key="k", **kw)

    def _invoke(self, model, side_effect):
        create = mock.AsyncMock(side_effect=side_effect)
        client = mock.Mock()
        client.chat.completions.create = create
        with mock.patch.object(model, "get_client", return_value=client):
            result = asyncio.run(model.invoke([Message(role="user", content="hi")]))
        return result, create

    def test_rejected_catalog_cap_is_dropped_and_request_resent(self):
        model = self._model()
        ok = object()
        result, create = self._invoke(model, [_bad_request("max_tokens=393216 cannot be greater than 163840"), ok])
        self.assertIs(result, ok)
        self.assertEqual(create.await_args_list[0].kwargs["max_tokens"], 393216)
        self.assertNotIn("max_tokens", create.await_args_list[1].kwargs)
        # Learned for the rest of the process: no more doomed first attempts.
        self.assertNotIn("max_tokens", model.request_kwargs)

    def test_unrelated_400_keeps_the_cap_and_surfaces_the_error(self):
        model = self._model()
        with self.assertRaises(openai.BadRequestError):
            self._invoke(model, [_bad_request("message has no content"), _bad_request("message has no content")])
        self.assertEqual(model.request_kwargs["max_tokens"], 393216)

    def test_user_cap_is_never_dropped(self):
        model = self._model(max_tokens=393216)
        with self.assertRaises(openai.BadRequestError):
            _, create = self._invoke(model, [_bad_request("too big")])
        self.assertEqual(model.request_kwargs["max_tokens"], 393216)

    def test_responses_wire_drops_its_own_cap_field(self):
        with mock.patch(CEILING, return_value=128000):
            model = OpenAIResponses(id="gpt-5.5", api_key="k")
        ok = object()
        create = mock.AsyncMock(side_effect=[_bad_request("max_output_tokens too large"), ok])
        client = mock.Mock()
        client.responses.create = create
        with mock.patch.object(model, "get_client", return_value=client):
            result = asyncio.run(model.invoke([Message(role="user", content="hi")]))
        self.assertIs(result, ok)
        self.assertEqual(create.await_args_list[0].kwargs["max_output_tokens"], 128000)
        self.assertNotIn("max_output_tokens", create.await_args_list[1].kwargs)
        self.assertNotIn("max_output_tokens", model.request_kwargs)


class TestRaiseOutputCap(unittest.TestCase):
    def test_doubles_up_to_the_ceiling(self):
        with mock.patch(CEILING, return_value=None):
            model = OpenAIChat(id="claude-opus-5-5", api_key="k")
        with mock.patch(CEILING, return_value=128000):
            self.assertTrue(model.raise_output_cap(20000))
            self.assertEqual(model.max_tokens, 40000)
            self.assertTrue(model.raise_output_cap(100000))
            self.assertEqual(model.max_tokens, 128000)
            self.assertFalse(model.raise_output_cap(128000))

    def test_user_cap_is_not_raised(self):
        with mock.patch(CEILING, return_value=128000):
            model = OpenAIChat(id="claude-opus-5-5", api_key="k", max_tokens=20000)
            self.assertFalse(model.raise_output_cap(20000))
        self.assertEqual(model.max_tokens, 20000)


class TestReasoningOnlyTruncationRecovery(unittest.TestCase):
    def _truncated(self):
        msg = Message(role="assistant", content=None, reasoning_content="r" * 50)
        msg.metrics["output_tokens"] = 20000
        return [Message(role="user", content="q"), msg]

    def test_raises_cap_and_resends_without_a_prompt(self):
        with mock.patch(CEILING, return_value=None):
            model = OpenAIChat(id="claude-opus-5-5", api_key="k")
        model.last_finish_reason = "length"
        messages = self._truncated()
        with mock.patch(CEILING, return_value=128000):
            self.assertTrue(Runner._loop_post_response(messages, model, LoopState(), had_tool_calls=False))
        self.assertEqual(model.max_tokens, 40000)
        self.assertEqual(messages[-1].role, "assistant")  # nothing injected

    def test_cap_at_ceiling_falls_back_to_next_step_prompt(self):
        with mock.patch(CEILING, return_value=20000):
            model = OpenAIChat(id="claude-opus-5-5", api_key="k")
            model.last_finish_reason = "length"
            messages = self._truncated()
            self.assertTrue(Runner._loop_post_response(messages, model, LoopState(), had_tool_calls=False))
        self.assertEqual(messages[-1].role, "user")
        self.assertIn("next concrete step", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
