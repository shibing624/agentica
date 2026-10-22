# -*- coding: utf-8 -*-
"""Output-cap wire-key negotiation for OpenAI Chat Completions.

The API's structured unsupported-parameter response drives the negotiation.
No model family is enumerated or inferred from its name.
"""
import asyncio
import json
import os
import unittest
from unittest import mock

import httpx
import openai

from agentica.model.openai import OpenAIChat
from agentica.model.message import Message


def _error_response() -> httpx.Response:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    return httpx.Response(400, request=request)


def _unsupported_param_error(parameter: str) -> openai.BadRequestError:
    return openai.BadRequestError(
        f"Unsupported parameter: {parameter}",
        response=_error_response(),
        body={
            "message": f"Unsupported parameter: {parameter}",
            "type": "invalid_request_error",
            "param": parameter,
            "code": "unsupported_parameter",
        },
    )


class TestOutputCapKeySwap(unittest.TestCase):
    def setUp(self):
        OpenAIChat._OUTPUT_CAP_KEY_BY_DEPLOYMENT.clear()

    def _recording_create(self, calls, responses):
        async def create(request_kwargs):
            calls.append(dict(request_kwargs))
            outcome = responses[len(calls) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return create

    def test_swap_preserves_cap_value_and_remembers_endpoint(self):
        calls = []
        create = self._recording_create(calls, [_unsupported_param_error("max_tokens"), "ok"])
        model = OpenAIChat(
            id="future-model-2030",
            api_key="fake",
            base_url="http://test/v1",
            max_tokens=16384,
        )

        result = asyncio.run(model._create_with_auto_cap(
            create, {"model": "future-model-2030", "max_tokens": 16384},
        ))

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["max_tokens"], 16384)
        self.assertNotIn("max_tokens", calls[1])
        self.assertEqual(calls[1]["max_completion_tokens"], 16384)
        self.assertEqual(
            OpenAIChat._OUTPUT_CAP_KEY_BY_DEPLOYMENT[
                ("OpenAI", "http://test/v1", "future-model-2030")
            ],
            "max_completion_tokens",
        )
        # The configured value stays canonical; only the wire key changes.
        self.assertEqual(model.max_tokens, 16384)
        self.assertIsNone(model.max_completion_tokens)

    def test_new_instance_for_same_endpoint_skips_the_400_round_trip(self):
        first_calls, second_calls = [], []
        first = OpenAIChat(
            id="future-model-2030", api_key="fake", base_url="http://test/v1",
        )
        second = OpenAIChat(
            id="future-model-2030", api_key="fake", base_url="http://test/v1",
        )

        asyncio.run(first._create_with_auto_cap(
            self._recording_create(first_calls, [_unsupported_param_error("max_tokens"), "ok"]),
            {"max_tokens": 8192},
        ))
        result = asyncio.run(second._create_with_auto_cap(
            self._recording_create(second_calls, ["ok-again"]), {"max_tokens": 8192},
        ))

        self.assertEqual(result, "ok-again")
        self.assertEqual(len(second_calls), 1)
        self.assertNotIn("max_tokens", second_calls[0])
        self.assertEqual(second_calls[0]["max_completion_tokens"], 8192)

    def test_environment_base_url_is_part_of_shared_deployment_identity(self):
        first_calls, second_calls = [], []
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "http://env-endpoint/v1"}):
            first = OpenAIChat(id="future-model-env", api_key="fake")
            second = OpenAIChat(id="future-model-env", api_key="fake")
            asyncio.run(first._create_with_auto_cap(
                self._recording_create(
                    first_calls, [_unsupported_param_error("max_tokens"), "ok"],
                ),
                {"max_tokens": 8192},
            ))
            asyncio.run(second._create_with_auto_cap(
                self._recording_create(second_calls, ["ok"]),
                {"max_tokens": 8192},
            ))

        self.assertEqual(len(second_calls), 1)
        self.assertEqual(second_calls[0]["max_completion_tokens"], 8192)

    def test_different_model_on_same_endpoint_negotiates_independently(self):
        first_calls, second_calls = [], []
        first = OpenAIChat(id="future-model-a", api_key="fake", base_url="http://test/v1")
        second = OpenAIChat(id="future-model-b", api_key="fake", base_url="http://test/v1")

        asyncio.run(first._create_with_auto_cap(
            self._recording_create(first_calls, [_unsupported_param_error("max_tokens"), "ok"]),
            {"max_tokens": 8192},
        ))
        asyncio.run(second._create_with_auto_cap(
            self._recording_create(second_calls, ["ok"]),
            {"max_tokens": 8192},
        ))

        self.assertIn("max_tokens", second_calls[0])
        self.assertNotIn("max_completion_tokens", second_calls[0])

    def test_negotiation_works_in_both_directions(self):
        calls = []
        model = OpenAIChat(
            id="legacy-model", api_key="fake", max_completion_tokens=4096,
        )

        result = asyncio.run(model._create_with_auto_cap(
            self._recording_create(
                calls, [_unsupported_param_error("max_completion_tokens"), "ok"],
            ),
            {"max_completion_tokens": 4096},
        ))

        self.assertEqual(result, "ok")
        self.assertEqual(calls[1]["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", calls[1])

    def test_error_without_replacement_hint_still_drops_the_cap(self):
        calls = []
        unrelated_400 = openai.BadRequestError(
            "Error code: 400 - {'error': {'message': 'max_tokens is too large: 999999'}}",
            response=_error_response(), body=None,
        )
        model = OpenAIChat(id="deepseek-v4", api_key="fake")
        model.max_tokens = 999999
        model._auto_output_cap = True

        result = asyncio.run(model._create_with_auto_cap(
            self._recording_create(calls, [unrelated_400, "ok"]),
            {"max_tokens": 999999},
        ))

        self.assertEqual(result, "ok")
        self.assertNotIn("max_tokens", calls[1])
        self.assertNotIn("max_completion_tokens", calls[1])
        self.assertIsNone(model.max_tokens)

    def test_hint_without_max_tokens_in_request_raises(self):
        calls = []
        model = OpenAIChat(id="gpt-5.5", api_key="fake")

        with self.assertRaises(openai.BadRequestError):
            asyncio.run(model._create_with_auto_cap(
                self._recording_create(calls, [_unsupported_param_error("max_tokens")]),
                {"temperature": 0.7},
            ))
        self.assertEqual(len(calls), 1)

    def test_real_request_path_swaps_key_end_to_end(self):
        """response() must serialize the swapped key onto the wire payload."""
        seen = []

        async def transport(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(400, json={"error": {
                    "message": "Unsupported parameter.",
                    "type": "invalid_request_error", "param": "max_tokens",
                    "code": "unsupported_parameter"}})
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-5.5",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        model = OpenAIChat(id="future-model-2030", api_key="fake", base_url="http://test/v1",
                           max_tokens=16384, cache_keepalive=False, max_retries=0)
        model.client = openai.AsyncOpenAI(
            api_key="fake", base_url="http://test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        )

        async def scenario():
            try:
                await model.response(messages=[Message(role="user", content="hi")])
            finally:
                await model.client.close()

        asyncio.run(scenario())
        first, second = (json.loads(r.content) for r in seen)
        self.assertEqual(first["max_tokens"], 16384)
        self.assertNotIn("max_tokens", second)
        self.assertEqual(second["max_completion_tokens"], 16384)
        self.assertEqual(len(seen), 2)
        self.assertEqual(
            OpenAIChat._OUTPUT_CAP_KEY_BY_DEPLOYMENT[
                ("OpenAI", "http://test/v1", "future-model-2030")
            ],
            "max_completion_tokens",
        )


if __name__ == "__main__":
    unittest.main()
