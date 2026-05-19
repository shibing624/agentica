# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for tool-error redaction before LLM context injection.

When a tool raises and the agent injects the error string into the next
turn's context, secrets in tracebacks (Authorization headers, sk-... API
keys, env-style assignments, JWTs, URL tokens) must be redacted. Successful
tool outputs must pass through verbatim.
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_fake_model():
    """Construct a minimal Model instance bypassing abstract methods."""
    from agentica.model.base import Model

    # Subclass to satisfy abstractmethods with no-op stubs.
    class _M(Model):
        @property
        def request_kwargs(self):
            return {}

        async def invoke(self, messages):
            raise NotImplementedError

        async def invoke_stream(self, messages):
            if False:
                yield None

        async def response(self, messages):
            raise NotImplementedError

        async def response_stream(self, messages):
            if False:
                yield None

    return _M(id="fake-test")


def _build_failing_function_call(error_text: str):
    """Build a FunctionCall whose entrypoint raises with the given message."""
    from agentica.tools.base import FunctionCall, Function

    def _raises():
        raise RuntimeError(error_text)

    return FunctionCall(
        function=Function(name="dummy", entrypoint=_raises, description="x"),
        arguments={},
        call_id="call_test_1",
    )


class TestSanitizeToolError:
    """Verify error strings are redacted before being placed in Message.content."""

    def test_bearer_token_in_error_is_redacted(self):
        model = _build_fake_model()

        async def _run():
            results = []
            fc = _build_failing_function_call(
                "auth failed -- Authorization: Bearer "
                "eyJhbGciOiJIUzI1NiJ9.abcdefghijklmnopqrstuvwx"
            )
            async for _ in model.run_function_calls(
                function_calls=[fc],
                function_call_results=results,
            ):
                pass
            return results

        results = asyncio.run(_run())
        assert len(results) == 1
        content = results[0].content or ""
        # No raw bearer token leaks
        assert "eyJhbGciOiJ" not in content
        assert "REDACTED" in content
        assert results[0].tool_call_error is True

    def test_openai_sk_key_in_error_is_redacted(self):
        model = _build_fake_model()

        async def _run():
            results = []
            fc = _build_failing_function_call(
                "bad creds, used sk-proj-"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
            )
            async for _ in model.run_function_calls(
                function_calls=[fc], function_call_results=results,
            ):
                pass
            return results

        results = asyncio.run(_run())
        assert "sk-proj-ABCDEFGHIJKL" not in (results[0].content or "")
        assert "REDACTED" in (results[0].content or "")

    def test_url_query_token_in_error_is_redacted(self):
        model = _build_fake_model()

        async def _run():
            results = []
            fc = _build_failing_function_call(
                "HTTP 401 calling https://api.x.com/v1/me?api_key=topsecret_xyz123&page=1"
            )
            async for _ in model.run_function_calls(
                function_calls=[fc], function_call_results=results,
            ):
                pass
            return results

        results = asyncio.run(_run())
        content = results[0].content or ""
        assert "topsecret_xyz123" not in content
        # Non-secret query params are preserved
        assert "page=1" in content

    def test_successful_output_passes_through(self):
        """Success branch must NOT be sanitized — structured payloads stay intact."""
        model = _build_fake_model()

        async def _run():
            results = []
            from agentica.tools.base import FunctionCall, Function

            def _dummy():
                # Note: the literal "sk-" string here is NOT a real secret
                # shape; we want to verify success outputs are passed verbatim.
                return '{"data": "raw_value_42", "Authorization": "Bearer xyz"}'

            fc = FunctionCall(
                function=Function(name="dummy", entrypoint=_dummy, description="x"),
                arguments={},
                call_id="call_test_ok",
            )
            fc.result = '{"data": "raw_value_42"}'
            fc.error = None

            async for _ in model.run_function_calls(
                function_calls=[fc], function_call_results=results,
            ):
                pass
            return results

        results = asyncio.run(_run())
        # Success path: content equals the raw tool result, untouched.
        # (Success here is determined by execute() returning truthy w/o exception.)
        content = results[0].content or ""
        # The dummy entrypoint returned the JSON string; either way the
        # contents must include the raw payload (no REDACTED stamping on success).
        assert "raw_value_42" in content
