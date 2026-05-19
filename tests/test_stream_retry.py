# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for stream_with_retry wrapper.

Verifies:
- Transient pre-yield failures retry up to max_retries times.
- Once a chunk has been yielded, errors propagate verbatim (no duplicate output).
- Non-retryable errors propagate immediately.
- Exhausted retries re-raise the last exception.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.model.stream_retry import stream_with_retry, default_is_parser_error


def _run(coro):
    return asyncio.run(coro)


async def _collect(it):
    out = []
    async for x in it:
        out.append(x)
    return out


class TestStreamRetryHelper:

    def test_success_no_retry(self):
        calls = {"n": 0}

        async def _open():
            calls["n"] += 1

            async def _gen():
                for x in ["a", "b", "c"]:
                    yield x

            return _gen()

        async def _do():
            return await _collect(stream_with_retry(_open))

        result = _run(_do())
        assert result == ["a", "b", "c"]
        assert calls["n"] == 1

    def test_retry_on_pre_yield_parser_error(self):
        calls = {"n": 0}

        async def _open():
            calls["n"] += 1
            if calls["n"] < 3:
                # Simulate a malformed first-chunk JSON decode failure
                raise json.JSONDecodeError("expecting value", "doc", 0)

            async def _gen():
                for x in ["x", "y"]:
                    yield x

            return _gen()

        async def _do():
            return await _collect(
                stream_with_retry(_open, max_retries=3, base_delay=0.01)
            )

        result = _run(_do())
        assert result == ["x", "y"]
        assert calls["n"] == 3  # 2 failures + 1 success

    def test_retry_on_extra_substring_open(self):
        """User-supplied substring (e.g. private proxy marker) must trigger retry.

        ``venus_error`` is intentionally NOT in SDK defaults — it's a private
        corp gateway marker. Pass it through ``extra_substrings`` to retry.
        """
        calls = {"n": 0}

        async def _open():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': '状态错误', "
                    "'type': 'venus_error', 'code': '4001'}}"
                )

            async def _gen():
                yield "ok"

            return _gen()

        async def _do():
            return await _collect(
                stream_with_retry(
                    _open,
                    extra_substrings=["venus_error"],
                    max_retries=2,
                    base_delay=0.01,
                )
            )

        assert _run(_do()) == ["ok"]
        assert calls["n"] == 2

    def test_no_retry_on_unknown_proxy_marker_without_opt_in(self):
        """Without ``extra_substrings``, vendor-specific marker must NOT retry."""
        calls = {"n": 0}

        async def _open():
            calls["n"] += 1
            raise RuntimeError(
                "Error code: 400 - {'type': 'venus_error', 'code': '4001'}"
            )

        async def _do():
            try:
                await _collect(stream_with_retry(_open, max_retries=2, base_delay=0.01))
                return "no-raise"
            except RuntimeError:
                return "raised"

        assert _run(_do()) == "raised"
        assert calls["n"] == 1  # no retry without explicit opt-in

    def test_no_retry_after_first_chunk_yielded(self):
        calls = {"n": 0}

        async def _open():
            calls["n"] += 1

            async def _gen():
                yield "first"
                raise json.JSONDecodeError("mid-stream", "doc", 5)

            return _gen()

        async def _do():
            chunks = []
            try:
                async for c in stream_with_retry(_open, max_retries=3, base_delay=0.01):
                    chunks.append(c)
            except json.JSONDecodeError:
                return chunks, "raised"
            return chunks, "no-raise"

        chunks, status = _run(_do())
        assert chunks == ["first"]
        assert status == "raised"
        # Critical: must NOT retry once a chunk has been delivered to caller
        assert calls["n"] == 1

    def test_non_retryable_error_propagates_immediately(self):
        calls = {"n": 0}

        class _AuthError(Exception):
            pass

        async def _open():
            calls["n"] += 1
            # 401 / "invalid api key" should NOT be retried
            raise _AuthError("401 invalid api key")

        async def _do():
            try:
                await _collect(
                    stream_with_retry(_open, max_retries=3, base_delay=0.01)
                )
                return "no-raise"
            except _AuthError:
                return "raised"

        assert _run(_do()) == "raised"
        assert calls["n"] == 1

    def test_exhausted_retries_reraise_last(self):
        calls = {"n": 0}

        async def _open():
            calls["n"] += 1
            raise RuntimeError("bad gateway 502")

        async def _do():
            try:
                await _collect(
                    stream_with_retry(_open, max_retries=2, base_delay=0.01)
                )
                return "no-raise"
            except RuntimeError as e:
                assert "502" in str(e)
                return "raised"

        assert _run(_do()) == "raised"
        # max_retries=2 -> total 3 attempts
        assert calls["n"] == 3


class TestDefaultIsParserError:

    def test_classifies_502_503_as_retryable(self):
        assert default_is_parser_error(RuntimeError("502 bad gateway"))
        assert default_is_parser_error(RuntimeError("503 service unavailable"))

    def test_classifies_json_decode_as_retryable(self):
        assert default_is_parser_error(json.JSONDecodeError("x", "doc", 0))

    def test_does_not_classify_auth_error(self):
        assert not default_is_parser_error(RuntimeError("401 invalid api key"))
        assert not default_is_parser_error(ValueError("the model 'gpt-9' does not exist"))

    def test_does_not_classify_vendor_proxy_by_default(self):
        # venus_error is private to a specific corp gateway — SDK has no
        # business hardcoding it. Without opt-in it must NOT match.
        assert not default_is_parser_error(RuntimeError("venus_error 4001"))

    def test_extra_substrings_enable_vendor_match(self):
        assert default_is_parser_error(
            RuntimeError("venus_error 4001 状态错误"),
            extra_substrings=["venus_error"],
        )

    def test_env_var_enables_vendor_match(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS", "venus_error,aiproxy_busy")
        assert default_is_parser_error(RuntimeError("aiproxy_busy: try later"))
        assert default_is_parser_error(RuntimeError("VENUS_ERROR 4001"))
