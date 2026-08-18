# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the tolerant SSE byte-stream sanitizer.

Reproduces the OpenAI-compatible proxy bug where SSE frames arrive
double-wrapped (``data: data: {...}``). The OpenAI SDK strips exactly one
``data:`` prefix and hands the remaining ``data: {...}`` to the JSON parser,
which dies with ``json.JSONDecodeError: Expecting value: line 1 column 1
(char 0)`` and kills the stream mid-way.
"""
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.model.message import Message
from agentica.model.openai.chat import OpenAIChat
from agentica.model.openai.sse_sanitize import (
    TolerantSSETransport,
    collapse_sse_data_prefixes,
)


def _run(coro):
    return asyncio.run(coro)


class _ListAsyncByteStream(httpx.AsyncByteStream):
    """Byte stream yielding the given parts, for fake transport responses."""

    def __init__(self, parts):
        self._parts = list(parts)
        self.closed = False

    async def __aiter__(self):
        for part in self._parts:
            yield part

    async def aclose(self):
        self.closed = True


async def _collect(it):
    out = []
    async for x in it:
        out.append(x)
    return out


# A frame shaped like a real proxy error: reasoning_content delta plus the
# proxy's private proxyMarker field, delivered with a doubled data: prefix.
DOUBLE_WRAPPED_FRAME = (
    b'data: data: {"id":"c1","object":"chat.completion.chunk","created":1786592626,'
    b'"model":"ep-test","choices":[{"index":0,"delta":{"reasoning_content":"0"},'
    b'"finish_reason":null}],"proxyMarker":{"spanId":"abc"}}\n\n'
)
NORMAL_FRAME = (
    b'data: {"id":"c1","object":"chat.completion.chunk","created":1786592626,'
    b'"model":"ep-test","choices":[{"index":0,"delta":{"content":"Hello"},'
    b'"finish_reason":null}]}\n\n'
)
DONE_FRAME = b"data: [DONE]\n\n"


class TestCollapseSSEDataPrefixes:
    def test_double_wrapped_line_collapsed(self):
        line = b'data: data: {"a": 1}\n'
        assert collapse_sse_data_prefixes(line) == b'data: {"a": 1}\n'

    def test_triple_wrapped_line_collapsed(self):
        line = b'data: data: data: {"a": 1}\n'
        assert collapse_sse_data_prefixes(line) == b'data: {"a": 1}\n'

    def test_single_prefix_line_untouched(self):
        line = b'data: {"a": 1}\n'
        assert collapse_sse_data_prefixes(line) == line

    def test_done_sentinel_untouched(self):
        assert collapse_sse_data_prefixes(b"data: [DONE]\n") == b"data: [DONE]\n"

    def test_double_wrapped_done_collapsed(self):
        assert collapse_sse_data_prefixes(b"data: data: [DONE]\n") == b"data: [DONE]\n"

    def test_comment_line_untouched(self):
        line = b": keepalive ping\n"
        assert collapse_sse_data_prefixes(line) == line

    def test_empty_line_untouched(self):
        assert collapse_sse_data_prefixes(b"\n") == b"\n"

    def test_crlf_preserved(self):
        line = b'data: data: {"a": 1}\r\n'
        assert collapse_sse_data_prefixes(line) == b'data: {"a": 1}\r\n'

    def test_json_payload_containing_data_substring_untouched(self):
        # "data: " inside the JSON string value must not trigger collapsing.
        line = b'data: {"content": "data: x"}\n'
        assert collapse_sse_data_prefixes(line) == line


class TestTolerantSSEByteStream:
    def _wrap(self, parts):
        from agentica.model.openai.sse_sanitize import _TolerantSSEByteStream

        return _TolerantSSEByteStream(_ListAsyncByteStream(parts))

    def test_line_split_across_chunks(self):
        stream = self._wrap([b'data: da', b'ta: {"a": 1}\n\n', DONE_FRAME])
        out = _run(_collect(stream))
        assert b"".join(out) == b'data: {"a": 1}\n\n' + DONE_FRAME

    def test_final_partial_line_flushed(self):
        stream = self._wrap([b'data: data: {"a": 1}'])
        out = _run(_collect(stream))
        assert b"".join(out) == b'data: {"a": 1}'

    def test_well_formed_stream_passes_byte_identical(self):
        body = NORMAL_FRAME + DONE_FRAME
        stream = self._wrap([body])
        out = _run(_collect(stream))
        assert b"".join(out) == body

    def test_aclose_forwards_to_inner(self):
        inner = _ListAsyncByteStream([])
        from agentica.model.openai.sse_sanitize import _TolerantSSEByteStream

        stream = _TolerantSSEByteStream(inner)
        _run(stream.aclose())
        assert inner.closed


class TestTolerantSSETransport:
    def _read_body(self, content_type, parts):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": content_type},
                stream=_ListAsyncByteStream(parts),
            )

        async def _do():
            transport = TolerantSSETransport(httpx.MockTransport(handler))
            async with httpx.AsyncClient(transport=transport) as client:
                request = client.build_request("POST", "http://test/v1/chat/completions")
                response = await client.send(request, stream=True)
                try:
                    return await response.aread()
                finally:
                    await response.aclose()

        return _run(_do())

    def test_event_stream_response_is_sanitized(self):
        out = self._read_body(
            "text/event-stream; charset=utf-8",
            [DOUBLE_WRAPPED_FRAME, NORMAL_FRAME, DONE_FRAME],
        )
        assert out == DOUBLE_WRAPPED_FRAME.replace(b"data: data: ", b"data: ") + NORMAL_FRAME + DONE_FRAME

    def test_non_event_stream_response_passes_through_untouched(self):
        body = b'data: data: this is not SSE, do not touch\n'
        out = self._read_body("application/octet-stream", [body])
        assert out == body


class TestOpenAIChatStreamEndToEnd:
    def _model_with_body(self, parts):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_ListAsyncByteStream(parts),
            )

        http_client = httpx.AsyncClient(transport=TolerantSSETransport(httpx.MockTransport(handler)))
        return OpenAIChat(id="ep-test", api_key="test-key", http_client=http_client)

    def test_double_wrapped_stream_parses_without_error(self):
        model = self._model_with_body([DOUBLE_WRAPPED_FRAME, NORMAL_FRAME, DONE_FRAME])
        chunks = _run(_collect(model.invoke_stream(messages=[Message(role="user", content="hi")])))
        assert len(chunks) == 2
        # Extra vendor fields must survive parsing (extra="allow" on SDK models).
        assert getattr(chunks[0].choices[0].delta, "reasoning_content", None) == "0"
        assert chunks[0].model_extra.get("proxyMarker") == {"spanId": "abc"}
        assert chunks[1].choices[0].delta.content == "Hello"

    def test_double_wrapped_frame_split_across_byte_chunks(self):
        body = DOUBLE_WRAPPED_FRAME + NORMAL_FRAME + DONE_FRAME
        # Split mid-line, inside the doubled prefix.
        parts = [body[:11], body[11:37], body[37:]]
        model = self._model_with_body(parts)
        chunks = _run(_collect(model.invoke_stream(messages=[Message(role="user", content="hi")])))
        assert len(chunks) == 2
        assert chunks[1].choices[0].delta.content == "Hello"


class TestGetClientWiring:
    def test_default_http_client_uses_tolerant_sse_transport(self):
        model = OpenAIChat(id="ep-test", api_key="test-key")
        client = model.get_client()
        assert isinstance(client._client._transport, TolerantSSETransport)
