# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tolerant SSE sanitizer for OpenAI-compatible streaming endpoints.

Some OpenAI-compatible proxies occasionally double-wrap SSE frames, emitting
lines like ``data: data: {...}``. The OpenAI SDK strips exactly one ``data:``
prefix and hands the remaining ``data: {...}`` to the JSON parser, which dies
with ``json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`` and
kills the stream mid-way — after output has already started, so retrying
would duplicate tokens and tool calls.

The framing error is unambiguous: a legitimate ``chat.completion.chunk``
payload is always a JSON object (``{...}``) or the ``[DONE]`` sentinel, never
something starting with ``data: ``. So we collapse repeated ``data: ``
prefixes at the transport layer, before the SDK's SSE decoder runs.

Only responses whose Content-Type is ``text/event-stream`` are rewritten;
binary downloads and plain JSON responses pass through byte-identical.
"""
import logging
from typing import AsyncIterator, cast

import httpx

logger = logging.getLogger(__name__)

_DATA_PREFIX = b"data:"


def collapse_sse_data_prefixes(line: bytes) -> bytes:
    """Collapse repeated ``data: `` prefixes on a single SSE line.

    ``line`` is a complete line including its trailing ``\\n`` (or the final
    partial line of a stream). Returns the line unchanged unless its data
    payload itself starts with another ``data:`` prefix, which only happens
    when a proxy double-wrapped the frame.
    """
    if line.endswith(b"\n"):
        body, tail = line[:-1], b"\n"
    else:
        body, tail = line, b""
    if body.endswith(b"\r"):
        body, tail = body[:-1], b"\r" + tail
    if not body.startswith(_DATA_PREFIX):
        return line
    payload = body[len(_DATA_PREFIX):]
    if payload.startswith(b" "):
        payload = payload[1:]
    collapsed = payload
    while collapsed.startswith(_DATA_PREFIX):
        collapsed = collapsed[len(_DATA_PREFIX):]
        if collapsed.startswith(b" "):
            collapsed = collapsed[1:]
    if collapsed == payload:
        return line
    logger.warning(
        "[sse-sanitize] collapsed doubled data: prefix on SSE frame (%.80r)",
        line[:80],
    )
    return b"data: " + collapsed + tail


class _TolerantSSEByteStream(httpx.AsyncByteStream):
    """Wraps a response byte stream, fixing double-wrapped ``data:`` lines.

    Buffers partial lines across chunk boundaries; only complete lines are
    rewritten. Everything else passes through byte-identical.
    """

    def __init__(self, inner: httpx.AsyncByteStream) -> None:
        self._inner = inner
        self._pending = b""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for part in self._inner:
            data = self._pending + part
            lines = data.split(b"\n")
            self._pending = lines.pop()
            for raw in lines:
                yield collapse_sse_data_prefixes(raw + b"\n")
        if self._pending:
            yield collapse_sse_data_prefixes(self._pending)
            self._pending = b""

    async def aclose(self) -> None:
        await self._inner.aclose()


class TolerantSSETransport(httpx.AsyncBaseTransport):
    """httpx transport that sanitizes event-stream responses.

    Wraps ``inner`` (the real transport). Responses with a
    ``text/event-stream`` Content-Type get their byte stream wrapped in
    :class:`_TolerantSSEByteStream`; all other responses pass through
    untouched.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if "text/event-stream" in response.headers.get("content-type", ""):
            # Async transports always hand back an async byte stream.
            response.stream = _TolerantSSEByteStream(cast(httpx.AsyncByteStream, response.stream))
        return response
