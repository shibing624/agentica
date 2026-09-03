# -*- coding: utf-8 -*-
"""
Tests for SearchYoucomTool against the You.com MCP endpoint
(https://api.you.com/mcp, keyless ``?profile=free`` profile without a key).
"""
import asyncio
import json
import os
from unittest.mock import patch

import httpx
import pytest

from agentica.tools.builtin.web_tools import BuiltinWebSearchTool
from agentica.tools.search_youcom_tool import (
    YOUCOM_MCP_FREE_URL,
    YOUCOM_MCP_TOOL,
    YOUCOM_MCP_URL,
    SearchYoucomTool,
)

# Trimmed from a live keyless you-search response: one text content block
# holding the JSON results payload.
WEB_RESPONSE = {
    "result": {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "results": {
                            "web": [
                                {
                                    "url": "https://github.com/shibing624/agentica",
                                    "title": "Agentica: one person, a team of agents",
                                    "description": "Multi-session CLI that collaborates across terminals.",
                                    "snippets": ["Agentica is a Python agent framework."],
                                },
                                {
                                    "url": "https://example.com/2",
                                    "title": "Second result",
                                    "description": "Another page.",
                                    "snippets": ["Two"],
                                },
                            ]
                        }
                    },
                    ensure_ascii=False,
                )
            }
        ]
    },
    "jsonrpc": "2.0",
    "id": 1,
}

_real_client = httpx.AsyncClient


class FakeYoucom:
    """Patches httpx so the tool talks to a fake MCP server, recording requests."""

    def __init__(self, status_code=200, body=None, content_type="text/event-stream"):
        self.status_code = status_code
        self.body = WEB_RESPONSE if body is None else body
        self.content_type = content_type
        self.requests = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.content_type == "text/event-stream":
            # The real endpoint answers as an SSE stream of JSON-RPC messages.
            body = f"event: message\ndata: {json.dumps(self.body)}\n\n".encode()
        else:
            body = None
        return httpx.Response(
            self.status_code,
            content=body,
            json=None if body is not None else self.body,
            headers={"Content-Type": self.content_type},
        )

    def __enter__(self):
        def factory(**kwargs):
            return _real_client(transport=httpx.MockTransport(self._handler), **kwargs)

        self._patch = patch("httpx.AsyncClient", factory)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


class TestYoucomRequest:
    def test_keyless_call_goes_to_the_free_profile_url(self):
        with FakeYoucom() as server:
            asyncio.run(SearchYoucomTool().search_youcom("agentica", max_results=3))
        request = server.last
        assert request.method == "POST"
        assert str(request.url).startswith(YOUCOM_MCP_FREE_URL)
        assert "Authorization" not in request.headers

    def test_keyed_call_goes_to_the_authenticated_url_with_bearer(self):
        with FakeYoucom() as server:
            asyncio.run(SearchYoucomTool(api_key="k").search_youcom("agentica"))
        request = server.last
        assert str(request.url).startswith(YOUCOM_MCP_URL)
        assert "?" not in str(request.url)
        assert request.headers["Authorization"] == "Bearer k"

    def test_api_key_falls_back_to_env(self):
        with patch.dict(os.environ, {"YDC_API_KEY": "env-key"}):
            tool = SearchYoucomTool()
        assert tool.api_key == "env-key"

    def test_call_is_a_stateless_tools_call_with_query_and_count(self):
        with FakeYoucom() as server:
            asyncio.run(SearchYoucomTool().search_youcom("agentica", max_results=3))
        payload = json.loads(server.last.content)
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == YOUCOM_MCP_TOOL
        assert payload["params"]["arguments"] == {"query": "agentica", "count": 3}

    def test_no_initialize_round_trip_is_needed(self):
        """The You.com endpoint is stateless for tools/call; no session handshake."""
        with FakeYoucom() as server:
            asyncio.run(SearchYoucomTool().search_youcom("agentica"))
        assert [json.loads(r.content)["method"] for r in server.requests] == ["tools/call"]


class TestYoucomResponse:
    def test_text_content_block_is_returned_as_is(self):
        with FakeYoucom():
            out = asyncio.run(SearchYoucomTool().search_youcom("agentica"))
        expected = json.loads(WEB_RESPONSE["result"]["content"][0]["text"])
        assert json.loads(out) == expected

    def test_json_content_type_is_parsed_too(self):
        with FakeYoucom(content_type="application/json"):
            out = asyncio.run(SearchYoucomTool().search_youcom("agentica"))
        assert json.loads(out)["results"]["web"][0]["url"] == "https://github.com/shibing624/agentica"

    def test_multiple_queries_are_keyed_by_query(self):
        with FakeYoucom() as server:
            out = asyncio.run(SearchYoucomTool().search_youcom(["a", "b"], max_results=1))
        assert [json.loads(r.content)["params"]["arguments"]["query"] for r in server.requests] == ["a", "b"]
        by_query = json.loads(out)
        assert set(by_query) == {"a", "b"}

    def test_mcp_error_is_reported_as_runtime_error(self):
        with FakeYoucom(body={"jsonrpc": "2.0", "error": {"message": "boom"}}):
            with pytest.raises(RuntimeError, match="You.com"):
                asyncio.run(SearchYoucomTool().search_youcom("q"))


class TestYoucomAsWebSearchBackend:
    def test_provider_youcom_dispatches_to_search_youcom(self):
        tool = BuiltinWebSearchTool(provider="youcom")
        assert tool.provider == "youcom"
        assert tool._search_fn.__name__ == "search_youcom"
        assert tool._search_fn.__self__.api_key is None

    def test_provider_youcom_reads_key_from_env(self):
        with patch.dict(os.environ, {"YDC_API_KEY": "k"}):
            tool = BuiltinWebSearchTool(provider="youcom")
        assert tool._search_fn.__self__.api_key == "k"

    def test_youcom_is_keyless_so_it_never_requires_a_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YDC_API_KEY", None)
            tool = BuiltinWebSearchTool(provider="youcom")  # must not raise
        assert tool.provider == "youcom"

    def test_web_search_returns_youcom_results(self):
        with FakeYoucom():
            tool = BuiltinWebSearchTool(provider="youcom")
            out = asyncio.run(tool.web_search("agentica", max_results=2))
        assert json.loads(out)["results"]["web"][0]["url"] == "https://github.com/shibing624/agentica"
