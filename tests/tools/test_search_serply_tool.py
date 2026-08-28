# -*- coding: utf-8 -*-
"""
@author:serply(googio@serply.io)
@description: Tests for SearchSerplyTool against the Serply /v1/{search,news,scholar}/ endpoints.
"""
import asyncio
import json
import os
from unittest.mock import patch

import httpx
import pytest


from agentica.tools.builtin.web_tools import BuiltinWebSearchTool
from agentica.tools.search_serply_tool import (
    MAX_RESULTS,
    SEARCH_TYPES,
    SERPLY_BASE_URL,
    SearchSerplyTool,
)

# Trimmed from live Serply responses; each vertical names its result list differently.
WEB_RESPONSE = {
    "results": [
        {
            "title": "Agentica: build AI agents in Python",
            "link": "https://github.com/shibing624/agentica",
            "description": "Agentica is a Python framework for building LLM agents.",
            "realPosition": 1,
        },
        {
            "title": "Second result",
            "link": "https://example.com/2",
            "description": "Another page.",
        },
    ],
    "ads": [],
    "total": 2,
}
NEWS_RESPONSE = {
    "entries": [
        {
            "title": "Framework release",
            "link": "https://news.example.com/release",
            "summary": "<p>Version <b>1.4</b> shipped.</p>",
            "source": {"title": "Example News", "href": "https://news.example.com"},
            "published": "Mon, 25 Aug 2026 09:00:00 GMT",
        },
        {"title": "Entry two", "link": "https://news.example.com/2", "summary": "Two"},
        {"title": "Entry three", "link": "https://news.example.com/3", "summary": "Three"},
    ]
}
SCHOLAR_RESPONSE = {
    "articles": [
        {
            "title": "Attention is all you need",
            "link": "https://arxiv.org/abs/1706.03762",
            "description": "The dominant sequence transduction models...",
            "extras": {"citations": {"count": 100000}},
        }
    ]
}

_real_client = httpx.AsyncClient


class FakeSerply:
    """Patches httpx so the tool talks to a fake server, recording requests."""

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = WEB_RESPONSE if body is None else body
        self.requests = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.body)

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


class TestSerplyRequest:
    def test_gets_search_endpoint_with_api_key_header(self):
        with FakeSerply() as server:
            asyncio.run(SearchSerplyTool(api_key="k").search_serply("agentica", max_results=3))
        request = server.last
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert str(request.url).startswith(f"{SERPLY_BASE_URL}/search/?")
        assert request.url.params["q"] == "agentica"
        assert request.url.params["num"] == "3"
        assert request.headers["x-api-key"] == "k"
        # Serply is fronted by Cloudflare, which rejects requests without a User-Agent.
        assert request.headers["user-agent"]

    def test_max_results_is_capped_at_one_page(self):
        with FakeSerply() as server:
            asyncio.run(SearchSerplyTool(api_key="k").search_serply("q", max_results=999))
        assert server.last.url.params["num"] == str(MAX_RESULTS)

    def test_max_results_is_clamped_to_one(self):
        with FakeSerply() as server:
            asyncio.run(SearchSerplyTool(api_key="k").search_serply("q", max_results=0))
        assert server.last.url.params["num"] == "1"

    @pytest.mark.parametrize("search_type,path", [(k, v[0]) for k, v in SEARCH_TYPES.items()])
    def test_search_type_selects_the_vertical_endpoint(self, search_type, path):
        with FakeSerply(body={}) as server:
            asyncio.run(SearchSerplyTool(api_key="k", search_type=search_type).search_serply("q"))
        assert server.last.url.path == f"/v1/{path}/"

    def test_api_key_falls_back_to_env(self):
        with patch.dict(os.environ, {"SERPLY_API_KEY": "env-key"}):
            tool = SearchSerplyTool()
        assert tool.api_key == "env-key"

    def test_missing_api_key_raises_on_search(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SERPLY_API_KEY", None)
            tool = SearchSerplyTool()
            with pytest.raises(ValueError, match="SERPLY_API_KEY"):
                asyncio.run(tool.search_serply("q"))

    def test_unknown_search_type_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="search_type"):
            SearchSerplyTool(api_key="k", search_type="images")


class TestSerplyResponse:
    def test_web_results_keep_title_snippet_link(self):
        with FakeSerply():
            out = asyncio.run(SearchSerplyTool(api_key="k").search_serply("agentica"))
        results = json.loads(out)
        assert results[0] == {
            "title": "Agentica: build AI agents in Python",
            "snippet": "Agentica is a Python framework for building LLM agents.",
            "link": "https://github.com/shibing624/agentica",
        }
        assert len(results) == 2

    def test_news_entries_use_summary_with_html_stripped(self):
        with FakeSerply(body=NEWS_RESPONSE):
            out = asyncio.run(SearchSerplyTool(api_key="k", search_type="news").search_serply("release"))
        results = json.loads(out)
        assert results[0]["snippet"] == "Version 1.4 shipped."
        assert results[0]["link"] == "https://news.example.com/release"

    def test_results_are_sliced_client_side_too(self):
        # The news endpoint ignores ``num`` server-side, so the slice must happen here.
        with FakeSerply(body=NEWS_RESPONSE):
            out = asyncio.run(SearchSerplyTool(api_key="k", search_type="news").search_serply("q", max_results=2))
        assert len(json.loads(out)) == 2

    def test_scholar_articles_are_parsed(self):
        with FakeSerply(body=SCHOLAR_RESPONSE):
            out = asyncio.run(SearchSerplyTool(api_key="k", search_type="scholar").search_serply("attention"))
        results = json.loads(out)
        assert results[0]["title"] == "Attention is all you need"
        assert results[0]["link"] == "https://arxiv.org/abs/1706.03762"

    def test_empty_response_gives_empty_list(self):
        with FakeSerply(body={"results": []}):
            out = asyncio.run(SearchSerplyTool(api_key="k").search_serply("nothing"))
        assert json.loads(out) == []

    def test_multiple_queries_are_keyed_by_query(self):
        with FakeSerply() as server:
            out = asyncio.run(SearchSerplyTool(api_key="k").search_serply(["a", "b"], max_results=1))
        assert [r.url.params["q"] for r in server.requests] == ["a", "b"]
        by_query = json.loads(out)
        assert set(by_query) == {"a", "b"}
        assert json.loads(by_query["a"])[0]["link"] == "https://github.com/shibing624/agentica"

    def test_unauthorized_is_reported_not_raised(self):
        with FakeSerply(status_code=401, body={"detail": "invalid key"}):
            out = asyncio.run(SearchSerplyTool(api_key="bad").search_serply("q"))
        assert out.startswith("Failed to search `q`")
        assert "Check your API key" in out

    def test_server_error_is_reported_not_raised(self):
        with FakeSerply(status_code=503, body={"detail": "down"}):
            out = asyncio.run(SearchSerplyTool(api_key="k").search_serply("q"))
        assert out.startswith("Failed to search `q`")


class TestSerplyAsWebSearchBackend:
    def test_provider_serply_dispatches_to_search_serply(self):
        with patch.dict(os.environ, {"SERPLY_API_KEY": "k"}):
            tool = BuiltinWebSearchTool(provider="serply")
        assert tool.provider == "serply"
        assert tool._search_fn.__name__ == "search_serply"
        assert tool._search_fn.__self__.api_key == "k"
        assert tool._search_fn.__self__.search_type == "web"

    def test_missing_key_raises_for_explicit_provider(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SERPLY_API_KEY", None)
            with pytest.raises(ValueError, match="SERPLY_API_KEY"):
                BuiltinWebSearchTool(provider="serply")

    def test_search_type_comes_from_env_on_the_dispatcher_path(self):
        with patch.dict(os.environ, {"SERPLY_API_KEY": "k", "AGENTICA_SERPLY_SEARCH_TYPE": "scholar"}):
            tool = BuiltinWebSearchTool(provider="serply")
        assert tool._search_fn.__self__.search_type == "scholar"

    def test_web_search_returns_serply_results(self):
        with patch.dict(os.environ, {"SERPLY_API_KEY": "k"}), FakeSerply():
            tool = BuiltinWebSearchTool(provider="serply")
            out = asyncio.run(tool.web_search("agentica", max_results=2))
        assert json.loads(out)[0]["link"] == "https://github.com/shibing624/agentica"
