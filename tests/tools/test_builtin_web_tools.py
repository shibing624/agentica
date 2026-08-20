# -*- coding: utf-8 -*-
"""Tests for BuiltinWebSearchTool, BuiltinFetchUrlTool, and get_builtin_tools()."""
import asyncio
import inspect
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from agentica.tools.builtin import (
    BuiltinFetchUrlTool,
    BuiltinWebSearchTool,
    get_builtin_tools,
    list_web_search_providers,
    register_web_search_backend,
)
from agentica.tools.builtin.web_tools import (
    BuiltinFetchUrlTool as CanonicalBuiltinFetchUrlTool,
    BuiltinWebSearchTool as CanonicalBuiltinWebSearchTool,
)


def test_web_tool_legacy_exports_point_to_canonical_classes():
    assert BuiltinWebSearchTool is CanonicalBuiltinWebSearchTool
    assert BuiltinFetchUrlTool is CanonicalBuiltinFetchUrlTool


def test_get_builtin_tools_still_returns_expected_tool_types():
    tools = get_builtin_tools(work_dir="/tmp")
    tool_names = {tool.__class__.__name__ for tool in tools}
    assert "BuiltinFileTool" in tool_names
    assert "BuiltinExecuteTool" in tool_names
    assert "BuiltinWebSearchTool" in tool_names
    assert "BuiltinFetchUrlTool" in tool_names
    assert "BuiltinTodoTool" in tool_names
    assert "BuiltinTaskTool" in tool_names

class TestBuiltinWebSearchTool:
    def test_web_search_delegates_to_baidu(self):
        """Verify web_search calls BaiduSearchTool.baidu_search under the hood."""
        tool = BuiltinWebSearchTool()

        mock_result = json.dumps([{"title": "test", "url": "http://example.com", "content": "result"}])
        tool._search_fn = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.web_search("test query"))
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "test"
        tool._search_fn.assert_awaited_once_with("test query", max_results=5)

    def test_web_search_multiple_queries(self):
        tool = BuiltinWebSearchTool()
        mock_result = json.dumps({"q1": [], "q2": []})
        tool._search_fn = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.web_search(["q1", "q2"], max_results=3))
        tool._search_fn.assert_awaited_once_with(["q1", "q2"], max_results=3)

    def test_web_search_error_handling(self):
        tool = BuiltinWebSearchTool()
        tool._search_fn = AsyncMock(side_effect=Exception("network error"))

        # After方案A: search failures propagate as exceptions instead of Error strings.
        # Runner/FunctionCall.invoke captures them into function_call.error.
        with pytest.raises(Exception, match="network error"):
            asyncio.run(tool.web_search("fail"))


class TestWebSearchProviderSelection:
    """The engine behind `web_search` is swappable; the tool name is not."""

    def test_default_provider_is_baidu_and_binds_baidu_method(self):
        tool = BuiltinWebSearchTool()
        assert tool.provider == "baidu"
        assert tool._search_fn.__name__ == "baidu_search"

    def test_tool_name_stays_web_search_across_providers(self):
        """RunConfig whitelists and prompts key off the function name."""
        for provider in ("baidu", "duckduckgo", "exa"):
            tool = BuiltinWebSearchTool(provider=provider)
            assert "web_search" in tool.functions

    def test_explicit_provider_arg_wins_over_env(self):
        with patch.dict(os.environ, {"AGENTICA_WEB_SEARCH": "duckduckgo"}):
            tool = BuiltinWebSearchTool(provider="baidu")
        assert tool.provider == "baidu"

    def test_env_selects_provider(self):
        with patch.dict(os.environ, {"AGENTICA_WEB_SEARCH": "duckduckgo"}):
            tool = BuiltinWebSearchTool()
        assert tool.provider == "duckduckgo"
        assert tool._search_fn.__name__ == "duckduckgo_search"

    def test_api_key_never_selects_the_provider(self):
        """A key set for another purpose must not silently reroute searches."""
        with patch.dict(os.environ, {"BOCHA_API_KEY": "k", "SERPER_API_KEY": "k"}, clear=False):
            os.environ.pop("AGENTICA_WEB_SEARCH", None)
            tool = BuiltinWebSearchTool()
        assert tool.provider == "baidu"

    def test_keyed_provider_reads_key_from_env(self):
        with patch.dict(os.environ, {"BOCHA_API_KEY": "bocha-key"}):
            tool = BuiltinWebSearchTool(provider="bocha")
        assert tool.provider == "bocha"
        assert tool._search_fn.__self__.api_key == "bocha-key"

    def test_explicit_api_key_wins_over_env(self):
        with patch.dict(os.environ, {"BOCHA_API_KEY": "env-key"}):
            tool = BuiltinWebSearchTool(provider="bocha", api_key="arg-key")
        assert tool._search_fn.__self__.api_key == "arg-key"

    def test_missing_required_key_raises_instead_of_silent_fallback(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SERPER_API_KEY", None)
            with pytest.raises(ValueError, match="requires an API key"):
                BuiltinWebSearchTool(provider="serper")

    def test_exa_works_without_key_anonymously(self):
        """Exa's public MCP endpoint answers without credentials."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            tool = BuiltinWebSearchTool(provider="exa")
        assert tool.provider == "exa"
        assert tool._search_fn.__self__.api_key is None

    def test_unknown_provider_raises_and_lists_options(self):
        with pytest.raises(ValueError, match="Unknown web_search provider"):
            BuiltinWebSearchTool(provider="nope")

    def test_env_provider_missing_its_key_degrades_instead_of_raising(self):
        """Deployment config must not be able to abort agent construction.

        AGENTICA_WEB_SEARCH may be seeded from config.yaml for an entirely
        different process on the same machine; raising here takes the whole
        service down because one optional tool is missing one optional key.
        """
        with patch.dict(os.environ, {"AGENTICA_WEB_SEARCH": "serper"}, clear=False):
            os.environ.pop("SERPER_API_KEY", None)
            tool = BuiltinWebSearchTool()
        assert tool.provider == "baidu"
        assert "web_search" in tool.functions

    def test_unknown_env_provider_degrades_too(self):
        with patch.dict(os.environ, {"AGENTICA_WEB_SEARCH": "nope"}, clear=False):
            tool = BuiltinWebSearchTool()
        assert tool.provider == "baidu"

    def test_a_deep_agent_still_builds_under_a_dirty_env(self):
        with patch.dict(os.environ, {"AGENTICA_WEB_SEARCH": "serper"}, clear=False):
            os.environ.pop("SERPER_API_KEY", None)
            tools = get_builtin_tools(
                include_file_tools=False, include_execute=False,
                include_fetch_url=False, include_todos=False, include_task=False,
            )
        assert [t.provider for t in tools] == ["baidu"]

    def test_get_builtin_tools_passes_provider_through(self):
        tools = get_builtin_tools(
            include_file_tools=False, include_execute=False, include_fetch_url=False,
            include_todos=False, include_task=False, web_search_provider="duckduckgo",
        )
        assert [t.provider for t in tools] == ["duckduckgo"]


class TestWebSearchCustomBackend:
    """Custom engines plug in by callable (SDK) or by name (also CLI/env)."""

    def test_search_fn_is_used_verbatim(self):
        calls = []

        async def my_search(queries, max_results=5):
            calls.append((queries, max_results))
            return "custom result"

        tool = BuiltinWebSearchTool(search_fn=my_search)
        assert tool.provider == "custom"
        assert asyncio.run(tool.web_search("q", max_results=2)) == "custom result"
        assert calls == [("q", 2)]

    def test_search_fn_wins_over_provider(self):
        async def my_search(queries, max_results=5):
            return "custom"

        tool = BuiltinWebSearchTool(provider="serper", search_fn=my_search)
        assert tool.provider == "custom"

    def test_registered_backend_is_selectable_by_name_and_env(self):
        class FakeBingTool:
            def __init__(self, api_key):
                self.api_key = api_key

            async def bing_search(self, queries, max_results=5):
                return f"bing:{queries}:{self.api_key}"

        register_web_search_backend(
            "bing-test", lambda key: FakeBingTool(key), "bing_search",
            key_env="BING_TEST_API_KEY", key_required=True,
        )
        assert "bing-test" in list_web_search_providers()

        with patch.dict(os.environ, {"AGENTICA_WEB_SEARCH": "bing-test", "BING_TEST_API_KEY": "bk"}):
            tool = BuiltinWebSearchTool()
        assert tool.provider == "bing-test"
        assert asyncio.run(tool.web_search("q")) == "bing:q:bk"

    def test_registered_backend_honours_key_required(self):
        register_web_search_backend(
            "needs-key-test", lambda key: object(), "whatever",
            key_env="NEEDS_KEY_TEST", key_required=True,
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEEDS_KEY_TEST", None)
            with pytest.raises(ValueError, match="requires an API key"):
                BuiltinWebSearchTool(provider="needs-key-test")


class TestWebSearchBackendSignatureUniformity:
    """Every engine must answer to the same (queries, max_results) call."""

    def test_all_backends_share_the_dispatch_signature(self):
        import inspect

        from agentica.tools.baidu_search_tool import BaiduSearchTool
        from agentica.tools.duckduckgo_tool import DuckDuckGoTool
        from agentica.tools.search_bocha_tool import SearchBochaTool
        from agentica.tools.search_mcp_tool import McpSearchTool
        from agentica.tools.search_serper_tool import SearchSerperTool
        from agentica.tools.zhipu_web_search_tool import ZhipuWebSearchTool

        cases = [
            (BaiduSearchTool, "baidu_search"),
            (DuckDuckGoTool, "duckduckgo_search"),
            (SearchBochaTool, "search_bocha"),
            (SearchSerperTool, "search_google"),
            (ZhipuWebSearchTool, "zhipu_web_search"),
            (McpSearchTool, "mcp_search"),
        ]
        for cls, method_name in cases:
            method = getattr(cls, method_name)
            params = list(inspect.signature(method).parameters)
            assert inspect.iscoroutinefunction(method), f"{method_name} must be async"
            assert params[:3] == ["self", "queries", "max_results"], f"{method_name} signature drifted"


# ===========================================================================
# BuiltinFetchUrlTool tests
# ===========================================================================

class TestBuiltinFetchUrlTool:
    def test_fetch_url_delegates_to_crawler(self):
        """Verify fetch_url calls UrlCrawlerTool.url_crawl under the hood."""
        tool = BuiltinFetchUrlTool()

        mock_result = json.dumps({"url": "http://example.com", "content": "page content", "save_path": "/tmp/x"})
        tool._crawler.url_crawl = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.fetch_url("http://example.com"))
        parsed = json.loads(result)
        assert parsed["url"] == "http://example.com"
        assert parsed["content"] == "page content"
        tool._crawler.url_crawl.assert_awaited_once_with("http://example.com")
