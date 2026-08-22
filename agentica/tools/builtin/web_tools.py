# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Canonical built-in web tools.

``BuiltinWebSearchTool`` is a thin dispatcher: the tool the model sees is always
named ``web_search`` with the same ``(queries, max_results)`` signature, while
the engine behind it is swappable. That keeps prompts, ``RunConfig`` tool
whitelists and permission rules stable across engines.

Selecting an engine, highest priority first:

1. ``provider=`` argument
2. ``AGENTICA_WEB_SEARCH`` environment variable
3. ``DEFAULT_WEB_SEARCH_PROVIDER``

The engine is never inferred from which API keys happen to be set — a key set
for some other purpose must not silently reroute the agent's searches.
"""

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from agentica.tools.base import Tool
from agentica.tools.url_crawler_tool import UrlCrawlerTool
from agentica.utils.log import logger

DEFAULT_WEB_SEARCH_PROVIDER = "exa"
WEB_SEARCH_PROVIDER_ENV = "AGENTICA_WEB_SEARCH"

# A backend search callable: async (queries, max_results) -> str
SearchFn = Callable[[Union[str, List[str]], int], Awaitable[str]]


@dataclass(frozen=True)
class WebSearchBackend:
    """One selectable ``web_search`` engine.

    Attributes:
        factory: ``(api_key) -> tool instance``. Imports its module lazily so
            that engines with optional dependencies never break the others.
        method: Name of the instance's async ``(queries, max_results)`` method.
        key_env: Environment variable holding the API key, if the engine uses one.
        key_required: When True, constructing without a key is an error rather
            than a degraded call.
    """

    factory: Callable[[Optional[str]], Any]
    method: str
    key_env: Optional[str] = None
    key_required: bool = False


def _baidu_backend(api_key: Optional[str]) -> Any:
    from agentica.tools.baidu_search_tool import BaiduSearchTool
    return BaiduSearchTool()


def _duckduckgo_backend(api_key: Optional[str]) -> Any:
    from agentica.tools.duckduckgo_tool import DuckDuckGoTool
    return DuckDuckGoTool()


def _bocha_backend(api_key: Optional[str]) -> Any:
    from agentica.tools.search_bocha_tool import SearchBochaTool
    return SearchBochaTool(api_key=api_key)


def _serper_backend(api_key: Optional[str]) -> Any:
    from agentica.tools.search_serper_tool import SearchSerperTool
    return SearchSerperTool(api_key=api_key)


def _zhipu_backend(api_key: Optional[str]) -> Any:
    """Zhipu's engines differ in quality *and* price, so the tier is selectable."""
    from agentica.tools.zhipu_web_search_tool import DEFAULT_SEARCH_ENGINE, ZhipuWebSearchTool

    engine = os.getenv("AGENTICA_ZHIPU_SEARCH_ENGINE", DEFAULT_SEARCH_ENGINE)
    return ZhipuWebSearchTool(api_key=api_key, search_engine=engine)


def _exa_backend(api_key: Optional[str]) -> Any:
    from agentica.tools.search_mcp_tool import McpSearchTool
    return McpSearchTool(api_key=api_key)


def _custom_mcp_backend(api_key: Optional[str]) -> Any:
    """Any MCP server exposing a search tool, described entirely by env vars."""
    from agentica.tools.search_mcp_tool import McpSearchTool

    url = os.getenv("AGENTICA_WEB_SEARCH_MCP_URL")
    tool_name = os.getenv("AGENTICA_WEB_SEARCH_MCP_TOOL")
    if not url or not tool_name:
        raise ValueError(
            "web_search provider 'mcp' needs AGENTICA_WEB_SEARCH_MCP_URL and "
            "AGENTICA_WEB_SEARCH_MCP_TOOL to be set."
        )
    return McpSearchTool(
        url=url,
        tool_name=tool_name,
        api_key=api_key,
        api_key_query_param=None,  # generic servers take a Bearer header
        count_arg=os.getenv("AGENTICA_WEB_SEARCH_MCP_COUNT_ARG", "numResults"),
    )


_BACKENDS: Dict[str, WebSearchBackend] = {
    "baidu": WebSearchBackend(_baidu_backend, "baidu_search"),
    "duckduckgo": WebSearchBackend(_duckduckgo_backend, "duckduckgo_search"),
    # Exa's public MCP endpoint answers anonymously from a rate-limited shared
    # pool; EXA_API_KEY moves the call onto your own quota.
    "exa": WebSearchBackend(_exa_backend, "mcp_search", "EXA_API_KEY", key_required=False),
    "bocha": WebSearchBackend(_bocha_backend, "search_bocha", "BOCHA_API_KEY", key_required=True),
    "serper": WebSearchBackend(_serper_backend, "search_google", "SERPER_API_KEY", key_required=True),
    "zhipu": WebSearchBackend(_zhipu_backend, "zhipu_web_search", "ZAI_API_KEY", key_required=True),
    "mcp": WebSearchBackend(_custom_mcp_backend, "mcp_search", "AGENTICA_WEB_SEARCH_API_KEY"),
}


def register_web_search_backend(
        name: str,
        factory: Callable[[Optional[str]], Any],
        method: str,
        key_env: Optional[str] = None,
        key_required: bool = False,
) -> None:
    """Register a custom ``web_search`` engine under ``name``.

    Makes the engine selectable by name, including through the
    ``AGENTICA_WEB_SEARCH`` environment variable, so a custom engine registered
    at import time also works for CLI sessions.

    Args:
        name: Provider name used by ``provider=`` / ``AGENTICA_WEB_SEARCH``.
        factory: ``(api_key) -> tool instance``.
        method: Name of the instance's async ``(queries, max_results)`` method.
        key_env: Environment variable holding the engine's API key, if any.
        key_required: Whether a missing key should be an error.

    Example:
        >>> register_web_search_backend(
        ...     "bing", lambda key: MyBingTool(api_key=key), "bing_search",
        ...     key_env="BING_API_KEY", key_required=True,
        ... )
    """
    _BACKENDS[name] = WebSearchBackend(factory, method, key_env, key_required)


def list_web_search_providers() -> List[str]:
    """Return the names of all selectable ``web_search`` engines."""
    return sorted(_BACKENDS)


def resolve_web_search_provider(provider: Optional[str] = None) -> str:
    """Resolve the engine name from the argument, the environment, or the default."""
    return provider or os.getenv(WEB_SEARCH_PROVIDER_ENV) or DEFAULT_WEB_SEARCH_PROVIDER


class BuiltinWebSearchTool(Tool):
    """
    Built-in web search tool with a swappable engine.
    Exposed as web_search function.
    """

    def __init__(
            self,
            provider: Optional[str] = None,
            api_key: Optional[str] = None,
            search_fn: Optional[SearchFn] = None,
    ):
        """
        Initialize BuiltinWebSearchTool.

        Args:
            provider: Engine name, e.g. "exa", "baidu", "bocha", "serper",
                "duckduckgo", "zhipu", "mcp", or any name passed to
                ``register_web_search_backend``. Defaults to the
                ``AGENTICA_WEB_SEARCH`` env var, then to
                ``DEFAULT_WEB_SEARCH_PROVIDER`` (``exa``).
            api_key: Engine API key. Defaults to the engine's own key env var.
            search_fn: Escape hatch — an async ``(queries, max_results) -> str``
                callable used verbatim, for engines that are easier to express
                as a function (e.g. wrapping an existing MCP client) than to
                register. Takes precedence over ``provider``.

        Raises:
            ValueError: Only when ``provider`` was passed explicitly and is
                unknown or missing its API key. An engine named in code is an
                intent, and silently searching with a different one would be
                lying to the caller.

        A provider that came from the environment instead (``AGENTICA_WEB_SEARCH``,
        which ``config.yaml``'s env block also feeds) degrades to the default
        keyless engine with a warning. That value is deployment configuration,
        possibly set for an entirely different process on the same machine;
        letting it abort construction takes down the whole service because one
        optional tool is missing one optional key.
        """
        super().__init__(name="builtin_web_search_tool")

        if search_fn is not None:
            self.provider = "custom"
            self._search_fn: SearchFn = search_fn
        else:
            explicit = provider is not None
            self.provider = resolve_web_search_provider(provider)
            backend, reason = self._resolve_backend(self.provider, api_key)
            if backend is None:
                if explicit:
                    raise ValueError(reason)
                logger.warning(f"{reason} Falling back to {DEFAULT_WEB_SEARCH_PROVIDER!r}.")
                self.provider = DEFAULT_WEB_SEARCH_PROVIDER
                backend, _ = self._resolve_backend(self.provider, None)
            key = api_key or (os.getenv(backend.key_env) if backend.key_env else None)
            self._search_fn = getattr(backend.factory(key), backend.method)

        logger.debug(f"BuiltinWebSearchTool using provider: {self.provider}")
        self.register(self.web_search, concurrency_safe=True, is_read_only=True)

    @staticmethod
    def _resolve_backend(
            provider: str, api_key: Optional[str]
    ) -> Tuple[Optional[WebSearchBackend], str]:
        """The backend for ``provider``, or None plus why it is unusable."""
        backend = _BACKENDS.get(provider)
        if backend is None:
            return None, (
                f"Unknown web_search provider {provider!r}. "
                f"Available: {', '.join(list_web_search_providers())}."
            )
        key = api_key or (os.getenv(backend.key_env) if backend.key_env else None)
        if backend.key_required and not key:
            return None, (
                f"web_search provider {provider!r} requires an API key: "
                f"set {backend.key_env} or pass api_key=..."
            )
        return backend, ""

    async def web_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search the web for multiple queries and return results

        Args:
            queries (Union[str, List[str]]): Search keyword(s), can be a single string or a list of strings
            max_results (int, optional): Number of results to return for each query, default 5

        Returns:
            str: A JSON formatted string containing the search results.

        IMPORTANT: After using this tool:
        1. Read through the 'content' field of each result
        2. Extract relevant information that answers the user's question
        3. Synthesize this into a clear, natural language response
        4. Cite sources by mentioning the page titles or URLs
        5. NEVER show the raw JSON to the user - always provide a formatted response
        """
        result = await self._search_fn(queries, max_results=max_results)
        logger.debug(f"Web search for '{queries}', result length: {len(result)} characters.")
        return result


class BuiltinFetchUrlTool(Tool):
    """
    Built-in URL fetching tool that wraps UrlCrawlerTool.
    Exposed as fetch_url function for consistent naming in Agent.
    """

    def __init__(self, max_content_length: int = 16000):
        """
        Initialize BuiltinFetchUrlTool.

        Args:
            max_content_length: Maximum length of returned content
        """
        super().__init__(name="builtin_fetch_url_tool")
        self.max_content_length = max_content_length
        self._crawler = UrlCrawlerTool(max_content_length=max_content_length)
        self.register(self.fetch_url, concurrency_safe=True, is_read_only=True)

    async def fetch_url(self, url: str) -> str:
        """Fetch URL content and convert to clean text format.

        Args:
            url: URL to fetch, url starts with http:// or https://

        Returns:
            str, JSON formatted fetch result containing url and content.

        IMPORTANT: After using this tool:
        1. The ``content`` field already holds the extracted, ready-to-use
           text. Work directly from it — do NOT open or read any cache/file
           path; the content here is what you need.
        2. If the page was truncated and you need a different section, call
           fetch_url again or use web_search for a more specific source —
           never try to read a raw cached file.
        3. Extract the relevant information that answers the user's question
           and synthesize a clear, natural-language response.
        4. NEVER show the raw JSON to the user unless specifically requested.
        """
        result = await self._crawler.url_crawl(url)
        logger.debug(f"Fetched URL: {url}, result length: {len(result)} characters.")
        return result
