# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Canonical built-in web tools.
"""

from typing import List, Union

from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.base import Tool
from agentica.tools.url_crawler_tool import UrlCrawlerTool
from agentica.utils.log import logger


class BuiltinWebSearchTool(Tool):
    """
    Built-in web search tool using Baidu search.
    Exposed as web_search function.
    """

    def __init__(self):
        """
        Initialize BuiltinWebSearchTool.

        Note: BaiduSearchTool's bs4 dependency is in agentica core (since v1.3.6),
        so this always works after `pip install agentica`.
        """
        super().__init__(name="builtin_web_search_tool")
        self._search = BaiduSearchTool()
        self.register(self.web_search, concurrency_safe=True, is_read_only=True)

    async def web_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search the web using Baidu for multiple queries and return results

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

        result = await self._search.baidu_search(queries, max_results=max_results)
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
