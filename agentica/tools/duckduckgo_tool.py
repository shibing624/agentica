# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description: DuckDuckGo web search over the public HTML endpoint.

Uses httpx + BeautifulSoup, both core dependencies, instead of the
`duckduckgo-search` package: nothing extra to install, and no degrading to an
"not installed" error string at runtime.
"""
import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from agentica.tools.base import Tool
from agentica.utils.log import logger

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Sponsored results are routed through this tracker rather than linking out.
_AD_MARKER = "duckduckgo.com/y.js"


def _resolve_url(href: Optional[str]) -> str:
    """Turn a result link into the destination URL.

    DuckDuckGo sometimes links out directly and sometimes wraps the target in
    a ``/l/?uddg=<encoded>`` redirect; the visible ``.result__url`` text is
    truncated for display and unusable as an actual link.
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


class DuckDuckGoTool(Tool):
    def __init__(
            self,
            headers: Optional[Any] = None,
            proxy: Optional[Any] = None,
            timeout: Optional[int] = 10,
    ):
        super().__init__(name="duckduckgo_tool")

        self.headers: Optional[Any] = headers
        self.proxy: Optional[Any] = proxy
        self.timeout: Optional[int] = timeout
        self.register(self.duckduckgo_search)

    async def duckduckgo_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search DuckDuckGo for single or multiple queries.

        Args:
            queries (Union[str, List[str]]): A single query string or a list of query strings.
            max_results (optional, default=5): The maximum number of results to return for each query.

        Returns:
            The result from DuckDuckGo, in JSON format. The result includes the title, URL, and snippet.
        """
        if not isinstance(queries, str):
            all_results = {}
            for query in queries:
                all_results[query] = await self.duckduckgo_search_single_query(query, max_results)
            return json.dumps(all_results, ensure_ascii=False)
        return await self.duckduckgo_search_single_query(queries, max_results)

    async def duckduckgo_search_single_query(self, query: str, max_results: int = 5) -> str:
        """Search DuckDuckGo for a single query.

        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Returns:
            The result from DuckDuckGo, in JSON format. The result includes the title, URL, and snippet.
        """
        headers = {"User-Agent": USER_AGENT}
        if self.headers:
            headers.update(self.headers)

        async with httpx.AsyncClient(timeout=self.timeout, proxy=self.proxy, follow_redirects=True) as client:
            response = await client.post(DUCKDUCKGO_URL, data={"q": query}, headers=headers)
            response.raise_for_status()

        results = self._parse_results(response.text, max_results)
        logger.debug(f"Searching DDG for: {query}, results count: {len(results)}")
        return json.dumps(results, indent=2, ensure_ascii=False)

    @staticmethod
    def _parse_results(html: str, max_results: int) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[Dict[str, str]] = []
        for element in soup.select(".result__body"):
            link = element.select_one(".result__a")
            if link is None:
                continue
            href = link.get("href")
            if href and _AD_MARKER in href:
                continue
            url = _resolve_url(href)
            if not url:
                continue
            snippet = element.select_one(".result__snippet")
            results.append({
                "title": link.get_text().strip(),
                "url": url,
                "snippet": snippet.get_text().strip() if snippet else "",
            })
            if len(results) >= max_results:
                break
        return results


if __name__ == '__main__':
    import asyncio

    m = DuckDuckGoTool()
    print(asyncio.run(m.duckduckgo_search("Python newest version")))
    print(asyncio.run(m.duckduckgo_search(["rust tokio", "go goroutines"], max_results=2)))
