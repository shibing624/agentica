# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import json
import ssl
from typing import Any, Optional
import httpx
from bs4 import BeautifulSoup

from agentica.tools.base import Tool
from agentica.utils.log import logger

try:
    from duckduckgo_search import DDGS

    ddgs_enable = True
except ImportError:
    logger.warning("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")
    ddgs_enable = False

# Create a default context for HTTPS requests (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context
DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "search-app/1.0"


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
        self.ddgs = None
        if ddgs_enable:
            self.ddgs = DDGS(headers=self.headers, proxies=self.proxy, timeout=self.timeout)
        self.register(self.duckduckgo_search)

    @staticmethod
    def search_with_ddgs(query: str):
        """
        Search with ddgs and return the contexts.
        """
        contexts = []
        search_results = []
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, backend="lite", timelimit="d, w, m, y")
            for r in ddgs_gen:
                search_results.append(r)
        for idx, result in enumerate(search_results):
            if result["body"] and result["href"]:
                contexts.append({
                    "name": result["title"],
                    "url": result["href"],
                    "snippet": result["body"]
                })
        return contexts

    def duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        """Search DuckDuckGo for a query.

        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Returns:
            The result from DuckDuckGo, in JSON format. The result includes the title, URL, and snippet.
        """
        logger.debug(f"Searching DDG for: {query}")
        try:
            gen_res = self.ddgs.text(query, backend="lite", timelimit="d, w, m, y")
            res = list(gen_res)[:max_results]
            return json.dumps(res, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"DDGS search failed, fallback to DDGS HTML search. reason: {e}")
            res = self.ddgs_html_search(query, max_results)
            return json.dumps(res, indent=2, ensure_ascii=False)

    def ddgs_html_search(self, query: str, max_results: int = 5) -> list:
        """Fallback: Use DuckDuckGo HTML page and parse results."""
        formatted_query = query.replace(" ", "+")
        url = f"{DUCKDUCKGO_URL}?q={formatted_query}"
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client() as client:
                response = client.get(url, headers=headers, timeout=15.0)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                result_elements = soup.select('.result__body')
                results = []
                for result in result_elements[:max_results]:
                    title_elem = result.select_one('.result__a')
                    url_elem = result.select_one('.result__url')
                    snippet_elem = result.select_one('.result__snippet')
                    if title_elem and url_elem:
                        results.append({
                            "title": title_elem.get_text().strip(),
                            "url": url_elem.get_text().strip(),
                            "snippet": snippet_elem.get_text().strip() if snippet_elem else ""
                        })
                return results
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return [{"error": f"Fallback search failed: {str(e)}"}]


if __name__ == '__main__':
    # from agentica.tools.duckduckgo_tool import DuckDuckGoTool
    m = DuckDuckGoTool()
    print(m.duckduckgo_search("Python newest version"))
