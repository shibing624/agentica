# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import json
import ssl
from typing import Any, Optional

from agentica.tools.base import Tool
from agentica.utils.log import logger

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")

# Create a default context for HTTPS requests (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context


class DuckDuckGoTool(Tool):
    def __init__(
            self,
            search: bool = True,
            news: bool = True,
            headers: Optional[Any] = None,
            proxy: Optional[Any] = None,
            timeout: Optional[int] = 10,
    ):
        super().__init__(name="duckduckgo_tool")

        self.headers: Optional[Any] = headers
        self.proxy: Optional[Any] = proxy
        self.timeout: Optional[int] = timeout
        self.ddgs = DDGS(headers=self.headers, proxies=self.proxy, timeout=self.timeout)
        if search:
            self.register(self.duckduckgo_search)
        if news:
            self.register(self.duckduckgo_news)

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

        Example:
            from agentica.tools.duckduckgo_tool import DuckDuckGoTool
            m = DuckDuckGoTool()
            search_results = m.duckduckgo_search("Python")
            print(search_results)

        Returns:
            The result from DuckDuckGo, in JSON format. The result includes the title, URL, and snippet.
        """
        logger.debug(f"Searching DDG for: {query}")
        gen_res = self.ddgs.text(query, backend="lite", timelimit="d, w, m, y")
        res = list(gen_res)[:max_results]
        return json.dumps(res, indent=2, ensure_ascii=False)

    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        """Get the latest news from DuckDuckGo.

        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Example:
            from agentica.tools.duckduckgo_tool import DuckDuckGoTool
            m = DuckDuckGoTool()
            news_results = m.duckduckgo_news("Python")
            print(news_results)

        Returns:
            The latest news from DuckDuckGo.
        """
        logger.debug(f"Searching DDG news for: {query}")
        gen_res = self.ddgs.news(query, timelimit="d, w, m, y")
        res = list(gen_res)[:max_results]
        return json.dumps(res, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # from agentica.tools.duckduckgo_tool import DuckDuckGoTool
    m = DuckDuckGoTool()
    print(m.duckduckgo_search("Python"))
    print(m.duckduckgo_news("Python"))
