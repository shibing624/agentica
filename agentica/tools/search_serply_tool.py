# -*- coding: utf-8 -*-
"""
@author:serply(googio@serply.io)
@description: Search Google via Serply (https://serply.io) for a query.

One API key covers Google web search, Google News and Google Scholar; the
``search_type`` argument picks the vertical. API reference: https://serply.io/docs
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import httpx

from agentica.tools.base import Tool
from agentica.utils.log import logger

SERPLY_BASE_URL = "https://api.serply.io/v1"
# search_type -> (endpoint path, key holding the result list in the response)
SEARCH_TYPES: Dict[str, Tuple[str, str]] = {
    "web": ("search", "results"),
    "news": ("news", "entries"),
    "scholar": ("scholar", "articles"),
}
DEFAULT_SEARCH_TYPE = "web"
MAX_RESULTS = 10  # one Serply page; larger ``num`` values are ignored server-side

_TAG_RE = re.compile(r"<[^>]+>")


def _validate_search_type(search_type: str) -> str:
    if search_type not in SEARCH_TYPES:
        raise ValueError(
            f"Unknown Serply search_type {search_type!r}. Available: {', '.join(SEARCH_TYPES)}."
        )
    return search_type


class SerplyWrapper:
    def __init__(self, api_key: Optional[str], search_type: str = DEFAULT_SEARCH_TYPE, timeout: int = 60):
        if not api_key:
            raise ValueError(
                "To use the Serply search engine, provide `api_key` or set SERPLY_API_KEY. "
                "You can obtain an API key from https://serply.io/."
            )
        self.api_key = api_key
        self.search_type = _validate_search_type(search_type)
        self.timeout = timeout

    def get_headers(self) -> Dict[str, str]:
        # Serply sits behind Cloudflare, which rejects requests without a User-Agent.
        return {"X-Api-Key": self.api_key, "Accept": "application/json", "User-Agent": "agentica"}

    async def run(self, query: str, max_results: int = 8, as_string: bool = True) -> Union[str, List[dict]]:
        """Run query through Serply and parse result"""
        path, list_key = SEARCH_TYPES[self.search_type]
        num = max(1, min(int(max_results), MAX_RESULTS))

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{SERPLY_BASE_URL}/{path}/",
                    params={"q": query, "num": num},
                    headers=self.get_headers(),
                )
            if response.status_code in (401, 403):
                raise ValueError("Unauthorized access to Serply API https://serply.io/. Check your API key.")
            response.raise_for_status()
            data = response.json()

            logger.debug(data)
            if isinstance(data, dict) and data.get("error"):
                raise ValueError(f"Error from Serply API https://serply.io/: {data['error']}")

            res = self._process_response(data, list_key, num, as_string=as_string)
        except Exception as e:
            msg = f"Failed to search `{query}` due to {e}"
            logger.error(msg)
            res = msg
        return res

    @staticmethod
    def _process_response(
            data: dict, list_key: str, max_results: int, as_string: bool = False
    ) -> Union[str, List[dict]]:
        """Keep title / snippet / link of each result, the same shape as the other search engines."""
        items = data.get(list_key) if isinstance(data, dict) else None
        toret_l = []
        for item in (items or [])[:max_results]:
            if not isinstance(item, dict):
                continue
            # web and scholar results carry ``description``; news entries carry an HTML ``summary``
            snippet = item.get("description") or _TAG_RE.sub("", item.get("summary") or "")
            toret_l.append({"title": item.get("title", ""), "snippet": snippet.strip(), "link": item.get("link", "")})
        return json.dumps(toret_l, ensure_ascii=False) if as_string else toret_l


class SearchSerplyTool(Tool):
    """
    Search Google through Serply (https://serply.io). ``search_type`` selects Google web
    search (default), Google News or Google Scholar; all three share one SERPLY_API_KEY.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            search_type: str = DEFAULT_SEARCH_TYPE,
            timeout: int = 60
    ):
        super().__init__(name="search_serply")

        self.timeout: Optional[int] = timeout
        self.api_key: Optional[str] = api_key or os.getenv("SERPLY_API_KEY")
        self.search_type: str = _validate_search_type(search_type)
        self.register(self.search_serply, concurrency_safe=True, is_read_only=True)

    async def search_serply_single_query(
            self,
            query: str,
            max_results: int = 8,
            as_string: bool = True,
    ) -> str:
        """
        Use this function to search Google via Serply for a query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return. Defaults to 8, at most 10.
            as_string: Whether to return the results as a string or a list of dictionaries. Defaults to True.

        Returns:
            The search results as a string or a list of dictionaries.
        """
        wrapper = SerplyWrapper(api_key=self.api_key, search_type=self.search_type, timeout=self.timeout)
        res = await wrapper.run(query, max_results=max_results, as_string=as_string)
        logger.debug(f"Search serply ({self.search_type}) for query: {query}, result: {res}")
        return res

    async def search_serply(self, queries: Union[List[str], str], max_results: int = 8) -> str:
        """
        Search Google for information. Use this tool first to find relevant web pages before visiting them.
        This function searches Google (web, news or scholar, depending on configuration) for one or more
        queries and returns search results with titles, snippets and URLs.

        Args:
            queries: The search query string, or a list of search queries.
            max_results: The maximum number of results to return for each query. Defaults to 8, at most 10.

        Returns:
            Search results containing titles, snippets and URLs that can be used to visit pages for more details.
        """
        if isinstance(queries, str):
            return await self.search_serply_single_query(queries, max_results=max_results, as_string=True)
        all_results = {}
        for query in queries:
            res = await self.search_serply_single_query(query, max_results=max_results, as_string=True)
            all_results[query] = res
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == '__main__':
    import asyncio

    search = SearchSerplyTool()
    r = asyncio.run(search.search_serply(["agentica python agent framework", "retrieval augmented generation"]))
    print(type(r), '\n\n', r)
