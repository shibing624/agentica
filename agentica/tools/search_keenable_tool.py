# -*- coding: utf-8 -*-
"""
@description: Search the web with Keenable (https://keenable.ai).

Works without an API key through Keenable's free public endpoint; with
KEENABLE_API_KEY set it uses the keyed endpoint on your own quota.
"""

import json
from os import getenv
from typing import List, Optional, Union

import httpx

from agentica.tools.base import Tool
from agentica.utils.log import logger

KEENABLE_PUBLIC_URL = "https://api.keenable.ai/v1/search/public"
KEENABLE_KEYED_URL = "https://api.keenable.ai/v1/search"
SNIPPET_MAX_CHARS = 1000


class SearchKeenableTool(Tool):
    def __init__(self, api_key: Optional[str] = None, count: int = 10):
        """Initialize SearchKeenableTool.

        Args:
            api_key: Optional Keenable API key. Falls back to KEENABLE_API_KEY; without
                a key the free public endpoint is used.
            count: Number of results to return. Defaults to 10.
        """
        super().__init__(name="search_keenable")
        self.api_key = api_key or getenv("KEENABLE_API_KEY")
        self.count = count
        self.register(self.search_keenable, concurrency_safe=True, is_read_only=True)

    @staticmethod
    def _snippet(item: dict) -> str:
        """Prefer the short description; otherwise truncate the (possibly long) snippet."""
        description = (item.get("description") or "").strip()
        if description:
            return description
        snippet = item.get("snippet") or ""
        return snippet[:SNIPPET_MAX_CHARS] + "…" if len(snippet) > SNIPPET_MAX_CHARS else snippet

    async def search_keenable_single_query(self, query: str, max_results: Optional[int] = None) -> str:
        """Search Keenable for a single query and return results as JSON."""
        headers = {"Content-Type": "application/json", "X-Keenable-Title": "Agentica"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        url = KEENABLE_KEYED_URL if self.api_key else KEENABLE_PUBLIC_URL
        payload = {"query": query, "max_results": max_results or self.count}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=json.dumps(payload), timeout=30)
            response.raise_for_status()
            data = response.json()

        parsed_results = []
        for item in data.get("results", []) or []:
            if not item.get("url"):
                continue
            result_dict = {"url": item["url"], "title": item.get("title", "")}
            snippet = self._snippet(item)
            if snippet:
                result_dict["snippet"] = snippet
            if item.get("published_at"):
                result_dict["published_date"] = item["published_at"]
            parsed_results.append(result_dict)

        logger.debug(f"Searching keenable for: {query}, results count: {len(parsed_results)}")
        return json.dumps(parsed_results, ensure_ascii=False)

    async def search_keenable(self, queries: Union[str, List[str]], max_results: Optional[int] = None) -> str:
        """Search the web with Keenable for single or multiple queries.

        Args:
            queries: A single query string or a list of query strings.
            max_results: Number of results to return for each query. If not provided, default 10.

        Returns:
            str: The search results in JSON format.
        """
        if isinstance(queries, str):
            return await self.search_keenable_single_query(queries, max_results=max_results)

        all_results = {}
        for query in queries:
            all_results[query] = await self.search_keenable_single_query(query, max_results=max_results)
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(SearchKeenableTool().search_keenable("why is the sky blue", max_results=3)))
