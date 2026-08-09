# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Search Exa (a web search engine) for a query.

Talks to the Exa REST API with httpx rather than the `exa_py` SDK: one less
dependency, natively async, and no import-time hard failure.
"""

import json
from os import getenv
from typing import Optional, Dict, Any, List, Union

import httpx

from agentica.tools.base import Tool
from agentica.utils.log import logger

EXA_SEARCH_URL = "https://api.exa.ai/search"


class SearchExaTool(Tool):
    def __init__(
            self,
            text: bool = True,
            text_length_limit: int = 1000,
            highlights: bool = True,
            api_key: Optional[str] = None,
            num_results: Optional[int] = None,
            start_crawl_date: Optional[str] = None,
            end_crawl_date: Optional[str] = None,
            start_published_date: Optional[str] = None,
            end_published_date: Optional[str] = None,
            type: Optional[str] = None,
            category: Optional[str] = None,
            include_domains: Optional[List[str]] = None,
            timeout: int = 30,
    ):
        super().__init__(name="search_exa")

        self.api_key = api_key or getenv("EXA_API_KEY")
        if not self.api_key:
            logger.error("EXA_API_KEY not set. Please set the EXA_API_KEY environment variable.")
        self.text: bool = text
        self.text_length_limit: int = text_length_limit
        self.highlights: bool = highlights
        self.num_results: Optional[int] = num_results
        self.start_crawl_date: Optional[str] = start_crawl_date
        self.end_crawl_date: Optional[str] = end_crawl_date
        self.start_published_date: Optional[str] = start_published_date
        self.end_published_date: Optional[str] = end_published_date
        self.type: Optional[str] = type
        self.include_domains: Optional[List[str]] = include_domains
        self.category: Optional[str] = category
        self.timeout: int = timeout

        self.register(self.search_exa, concurrency_safe=True, is_read_only=True)

    def _build_payload(self, query: str, max_results: int) -> Dict[str, Any]:
        contents: Dict[str, Any] = {}
        if self.text:
            # Cap the text server-side so the trimmed characters are never transferred.
            contents["text"] = {"maxCharacters": self.text_length_limit} if self.text_length_limit else True
        if self.highlights:
            contents["highlights"] = True

        payload: Dict[str, Any] = {
            "query": query,
            "numResults": max_results or self.num_results,
            "startCrawlDate": self.start_crawl_date,
            "endCrawlDate": self.end_crawl_date,
            "startPublishedDate": self.start_published_date,
            "endPublishedDate": self.end_published_date,
            "type": self.type,
            "category": self.category,
            "includeDomains": self.include_domains,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if contents:
            payload["contents"] = contents
        return payload

    async def search_exa_single_query(self, query: str, max_results: int = 5) -> str:
        """Use this function to search Exa (a web search engine) for a query.

        Args:
            query (str): The query to search for.
            max_results (int): Number of results to return. Defaults to 5.

        Returns:
            str: The search results in JSON format.
        """
        if not self.api_key:
            return "Please set the EXA_API_KEY"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(EXA_SEARCH_URL, json=self._build_payload(query, max_results), headers=headers)
            response.raise_for_status()
            data = response.json()

        results = []
        for result in data.get("results", []):
            result_dict = {"url": result.get("url", "")}
            for key, field in (("title", "title"), ("author", "author"), ("published_date", "publishedDate")):
                value = result.get(field)
                if value:
                    result_dict[key] = value
            if result.get("text"):
                result_dict["text"] = result["text"]
            if result.get("highlights"):
                result_dict["highlights"] = result["highlights"]
            results.append(result_dict)
        parsed_results = json.dumps(results, ensure_ascii=False)
        logger.debug(f"Searching exa for: {query}, results count: {len(results)}")
        return parsed_results

    async def search_exa(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search Exa for single or multiple queries.

        Args:
            queries (Union[str, List[str]]): A single query string or a list of query strings.
            max_results (int): Number of results to return for each query. Defaults to 5.
        Returns:
            str: The search results in JSON format.
        """
        if isinstance(queries, str):
            return await self.search_exa_single_query(queries, max_results=max_results)
        all_results = {}
        for query in queries:
            result = await self.search_exa_single_query(query, max_results=max_results)
            all_results[query] = result
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == '__main__':
    import asyncio

    m = SearchExaTool()
    query = "苹果的最新产品是啥？"
    r = asyncio.run(m.search_exa(query))
    print(query, '\n\n', r)
    r = asyncio.run(m.search_exa(["北京的新闻top3", "上海的新闻top3"], max_results=3))
    print(r)
