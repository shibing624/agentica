# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Search Bocha (a web search engine) for a query.
"""

import json
from os import getenv
from typing import Optional, List, Union

import requests

from agentica.tools.base import Tool
from agentica.utils.log import logger


class SearchBochaTool(Tool):
    def __init__(
            self,
            api_key: Optional[str] = None,
            summary: bool = True,
            count: int = 10,
            freshness: Optional[str] = None,
            include: Optional[str] = None,
            exclude: Optional[str] = None,
    ):
        """Initialize SearchBochaTool.

        Args:
            api_key: Bocha API key. If not provided, will use BOCHA_API_KEY env variable.
            summary: Whether to show text summary in results. Defaults to True.
            count: Number of results to return (1-50). Defaults to 10.
            freshness: Time range for search. Options: noLimit, oneDay, oneWeek, oneMonth, oneYear,
                       or date range like "2025-01-01..2025-04-06". Defaults to None (noLimit).
            include: Domains to include in search, separated by | or ,. e.g. "qq.com|m.163.com"
            exclude: Domains to exclude from search, separated by | or ,. e.g. "qq.com|m.163.com"
        """
        super().__init__(name="search_bocha")

        self.api_key = api_key or getenv("BOCHA_API_KEY")
        if not self.api_key:
            logger.error("BOCHA_API_KEY not set. Please set the BOCHA_API_KEY environment variable.")

        self.summary = summary
        self.count = count
        self.freshness = freshness
        self.include = include
        self.exclude = exclude
        self.api_url = "https://api.bocha.cn/v1/web-search"

        self.register(self.search_bocha)

    def search_bocha_single_query(self, query: str, count: Optional[int] = None) -> str:
        """Search Bocha for a single query.

        Args:
            query: The search query string.
            count: Number of results to return. If not provided, uses instance default.

        Returns:
            str: The search results in JSON format.
        """
        if not self.api_key:
            return "Please set the BOCHA_API_KEY"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "query": query,
                "summary": self.summary,
                "count": count or self.count,
            }

            if self.freshness:
                payload["freshness"] = self.freshness
            if self.include:
                payload["include"] = self.include
            if self.exclude:
                payload["exclude"] = self.exclude

            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 200:
                error_msg = result.get("msg", "Unknown error")
                logger.error(f"Bocha API error: {error_msg}")
                return f"Error: {error_msg}"

            data = result.get("data", {})
            web_pages = data.get("webPages", {})
            values = web_pages.get("value", [])

            parsed_results = []
            for item in values:
                result_dict = {
                    "url": item.get("url", ""),
                    "title": item.get("name", ""),
                }
                if item.get("snippet"):
                    result_dict["snippet"] = item.get("summary") or item.get("snippet")
                if item.get("siteName"):
                    result_dict["site_name"] = item.get("siteName")
                if item.get("datePublished"):
                    result_dict["published_date"] = item.get("datePublished")

                parsed_results.append(result_dict)

            parsed_json = json.dumps(parsed_results, ensure_ascii=False)
            logger.debug(f"Searching bocha for: {query}, results count: {len(parsed_results)}")
            return parsed_json

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search bocha: {e}")
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Failed to search bocha: {e}")
            return f"Error: {e}"

    def search_bocha(self, queries: Union[str, List[str]], count: Optional[int] = None) -> str:
        """Search Bocha for single or multiple queries.

        Args:
            queries: A single query string or a list of query strings.
            count: Number of results to return for each query. If not provided, default 10.

        Returns:
            str: The search results in JSON format.
        """
        if isinstance(queries, str):
            return self.search_bocha_single_query(queries, count=count)

        all_results = {}
        for query in queries:
            result = self.search_bocha_single_query(query, count=count)
            all_results[query] = result
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == "__main__":
    m = SearchBochaTool()
    query = "天空为什么是蓝色的？"
    r = m.search_bocha(query)
    print(query, "\n\n", r)
    r = m.search_bocha(["北京的新闻top3", "上海的新闻top3"], count=3)
    print(r)
