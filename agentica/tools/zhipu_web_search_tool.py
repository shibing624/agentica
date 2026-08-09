# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ZhipuAI Web Search API, a search engine built for LLM consumption.

Talks to the dedicated ``/paas/v4/web_search`` endpoint, which takes real search
parameters (engine, result count, domain and recency filters) and returns a flat
result list.
"""

import json
import uuid
from os import getenv
from typing import Any, Dict, List, Optional, Union

import httpx

from agentica.tools.base import Tool
from agentica.utils.log import logger

ZHIPU_WEB_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"

# Engine code -> capability and list price. Doubles as the validation set, so a
# typo fails at construction with the choices spelled out.
SEARCH_ENGINES: Dict[str, str] = {
    "search_std": "基础版（智谱自研），满足日常查询，性价比最高，0.01 元/次",
    "search_pro": "高级版（智谱自研），多引擎协作，空结果率低、召回与准确率更高，0.03 元/次",
    "search_pro_sogou": "搜狗，覆盖腾讯生态（新闻/企鹅号）与知乎，百科、医疗等垂域权威性强，0.05 元/次",
    "search_pro_quark": "夸克，精准触达垂直内容，0.05 元/次",
}
DEFAULT_SEARCH_ENGINE = "search_pro"
RECENCY_FILTERS = ("oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit")
CONTENT_SIZES = ("medium", "high")
MIN_COUNT, MAX_COUNT = 1, 50
# Documented as a hard maximum, but the server accepts longer queries, so
# exceeding it is worth a warning rather than a rejection.
MAX_QUERY_CHARS = 70
USER_ID_LENGTH = (6, 128)
# A favicon URL and a footnote index are pure context bloat for a model.
_DROPPED_FIELDS = ("icon", "refer")


class ZhipuWebSearchTool(Tool):
    def __init__(
            self,
            api_key: Optional[str] = None,
            search_engine: str = DEFAULT_SEARCH_ENGINE,
            count: int = 10,
            search_domain_filter: Optional[str] = None,
            search_recency_filter: Optional[str] = None,
            content_size: Optional[str] = None,
            search_intent: bool = False,
            user_id: Optional[str] = None,
            timeout: int = 60,
    ):
        """Initialize ZhipuWebSearchTool.

        Args:
            api_key: ZhipuAI API key. Falls back to ZAI_API_KEY / ZHIPUAI_API_KEY.
            search_engine: One of ``SEARCH_ENGINES``; they differ in both
                quality and price. Note the two 智谱自研 tiers (``search_std``,
                ``search_pro``) return an empty ``link`` for roughly 40% of
                queries — all results or none, per query — so use
                ``search_pro_sogou`` / ``search_pro_quark`` when the caller
                needs a citable source URL for every result.
            count: Result count (clamped to 1-50) used when the caller does not
                pass ``max_results``. Note ``search_pro_sogou`` only accepts
                10/20/30/40/50, and ignores ``count`` when both
                ``search_domain_filter`` and ``search_recency_filter`` are set;
                results are truncated locally either way.
            search_domain_filter: Restrict results to a single whitelisted
                domain, e.g. "www.example.com".
            search_recency_filter: One of ``RECENCY_FILTERS``. Defaults to the
                API default (noLimit).
            content_size: Summary length, "medium" (default) or "high".
            search_intent: Ask the API to run intent recognition. When enabled
                the returned JSON becomes an object carrying both
                ``search_intent`` and ``search_result``.
            user_id: End-user identifier (6-128 chars) forwarded to Zhipu so it
                can attribute and intervene on abuse. Worth setting when one
                API key serves many end users, e.g. a gateway deployment.
            timeout: HTTP timeout in seconds.
        """
        super().__init__(name="zhipu_web_search")

        self.api_key = api_key or getenv("ZAI_API_KEY") or getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            logger.error("ZAI_API_KEY not set. Please set the ZAI_API_KEY environment variable.")
        if search_engine not in SEARCH_ENGINES:
            raise ValueError(
                f"Unknown search_engine {search_engine!r}. Available: {', '.join(SEARCH_ENGINES)}"
            )
        if search_recency_filter is not None and search_recency_filter not in RECENCY_FILTERS:
            raise ValueError(
                f"Unknown search_recency_filter {search_recency_filter!r}. "
                f"Available: {', '.join(RECENCY_FILTERS)}"
            )
        if content_size is not None and content_size not in CONTENT_SIZES:
            raise ValueError(
                f"Unknown content_size {content_size!r}. Available: {', '.join(CONTENT_SIZES)}"
            )
        if user_id is not None and not USER_ID_LENGTH[0] <= len(user_id) <= USER_ID_LENGTH[1]:
            raise ValueError(
                f"user_id must be {USER_ID_LENGTH[0]}-{USER_ID_LENGTH[1]} characters, "
                f"got {len(user_id)}"
            )

        self.search_engine = search_engine
        self.count = count
        self.search_domain_filter = search_domain_filter
        self.search_recency_filter = search_recency_filter
        self.content_size = content_size
        self.search_intent = search_intent
        self.user_id = user_id
        self.timeout = timeout
        self.register(self.zhipu_web_search, concurrency_safe=True, is_read_only=True)

    def _build_payload(self, query: str, count: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "search_query": query,
            "search_engine": self.search_engine,
            "search_intent": self.search_intent,
            "count": count,
            "request_id": str(uuid.uuid4()),
        }
        optional = {
            "search_domain_filter": self.search_domain_filter,
            "search_recency_filter": self.search_recency_filter,
            "content_size": self.content_size,
            "user_id": self.user_id,
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        return payload

    async def zhipu_web_search_single_query(self, query: str, max_results: int = 5) -> str:
        """Search the web for a single query.

        Args:
            query (str): The query to search for. Keep it short; Zhipu
                recommends under 70 characters.
            max_results (int): Number of results to return. Defaults to 5.

        Returns:
            str: The search results in JSON format, each with title, content,
                link, media and publish_date.
        """
        if not self.api_key:
            return "Please set the ZAI_API_KEY"

        if len(query) > MAX_QUERY_CHARS:
            logger.warning(
                f"Zhipu recommends search_query under {MAX_QUERY_CHARS} characters, "
                f"got {len(query)}; results may be weaker."
            )
        count = max(MIN_COUNT, min(max_results or self.count, MAX_COUNT))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                ZHIPU_WEB_SEARCH_URL, json=self._build_payload(query, count), headers=headers
            )
        if resp.status_code != 200:
            # The body carries the only actionable part (balance, concurrency cap,
            # no engine available); the bare status code reads as plain rate
            # limiting even when the real cause is an empty account.
            raise RuntimeError(f"Zhipu web_search failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()

        # ``count`` is only a hint: search_pro_sogou snaps it up to the nearest
        # of 10/20/30/40/50, and the others overshoot on some queries. Results
        # run ~1K characters each, so honouring max_results here is what keeps a
        # 3-result request from spending 10 results' worth of context.
        results = [
            {k: v for k, v in item.items() if k not in _DROPPED_FIELDS and v}
            for item in data.get("search_result", [])
        ][:count]
        output: Any = results
        if self.search_intent:
            output = {"search_intent": data.get("search_intent", []), "search_result": results}
        logger.debug(f"Searching zhipu {self.search_engine} for: {query}, results count: {len(results)}")
        return json.dumps(output, indent=2, ensure_ascii=False)

    async def zhipu_web_search(self, queries: Union[List[str], str], max_results: int = 5) -> str:
        """Search the web for single or multiple queries.

        Args:
            queries (Union[List[str], str]): A single query string or a list of
                query strings. Keep each query under 70 characters.
            max_results (int): Number of results to return for each query. Defaults to 5.

        Returns:
            str: The search results in JSON format.
        """
        if isinstance(queries, str):
            return await self.zhipu_web_search_single_query(queries, max_results=max_results)
        results = {}
        for query in queries:
            results[query] = await self.zhipu_web_search_single_query(query, max_results=max_results)
        return json.dumps(results, ensure_ascii=False)


if __name__ == '__main__':
    import asyncio

    m = ZhipuWebSearchTool()
    query = "苹果的最新产品是啥？"
    r = asyncio.run(m.zhipu_web_search(query, max_results=3))
    print(query, '\n\n', r)
    r = asyncio.run(m.zhipu_web_search(["湖北的新闻top3", "北京的娱乐新闻"], max_results=2))
    print(r)
