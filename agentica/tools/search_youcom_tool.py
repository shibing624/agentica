# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Search You.com (a web search engine) for a query.

Talks to You.com's remote MCP server (Streamable HTTP) directly over httpx,
the same way ``search_mcp_tool.py`` does, instead of requiring the optional
``mcp`` package. The endpoint is stateless for ``tools/call``: no session id
is issued or needed, so one POST per query is enough.

Without ``YDC_API_KEY`` the tool talks to the keyless ``?profile=free``
endpoint, which answers anonymously from a shared pool with basic
``you-search``. Setting the key moves the call onto your own quota with the
full You.com search surface (see https://you.com/platform/api-keys).
"""

import json
from os import getenv
from typing import Any, Dict, List, Optional, Union

import httpx

from agentica.tools.base import Tool
from agentica.utils.log import logger

YOUCOM_MCP_URL = "https://api.you.com/mcp"
YOUCOM_MCP_FREE_URL = "https://api.you.com/mcp?profile=free"
YOUCOM_MCP_TOOL = "you-search"

_MCP_PROTOCOL_VERSION = "2024-11-05"
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _parse_mcp_response(response: httpx.Response) -> Dict[str, Any]:
    """Decode one MCP reply, which may arrive as JSON or as an SSE stream."""
    if "text/event-stream" not in response.headers.get("Content-Type", ""):
        return response.json()

    # SSE: the payload is the last complete `data:` JSON object.
    last_obj: Dict[str, Any] = {}
    for line in response.text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            last_obj = json.loads(data)
        except json.JSONDecodeError:
            continue
    return last_obj


class SearchYoucomTool(Tool):
    """Web search via You.com's MCP server. Keyless by default."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            url: str = YOUCOM_MCP_URL,
            free_url: str = YOUCOM_MCP_FREE_URL,
            tool_name: str = YOUCOM_MCP_TOOL,
            timeout: int = 30,
    ):
        """Initialize SearchYoucomTool.

        Args:
            api_key: You.com API key. Without one, the keyless ``?profile=free``
                endpoint is used (shared free pool, basic search). Defaults to
                ``YDC_API_KEY``.
            url: Authenticated MCP endpoint.
            free_url: Keyless MCP endpoint used when no key is set.
            tool_name: Name of the search tool to call on that server.
            timeout: Per-request timeout in seconds.
        """
        super().__init__(name="search_youcom")

        self.api_key = api_key or getenv("YDC_API_KEY")
        self.url = url
        self.free_url = free_url
        self.tool_name = tool_name
        self.timeout = timeout

        self.register(self.search_youcom, concurrency_safe=True, is_read_only=True)

    def _endpoint(self) -> str:
        return self.url if self.api_key else self.free_url

    def _headers(self) -> Dict[str, str]:
        headers = dict(_MCP_HEADERS)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def search_youcom_single_query(self, query: str, max_results: int = 5) -> str:
        """Use this function to search You.com (a web search engine) for a query.

        Args:
            query (str): The query to search for.
            max_results (int): Number of results to return. Defaults to 5.

        Returns:
            str: The search results in JSON format.
        """
        arguments: Dict[str, Any] = {"query": query, "count": max_results}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._endpoint(),
                headers=self._headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": self.tool_name, "arguments": arguments},
                },
            )
            response.raise_for_status()

        payload = _parse_mcp_response(response)
        if "error" in payload:
            raise RuntimeError(f"MCP search error from You.com: {payload['error']}")

        blocks = (payload.get("result") or {}).get("content", [])
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        result = "\n\n".join(t for t in texts if t)
        logger.debug(f"Searching youcom for: {query}, result length: {len(result)}")
        return result

    async def search_youcom(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search You.com for single or multiple queries.

        Args:
            queries (Union[str, List[str]]): A single query string or a list of query strings.
            max_results (int): Number of results to return for each query. Defaults to 5.
        Returns:
            str: The search results in JSON format.
        """
        if isinstance(queries, str):
            return await self.search_youcom_single_query(queries, max_results=max_results)
        all_results = {}
        for query in queries:
            all_results[query] = await self.search_youcom_single_query(query, max_results=max_results)
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == '__main__':
    import asyncio

    m = SearchYoucomTool()
    query = "agentica python framework"
    r = asyncio.run(m.search_youcom(query))
    print(query, '\n\n', r)
