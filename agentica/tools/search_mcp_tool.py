# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search backed by any MCP server that exposes a search tool.

Speaks the MCP Streamable-HTTP transport directly over httpx (a core dependency)
instead of going through ``agentica.mcp``, which needs the optional ``mcp``
package. That keeps this usable as a default ``web_search`` backend.

Exa is the shipped preset because its public endpoint answers anonymously
(shared free pool); setting ``EXA_API_KEY`` moves the call onto your own quota.
Point ``url`` / ``tool_name`` at any other MCP server to plug in a custom
search engine without writing code.
"""

import json
from os import getenv
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import httpx

from agentica.tools.base import Tool
from agentica.utils.log import logger

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
EXA_MCP_TOOL = "web_search_exa"

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


class McpSearchTool(Tool):
    """Web search via an MCP server. Defaults to Exa's public endpoint."""

    def __init__(
            self,
            url: str = EXA_MCP_URL,
            tool_name: str = EXA_MCP_TOOL,
            api_key: Optional[str] = None,
            api_key_query_param: Optional[str] = "exaApiKey",
            query_arg: str = "query",
            count_arg: Optional[str] = "numResults",
            timeout: int = 60,
    ):
        """Initialize McpSearchTool.

        Args:
            url: MCP server endpoint (Streamable HTTP).
            tool_name: Name of the search tool to call on that server.
            api_key: Optional key. Without one, Exa's endpoint still answers
                from a rate-limited shared pool.
            api_key_query_param: Query-string parameter used to pass the key
                (Exa wants ``exaApiKey``). Set to None to send the key as an
                ``Authorization: Bearer`` header instead.
            query_arg: Name of the tool argument carrying the query text.
            count_arg: Name of the tool argument carrying the result count.
                None for servers that take no count.
            timeout: Per-request timeout in seconds.
        """
        super().__init__(name="mcp_search")
        self.url = url
        self.tool_name = tool_name
        self.api_key = api_key or getenv("EXA_API_KEY")
        self.api_key_query_param = api_key_query_param
        self.query_arg = query_arg
        self.count_arg = count_arg
        self.timeout = timeout

        self.register(self.mcp_search, concurrency_safe=True, is_read_only=True)

    def _endpoint(self) -> str:
        if self.api_key and self.api_key_query_param:
            sep = "&" if "?" in self.url else "?"
            return f"{self.url}{sep}{urlencode({self.api_key_query_param: self.api_key})}"
        return self.url

    def _headers(self) -> Dict[str, str]:
        headers = dict(_MCP_HEADERS)
        if self.api_key and not self.api_key_query_param:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def mcp_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search the web for single or multiple queries via an MCP search server.

        Args:
            queries (Union[str, List[str]]): A single query string or a list of query strings.
            max_results (int): Number of results to return for each query. Defaults to 5.

        Returns:
            str: The search results as text (JSON-wrapped when multiple queries).
        """
        if isinstance(queries, str):
            return await self.mcp_search_single_query(queries, max_results=max_results)
        all_results = {}
        for query in queries:
            all_results[query] = await self.mcp_search_single_query(query, max_results=max_results)
        return json.dumps(all_results, ensure_ascii=False)

    async def mcp_search_single_query(self, query: str, max_results: int = 5) -> str:
        """Run one MCP search round-trip: initialize, notify, then call the tool.

        A fresh session per call keeps the tool stateless and safe to run
        concurrently, at the cost of two extra short POSTs.
        """
        arguments: Dict[str, Any] = {self.query_arg: query}
        if self.count_arg:
            arguments[self.count_arg] = max_results

        endpoint, headers = self._endpoint(), self._headers()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            init = await client.post(endpoint, headers=headers, json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agentica", "version": "1.0.0"},
                },
            })
            init.raise_for_status()
            session_id = init.headers.get("Mcp-Session-Id")
            if not session_id:
                raise RuntimeError(f"MCP server {self.url} returned no Mcp-Session-Id")

            session_headers = {**headers, "Mcp-Session-Id": session_id}
            await client.post(endpoint, headers=session_headers, json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            })

            call = await client.post(endpoint, headers=session_headers, json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": self.tool_name, "arguments": arguments},
            })
            call.raise_for_status()

        payload = _parse_mcp_response(call)
        if "error" in payload:
            raise RuntimeError(f"MCP search error from {self.url}: {payload['error']}")

        blocks = (payload.get("result") or {}).get("content", [])
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        result = "\n\n".join(t for t in texts if t)
        logger.debug(f"MCP search '{query}' via {self.tool_name}, {len(result)} chars")
        return result


if __name__ == '__main__':
    import asyncio

    m = McpSearchTool()
    print(asyncio.run(m.mcp_search("what is the agentica python framework", max_results=3)))
