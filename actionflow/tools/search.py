# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import asyncio
import json
from typing import Any, Dict, Tuple, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from actionflow.config import SERPER_API_KEY
from actionflow.output import Output
from actionflow.tool import BaseTool


class SerperWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    payload: dict = Field(default_factory=lambda: {"page": 1, "num": 10})
    aiosession: Optional[aiohttp.ClientSession] = None
    proxy: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_serper(cls, values: dict) -> dict:
        if "serper_api_key" in values:
            values.setdefault("api_key", values["serper_api_key"])
            logger.warning("`serper_api_key` is deprecated, use `api_key` instead", DeprecationWarning)

        if "api_key" not in values:
            raise ValueError(
                "To use serper search engine, make sure you provide the `api_key` when constructing an object. You can obtain "
                "an API key from https://serper.dev/."
            )
        return values

    async def run(self, query: str, max_results: int = 8, as_string: bool = True, **kwargs: Any) -> str:
        """Run query through Serper and parse result async."""
        if isinstance(query, str):
            return self._process_response((await self.results([query], max_results))[0], as_string=as_string)
        else:
            results = [self._process_response(res, as_string) for res in await self.results(query, max_results)]
        return "\n".join(results) if as_string else results

    async def results(self, queries: list[str], max_results: int = 8) -> dict:
        """Use aiohttp to run query through Serper and return the results async."""

        def construct_url_and_payload_and_headers() -> Tuple[str, Dict[str, str]]:
            payloads = self.get_payloads(queries, max_results)
            url = "https://google.serper.dev/search"
            headers = self.get_headers()
            return url, payloads, headers

        url, payloads, headers = construct_url_and_payload_and_headers()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payloads, headers=headers, proxy=self.proxy) as response:
                    response.raise_for_status()
                    res = await response.json()
        else:
            async with self.aiosession.get.post(url, data=payloads, headers=headers, proxy=self.proxy) as response:
                response.raise_for_status()
                res = await response.json()

        return res

    def get_payloads(self, queries: list[str], max_results: int) -> Dict[str, str]:
        """Get payloads for Serper."""
        payloads = []
        for query in queries:
            _payload = {
                "q": query,
                "num": max_results,
            }
            payloads.append({**self.payload, **_payload})
        return json.dumps(payloads, sort_keys=True)

    def get_headers(self) -> Dict[str, str]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        return headers

    @staticmethod
    def _process_response(res: dict, as_string: bool = False) -> str:
        """Process response from SerpAPI."""
        # logger.debug(res)
        focus = ["title", "snippet", "link"]

        def get_focused(x):
            return {i: j for i, j in x.items() if i in focus}

        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif "answer_box" in res.keys() and "snippet_highlighted_words" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif "sports_results" in res.keys() and "game_spotlight" in res["sports_results"].keys():
            toret = res["sports_results"]["game_spotlight"]
        elif "knowledge_graph" in res.keys() and "description" in res["knowledge_graph"].keys():
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic"][0].keys():
            toret = res["organic"][0]["snippet"]
        else:
            toret = "No good search result found"

        toret_l = []
        if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret_l += [get_focused(res["answer_box"])]
        if res.get("organic"):
            toret_l += [get_focused(i) for i in res.get("organic")]

        return str(toret) + "\n" + str(toret_l) if as_string else toret_l


class Search(BaseTool):
    """
    This class inherits from the BaseFunction class. It defines a function for fetching the contents of a URL.
    """

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search using the Google Serper search engine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                },
            }
        }

    def execute(self, query: str, max_results: int = 8, as_string: bool = True, ignore_errors: bool = False) -> str:
        """
        Synchronously execute a search query by running the asynchronous search engine method.

        Args:
            query: The search query.
            max_results: The maximum number of results to return. Defaults to 8.
            as_string: Whether to return the results as a string or a list of dictionaries. Defaults to True.
            ignore_errors: Whether to ignore errors during the search. Defaults to False.

        Returns:
            The search results as a string or a list of dictionaries.
        """
        loop = asyncio.get_event_loop()
        try:
            return loop.run_until_complete(
                SerperWrapper(api_key=SERPER_API_KEY).run(query, max_results=max_results, as_string=as_string)
            )
        except Exception as e:
            # Handle errors in the API call
            logger.error(f"Failed to search {query} due to {e}")
            if not ignore_errors:
                raise e
            return "" if as_string else []


if __name__ == '__main__':
    output = Output('o')
    search = Search(output)
    r = search.execute("北京的新闻top3")
    print(type(r), '\n\n', r)
