# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import http.client
import json
import os

from typing import Dict, Optional, Union, List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentica.tools.base import Tool
from agentica.utils.log import logger


class SerperWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    payload: dict = Field(default_factory=lambda: {"page": 1, "num": 10})
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

    def get_headers(self) -> Dict[str, str]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        return headers

    def run(self, query: str, max_results: int = 8, as_string: bool = True):
        """Run query through Serper and parse result"""
        headers = self.get_headers()

        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": query, "num": max_results})
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            res_data = res.read()
            data = json.loads(res_data.decode("utf-8"))
            logger.debug(data)
            # Check for specific error messages
            if "error" in data:
                raise ValueError(f"Error from Serper API https://serper.dev/: {data['error']}")
            if "message" in data and "Unauthorized" in data["message"]:
                raise ValueError("Unauthorized access to Serper API https://serper.dev/. Check your API key.")
            if "message" in data and "Not enough credits" in data["message"]:
                raise ValueError("Not enough credits for Serper API.")

            res = self._process_response(data, as_string=as_string)
        except Exception as e:
            msg = f"Failed to search `{query}` due to {e}"
            logger.error(msg)
            res = msg
        return res

    @staticmethod
    def _process_response(res: dict, as_string: bool = False) -> Union[str, List[dict]]:
        """Process response from SerpAPI."""
        # logger.debug(res)
        focus = ["title", "snippet", "link"]
        def get_focused(x):
            return {i: j for i, j in x.items() if i in focus}
        toret_l = []
        if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret_l += [get_focused(res["answer_box"])]
        if res.get("organic"):
            toret_l += [get_focused(i) for i in res.get("organic")]
        return f"{toret_l}" if as_string else toret_l


class SearchSerperTool(Tool):
    """
    This class inherits from the BaseFunction class. It defines a function for fetching the contents of a URL.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            timeout: int = 60
    ):
        super().__init__(name="search_serper")

        self.timeout: Optional[int] = timeout
        self.api_key: Optional[str] = api_key or os.getenv("SERPER_API_KEY")
        self.register(self.search_google)

    def search_google_single_query(
            self,
            query: str,
            max_results: int = 8,
            as_string: bool = True,
    ) -> str:
        """
        Use this function to search google for a query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return. Defaults to 8.
            as_string: Whether to return the results as a string or a list of dictionaries. Defaults to True.

        Returns:
            The search results as a string or a list of dictionaries.
        """
        res =  SerperWrapper(api_key=self.api_key).run(query, max_results=max_results, as_string=as_string)
        logger.debug(f"Search google for query: {query}, result: {res}")
        return res

    def search_google(self, queries: Union[List[str], str], max_results: int = 8) -> str:
        """
        Search Google for information. Use this tool first to find relevant web pages before visiting them.
        This function searches google for one or more queries and returns search results with titles, snippets and URLs.

        Args:
            queries: The search query string, or a list of search queries.
            max_results: The maximum number of results to return for each query. Defaults to 8.

        Returns:
            Search results containing titles, snippets and URLs that can be used to visit pages for more details.
        """
        if isinstance(queries, str):
            return self.search_google_single_query(queries, max_results=max_results, as_string=True)
        all_results = {}
        for query in queries:
            res = self.search_google_single_query(query, max_results=max_results, as_string=True)
            all_results[query] = res
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == '__main__':
    search = SearchSerperTool()
    print(search.api_key)
    r = search.search_google(["湖北的新闻top3", "北京的娱乐新闻"])
    print(type(r), '\n\n', r)
