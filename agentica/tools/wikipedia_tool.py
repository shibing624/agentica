# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
from typing import Union, List
import json
from agentica.document import Document
from agentica.tools.base import Tool
from agentica.utils.log import logger


class WikipediaTool(Tool):
    def __init__(self):
        super().__init__(name="wikipedia_tool")
        self.register(self.search_wikipedia)

    def search_wikipedia_single_query(self, query: str) -> str:
        """Search Wikipedia for a query.

        Args:
            query (str): The query to search for.

        Returns:
            str: Relevant documents from Wikipedia.
        """
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "The `wikipedia` package is not installed. " "Please install it via `pip install wikipedia`."
            )
        data = Document(name=query, content=wikipedia.summary(query)).to_dict()
        logger.info(f"Searching wikipedia for: {query}, result: {data}")
        return json.dumps(data, ensure_ascii=False, indent=2)

    def search_wikipedia(self, queries: Union[List[str], str]) -> str:
        """Search Wikipedia for multiple queries.

        Args:
            queries (list | str): The queries to search for.
        Returns:
            str: Relevant documents from Wikipedia for all queries.
        """
        if isinstance(queries, str):
            return self.search_wikipedia_single_query(queries)
        all_results = {}
        for query in queries:
            res = self.search_wikipedia_single_query(query)
            all_results[query] = res
        return json.dumps(all_results, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    m = WikipediaTool()
    r = m.search_wikipedia("beijing")
    print(r)
