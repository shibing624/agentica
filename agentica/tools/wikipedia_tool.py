# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import json
from agentica.document import Document
from agentica.tools.base import Tool
from agentica.utils.log import logger


class WikipediaTool(Tool):
    def __init__(self):
        super().__init__(name="wikipedia_tool")
        self.register(self.search_wikipedia)

    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for a query.

        Args:
            query (str): The query to search for.

        Example:
            from agentica.tools.wikipedia_tool import WikipediaTool
            m = WikipediaTool()
            result = m.search_wikipedia("beijing")
            print(result)

        Returns:
            str: Relevant documents from Wikipedia.
        """
        try:
            import wikipedia  # noqa: F401
        except ImportError:
            raise ImportError(
                "The `wikipedia` package is not installed. " "Please install it via `pip install wikipedia`."
            )
        logger.info(f"Searching wikipedia for: {query}")
        data = Document(name=query, content=wikipedia.summary(query)).to_dict()
        return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    m = WikipediaTool()
    r = m.search_wikipedia("beijing")
    print(r)
