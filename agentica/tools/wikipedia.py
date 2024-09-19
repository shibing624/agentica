# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import json
import ssl

try:
    import wikipedia  # noqa: F401
except ImportError:
    raise ImportError(
        "The `wikipedia` package is not installed. " "Please install it via `pip install wikipedia`."
    )
from agentica.document import Document
from agentica.tools.base import Toolkit
from agentica.utils.log import logger

# Create a default context for HTTPS requests (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context


class WikipediaTool(Toolkit):
    def __init__(self, ):
        super().__init__(name="wikipedia_tool")

        self.register(self.search_wikipedia)

    def search_wikipedia(self, query: str) -> str:
        """Searches Wikipedia for a query.

        :param query: The query to search for.
        :return: Relevant documents from wikipedia.
        """
        logger.info(f"Searching wikipedia for: {query}")
        return json.dumps(
            Document(name=query, content=wikipedia.summary(query)).to_dict(),
            ensure_ascii=False, indent=2, sort_keys=True
        )
