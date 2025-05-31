# -*- encoding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description:
Search papers in DBLP API.
For detail usage of the DBLP API
please refer to https://dblp.org/faq/How+can+I+fetch+DBLP+data.html
"""
import json

import requests

from agentica.tools.base import Tool
from agentica.utils.log import logger


class DblpTool(Tool):
    def __init__(self):
        super().__init__(name="Dblp_tool")
        self.register(self.search_dblp_and_return_articles)

    def search_dblp_and_return_articles(
            self,
            question: str,
            num_results: int = 20,
            start: int = 0,
            num_completion: int = 10,
    ) -> str:
        """Search papers in the DBLP database.

        Args:
            question (`str`):
                The search query string.
            num_results (`int`, defaults to `20`):
                The number of search results to return.
            start (`int`, defaults to `0`):
                The index of the first search result to return.
            num_completion (`int`, defaults to `10`):
                The number of completions to generate.
        Example:
            from agentica.tools.dblp_tool import DblpTool
            m = DblpTool()
            search_results = m.search_dblp_and_return_articles(question="Extreme Learning Machine")
            print(search_results)

        Returns:
            str, a JSON of the articles with title, authors, venue, pages, year, type, DOI, and URL.
        """
        articles = []
        logger.info(f"Searching dblp for: {question}")
        url = "https://dblp.org/search/publ/api"
        params = {
            "q": question,
            "format": "json",
            "h": num_results,
            "f": start,
            "c": num_completion,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()

            hits = search_results.get("result", {}).get("hits", {}).get("hit", [])
            for hit in hits:
                info = hit.get("info", {})
                title = info.get("title", "No title available")
                venue = info.get("venue", "No venue available")
                pages = info.get("pages", "No page information")
                year = info.get("year", "Year not specified")
                pub_type = info.get("type", "Type not specified")
                doi = info.get("doi", "No DOI available")
                url = info.get("url", "No URL available")
                authors_info = info.get("authors", {}).get("author", [])
                if isinstance(
                        authors_info,
                        dict,
                ):  # Check if there's only one author in a dict format
                    authors_info = [authors_info]
                authors = ", ".join(
                    [author["text"] for author in authors_info if "text" in author],
                )
                data = {
                    "title": title,
                    "venue": venue,
                    "pages": pages,
                    "year": year,
                    "type": pub_type,
                    "doi": doi,
                    "url": url,
                    "authors": authors,
                }
                articles.append(data)
        except Exception as e:
            logger.error(f"Error processing article: {e}")
        return json.dumps(articles, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    m = DblpTool()
    search_results = m.search_dblp_and_return_articles(question="Extreme Learning Machine")
    print(search_results)
