import json
from typing import Any, Dict, Optional

from agentica.tools.base import Tool
from agentica.utils.log import logger

try:
    import newspaper
except ImportError:
    raise ImportError("`newspaper4k` not installed. Please run `pip install newspaper4k lxml_html_clean`.")


class NewspaperTool(Tool):
    def __init__(
            self,
            read_article: bool = True,
            include_summary: bool = False,
            article_length: Optional[int] = None,
    ):
        super().__init__(name="newspaper_tool")

        self.include_summary: bool = include_summary
        self.article_length: Optional[int] = article_length
        if read_article:
            self.register(self.read_article, concurrency_safe=True, is_read_only=True)

    def get_article_data(self, url: str) -> Dict[str, Any]:
        """Read and get article data from a URL.

        Args:
            url (str): The URL of the article.

        Returns:
            Dict[str, Any]: The article data (may be empty if newspaper parsed
            nothing useful).
        """
        article = newspaper.article(url)
        article_data: Dict[str, Any] = {}
        if article.title:
            article_data["title"] = article.title
        if article.authors:
            article_data["authors"] = article.authors
        if article.text:
            article_data["text"] = article.text
        if self.include_summary and article.summary:
            article_data["summary"] = article.summary

        # Some sites expose a broken publish_date object that raises on access
        # — treat that narrow failure as "no publish_date" rather than a full
        # failure of article parsing.
        try:
            if article.publish_date:
                article_data["publish_date"] = article.publish_date.isoformat()
        except (AttributeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to extract publish_date from {url}: {e}")

        return article_data

    def read_article(self, url: str) -> str:
        """Use this function to read an article from a URL.

        Args:
            url (str): The URL of the article.

        Returns:
            str: JSON containing the article author, publish date, and text.
        """
        article_data = self.get_article_data(url)
        if not article_data:
            raise ValueError(f"No article data parsed from {url}")

        if self.article_length and "text" in article_data:
            article_data["text"] = article_data["text"][:self.article_length]

        return json.dumps(article_data, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    url = "https://sputniknews.cn/20241222/1063319016.html"
    newspaper_tool = NewspaperTool(read_article=True, include_summary=False, article_length=500)
    article_data = newspaper_tool.read_article(url)
    print(article_data)
