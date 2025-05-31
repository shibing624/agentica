# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: URL Crawler Tool
"""
import hashlib
import os
import re
import json
from urllib.parse import urlparse

import markdownify
import requests
from bs4 import BeautifulSoup

from agentica.tools.base import Tool
from agentica.utils.log import logger


class UrlCrawlerTool(Tool):
    def __init__(
            self,
            base_dir: str = os.path.curdir,
            max_content_length: int = 8000,
    ):
        super().__init__(name="url_crawler_tool")
        self.base_dir = base_dir
        self.max_content_length = max_content_length
        self.register(self.url_crawl)

    @staticmethod
    def _generate_file_name_from_url(url: str, max_length=255) -> str:
        url_bytes = url.encode("utf-8")
        hash = hashlib.blake2b(url_bytes).hexdigest()
        parsed_url = urlparse(url)
        file_name = os.path.basename(url)
        prefix = f"{parsed_url.netloc}_{file_name}"
        end = hash[:min(8, max_length - len(parsed_url.netloc) - len(file_name) - 1)]
        file_name = f"{prefix}_{end}"
        return file_name

    @staticmethod
    def parse_html_to_markdown(html: str, url: str = None) -> str:
        """Parse HTML to markdown."""
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string if soup.title else "No Title"
        # Remove javascript, style blocks, and hyperlinks
        for element in soup(["script", "style", "a"]):
            element.extract()
        # Remove other common irrelevant elements, e.g., nav, footer, etc.
        for element in soup.find_all(["nav", "footer", "aside", "form", "figure"]):
            element.extract()

        # Convert to markdown -- Wikipedia gets special attention to get a clean version of the page
        if isinstance(url, str) and "wikipedia.org" in url:
            body_elm = soup.find("div", {"id": "mw-content-text"})
            title_elm = soup.find("span", {"class": "mw-page-title-main"})

            if body_elm:
                # What's the title
                main_title = title
                if title_elm and len(title_elm) > 0:
                    main_title = title_elm.string
                webpage_text = "# " + main_title + "\n\n" + markdownify.MarkdownConverter().convert_soup(body_elm)
            else:
                webpage_text = markdownify.MarkdownConverter().convert_soup(soup)
        else:
            webpage_text = markdownify.MarkdownConverter().convert_soup(soup)

        # Convert newlines
        webpage_text = re.sub(r"\r\n", "\n", webpage_text)
        webpage_text = re.sub(r"\n{2,}", "\n\n", webpage_text).strip()
        webpage_text = "# " + title + "\n\n" + webpage_text
        return webpage_text

    def _trim_content(self, content: str) -> str:
        """Trim to the maximum allowed length."""
        if len(content) > self.max_content_length:
            truncated = content[: self.max_content_length]
            return truncated + "... (content truncated)"
        return content

    def url_crawl(self, url: str) -> str:
        """Crawl a website and return the content of the website as a json string.

        Args:
            url (str): The URL of the website to read.

        Example:
            from agentica.tools.url_crawler_tool import UrlCrawlerTool
            m = UrlCrawlerTool()
            url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
            r = m.url_crawl(url)
            print(url, '\n\n', r)

        Returns:
            str: The content of the website as a json string.
        """
        filename = self._generate_file_name_from_url(url)
        save_path = os.path.realpath(os.path.join(self.base_dir, filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        content = ""
        try:
            logger.info(f"Crawling URL: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            }
            response = requests.get(url, stream=True, headers=headers, timeout=60, verify=False)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                # Get the content of the response
                html = "".join(chunk for chunk in response.iter_content(chunk_size=8192, decode_unicode=True))
                content = self.parse_html_to_markdown(html, url)

                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        content += chunk.decode("utf-8", errors="ignore")
            logger.debug(f"Successfully crawled: {url}, saved to: {save_path}")
        except Exception as e:
            logger.debug(f"Failed to crawl: {url}: {e}")
        crawler_result = {
            "url": url,
            "content": self._trim_content(content),
            "save_path": save_path,
        }
        result = json.dumps(crawler_result, ensure_ascii=False)
        return result


if __name__ == '__main__':
    m = UrlCrawlerTool()
    url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
    r = m.url_crawl(url)
    print(url, '\n\n', r)
