# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: URL Crawler Tool
"""
import hashlib
import os
import re
import json
from urllib.parse import ParseResult, urlparse, urljoin

import requests
from bs4 import BeautifulSoup
import sys
sys.path.append("../..")
from agentica.tools.base import Tool
from agentica.utils.log import logger


def clean_text(text: str) -> str:
    """Clean text by removing control characters.
    
    Args:
        text: The raw text
        
    Returns:
        Cleaned text without control characters
    """
    if not text:
        return ""
    # Remove control characters (ASCII 0-31 except tab, newline, carriage return)
    # Also remove Unicode control characters like \u000b (vertical tab)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return cleaned


class UrlCrawlerTool(Tool):
    def __init__(
            self,
            base_dir: str = os.path.curdir,
            max_content_length: int = 16000,
    ):
        super().__init__(name="url_crawler_tool")
        self.base_dir = base_dir
        self.max_content_length = max_content_length
        self.register(self.url_crawl)

    @staticmethod
    def _generate_file_name_from_url(url: str, max_length=255) -> str:
        url_bytes = url.encode("utf-8")
        hash = hashlib.blake2b(url_bytes).hexdigest()
        parsed_url: ParseResult = urlparse(url)
        file_name = os.path.basename(url)
        prefix = f"{parsed_url.netloc}_{file_name}"
        end = hash[:min(8, max_length - len(parsed_url.netloc) - len(file_name) - 1)]
        file_name = f"{prefix}_{end}"
        return file_name

    @staticmethod
    def parse_html_to_markdown(html: str) -> str:
        """Parse HTML to markdown."""
        import markdownify

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string if soup.title else "No Title"
        # Remove javascript, style blocks, and hyperlinks
        for element in soup(["script", "style", "a"]):
            element.extract()
        # Remove other common irrelevant elements, e.g., nav, footer, etc.
        for element in soup.find_all(["nav", "footer", "aside", "form", "figure"]):
            element.extract()

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
        """Crawl a website url and return the content of the website as a json string.

        Args:
            url (str): The URL of the website to read, starting with http:// or https://
            
        Returns:
            str: The content of the website as a json string.
        """
        filename = self._generate_file_name_from_url(url)
        save_path = os.path.realpath(os.path.join(self.base_dir, filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        content = ""
        try:
            logger.debug(f"Crawling URL: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "noscript", "iframe", "svg"]):
                script.extract()

            # Handle links: Convert <a> tags to Markdown format [Text](URL)
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                if text:
                    href = a['href']
                    full_url = urljoin(url, href)
                    a.replace_with(f" [{text}]({full_url}) ")

            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            content = '\n'.join(line for line in lines if line)
            # Clean control characters from content
            content = clean_text(content)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.debug(f"Successfully crawled: {url}, saved to: {save_path}, content length: {len(content)}")
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
    # url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
    # r = m.url_crawl(url)
    # print(url, '\n\n', r)
    url = "https://baike.baidu.com/item/%E6%9D%8E%E7%91%9E"
    r = m.url_crawl(url)
    print(url, '\n\n', r)
