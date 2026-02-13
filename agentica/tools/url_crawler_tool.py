# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: URL Crawler Tool
"""
import asyncio
import hashlib
import os
import re
import json
from urllib.parse import ParseResult, urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
import sys
sys.path.append("../..")
from agentica.tools.base import Tool
from agentica.config import AGENTICA_HOME
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
    # Default cache directory for crawled web pages
    DEFAULT_CACHE_DIR = os.path.join(AGENTICA_HOME, "web_cache")

    def __init__(
            self,
            work_dir: str = None,
            max_content_length: int = 16000,
    ):
        """Initialize UrlCrawlerTool.

        Args:
            work_dir: Directory to save crawled web pages.
                      Defaults to ~/.cache/agentica/web_cache/
            max_content_length: Maximum length of returned content
        """
        super().__init__(name="url_crawler_tool")
        # Use default cache directory if not specified
        self.work_dir = work_dir if work_dir else self.DEFAULT_CACHE_DIR
        self.max_content_length = max_content_length
        # Ensure cache directory exists
        os.makedirs(self.work_dir, exist_ok=True)
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

    def _detect_encoding(self, response) -> str:
        """Detect the correct encoding for the response.

        Priority:
        1. Content-Type header charset
        2. HTML meta charset tag
        3. apparent_encoding (chardet detection)
        4. Default to utf-8

        Args:
            response: httpx.Response object

        Returns:
            str: Detected encoding
        """
        # 1. Check Content-Type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'charset=' in content_type:
            charset = content_type.split('charset=')[-1].split(';')[0].strip()
            if charset:
                return charset

        # 2. Check HTML meta tag for charset
        # Use raw content bytes to avoid encoding issues
        raw_content = response.content[:4096]  # Check first 4KB
        try:
            # Try to decode as ASCII to find charset declaration
            text_sample = raw_content.decode('ascii', errors='ignore')

            # Look for <meta charset="xxx">
            charset_match = re.search(r'<meta[^>]+charset=["\']?([^"\'\s>]+)', text_sample, re.IGNORECASE)
            if charset_match:
                return charset_match.group(1)

            # Look for <meta http-equiv="Content-Type" content="text/html; charset=xxx">
            content_type_match = re.search(
                r'<meta[^>]+content=["\'][^"\']*charset=([^"\'\s;]+)',
                text_sample,
                re.IGNORECASE
            )
            if content_type_match:
                return content_type_match.group(1)
        except Exception:
            pass

        # 3. Use charset_encoding from httpx (similar to apparent_encoding)
        apparent = response.encoding
        if apparent and apparent != 'utf-8':
            apparent_lower = apparent.lower()
            # GB2312/GBK/GB18030 are all compatible, use GB18030 for best coverage
            if apparent_lower in ('gb2312', 'gbk', 'gb18030'):
                return 'gb18030'
            return apparent

        # 4. Default to UTF-8
        return 'utf-8'

    async def url_crawl(self, url: str) -> str:
        """Crawl a website url and return the content of the website as a json string.

        Args:
            url (str): The URL of the website to read, starting with http:// or https://

        Returns:
            str: The content of the website as a json string.
        """
        filename = self._generate_file_name_from_url(url)
        save_path = os.path.realpath(os.path.join(self.work_dir, filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        content = ""
        try:
            logger.debug(f"Crawling URL: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10, follow_redirects=True)
                response.raise_for_status()

            # Improved encoding detection for Chinese websites
            encoding = self._detect_encoding(response)
            text = response.content.decode(encoding, errors='replace')

            soup = BeautifulSoup(text, 'html.parser')
            for script in soup(["script", "style", "noscript", "iframe", "svg"]):
                script.extract()

            # Handle links: Convert <a> tags to Markdown format [Text](URL)
            for a in soup.find_all('a', href=True):
                link_text = a.get_text(strip=True)
                if link_text:
                    href = a['href']
                    full_url = urljoin(url, href)
                    a.replace_with(f" [{link_text}]({full_url}) ")

            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            content = '\n'.join(line for line in lines if line)
            # Clean control characters from content
            content = clean_text(content)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_file, save_path, content)
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

    @staticmethod
    def _write_file(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


if __name__ == '__main__':
    import asyncio

    m = UrlCrawlerTool()
    url = "https://baike.baidu.com/item/%E6%9D%8E%E7%91%9E"
    r = asyncio.run(m.url_crawl(url))
    print(url, '\n\n', r)
