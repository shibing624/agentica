# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: URL Crawler Tool
"""

import json
import random
import ssl
import time
from typing import Set, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import httpx

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
except ImportError:
    raise ImportError("The `bs4` package is not installed. Please install it via `pip install beautifulsoup4`.")

from actionflow.tool import Toolkit
from actionflow.utils.log import logger
from actionflow.utils.misc import literal_similarity

# Create a default context for HTTPS requests (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context


class UrlCrawlerTool(Toolkit):
    max_depth: int = 2
    max_links: int = 10

    _visited: Set[str] = set()
    _urls_to_crawl: List[Tuple[str, int]] = []

    def __init__(
            self,
            max_depth: int = None,
            max_links: int = None,
    ):
        super().__init__(name="url_crawler_tool")
        self.max_depth = max_depth if max_depth is not None else self.max_depth
        self.max_links = max_links if max_links is not None else self.max_links
        self.register(self.url_crawl)

    def delay(self, min_seconds=1, max_seconds=3):
        """
        Introduce a random delay.

        :param min_seconds: Minimum number of seconds to delay. Default is 1.
        :param max_seconds: Maximum number of seconds to delay. Default is 3.
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)

    def _get_primary_domain(self, url: str) -> str:
        """
        Extract primary domain from the given URL.

        :param url: The URL to extract the primary domain from.
        :return: The primary domain.
        """
        domain_parts = urlparse(url).netloc.split(".")
        # Return primary domain (excluding subdomains)
        return ".".join(domain_parts[-2:])

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main content from a BeautifulSoup object.

        :param soup: The BeautifulSoup object to extract the main content from.
        :return: The main content in markdown format.
        """
        # Convert the whole document to markdown
        md = self._convert_to_markdown(soup)
        return md

    def _convert_to_markdown(self, element: Tag) -> str:
        """
        Converts an HTML element to markdown format.

        :param element: The HTML element to convert.
        :return: The element's content in markdown format.
        """
        markdown = ""
        for child in element.children:
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    markdown += text
            elif child.name in ["script", "style"]:
                continue  # Skip script style tags
            elif child.name == "img":
                continue
            elif child.name == "a":
                continue
            elif child.name in ["p", "h1", "h2", "h3", "h4", "h5", "h6"]:
                child_text = self._convert_to_markdown(child).strip()
                if child_text:
                    markdown += f"\n\n{child_text}"
            elif child.name == "ul":
                child_text = self._convert_to_markdown(child).strip()
                if child_text:
                    markdown += f"\n\n{child_text}"
            elif child.name == "ol":
                child_text = self._convert_to_markdown(child).strip()
                if child_text:
                    markdown += f"\n\n{child_text}"
            elif child.name == "li":
                child_text = self._convert_to_markdown(child).strip()
                if child_text:
                    markdown += f"\n* {child_text}"
            else:
                child_text = self._convert_to_markdown(child).strip()
                if child_text:
                    markdown += child_text

        return markdown.strip()

    def crawl(self, url: str, starting_depth: int = 1) -> Dict[str, str]:
        """
        Crawls a website and returns a dictionary of URLs and their corresponding content.

        Parameters:
        - url (str): The starting URL to begin the crawl.
        - starting_depth (int, optional): The starting depth level for the crawl. Defaults to 1.

        Returns:
        - Dict[str, str]: A dictionary where each key is a URL and the corresponding value is the main
                          content extracted from that URL.

        Note:
        The function focuses on extracting the main content by prioritizing content inside common HTML tags
        The crawler will also respect the `max_depth` attribute of the WebCrawler class, ensuring it does not
        crawl deeper than the specified depth.
        """
        num_links = 0
        crawler_result: Dict[str, str] = {}
        primary_domain = self._get_primary_domain(url)
        # Add starting URL with its depth to the global list
        self._urls_to_crawl.append((url, starting_depth))
        while self._urls_to_crawl:
            # Unpack URL and depth from the global list
            current_url, current_depth = self._urls_to_crawl.pop(0)

            # Skip if
            # - URL is already visited
            # - does not end with the primary domain,
            # - exceeds max depth
            # - exceeds max links
            if (
                    current_url in self._visited
                    or not urlparse(current_url).netloc.endswith(primary_domain)
                    or current_depth > self.max_depth
                    or num_links >= self.max_links
            ):
                continue

            self._visited.add(current_url)
            self.delay()

            try:
                logger.debug(f"Crawling: {current_url}")
                response = httpx.get(current_url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract main content
                main_content = self._extract_main_content(soup)
                if main_content:
                    is_duplicate = False
                    for _url, c in crawler_result.items():
                        if literal_similarity(main_content, c) > 0.8:
                            logger.debug(f"Duplicate content found: {_url}, skipping...")
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        crawler_result[current_url] = main_content
                        num_links += 1

                # Add found URLs to the global list, with incremented depth
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current_url, link["href"])
                    parsed_url = urlparse(full_url)
                    if parsed_url.netloc.endswith(primary_domain) and not any(
                            parsed_url.path.endswith(ext) for ext in [".pdf", ".jpg", ".png"]
                    ):
                        if full_url not in self._visited and (full_url, current_depth + 1) not in self._urls_to_crawl:
                            self._urls_to_crawl.append((full_url, current_depth + 1))

            except Exception as e:
                logger.debug(f"Failed to crawl: {current_url}: {e}")
                pass

        return crawler_result

    def url_crawl(self, url: str) -> str:
        """
        Reads a website and returns a json str.

        This function first converts the website into a dictionary of URLs and their corresponding content.
        Then iterates through the dictionary and returns chunks of content.

        :param url: The URL of the website to read.
        :return: str
        """
        logger.info(f"Crawling URL: {url}")
        crawler_result = self.crawl(url)
        result = json.dumps(crawler_result, indent=2, ensure_ascii=False)
        return result


if __name__ == '__main__':
    m = UrlCrawlerTool(max_depth=2)
    from actionflow.utils.log import set_log_level_to_debug

    set_log_level_to_debug()

    url = "https://www.jpmorgan.com/insights/business/business-planning/409a-valuations-a-guide-for-startups"
    r = m.url_crawl(url)
    print(url, '\n\n', r)
