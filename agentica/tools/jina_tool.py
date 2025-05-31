# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Jina 有2个功能：
1）Jina Reader, 从URL获取内容，输出markdown格式的网页内容
2）Jina Search, 输出查询query，结合Reader的检索结果，输出markdown格式的检索网页内容，用于LLM做RAG。
    如：苹果的最新产品是啥？ -> ## 苹果的最新产品是Apple Intelligence，iOS 18。
"""
import hashlib
import os
import requests
from os import getenv
from typing import Optional
from urllib.parse import urlparse

from agentica.tools.base import Tool
from agentica.utils.log import logger


class JinaTool(Tool):
    def __init__(
            self,
            api_key: Optional[str] = None,
            jina_reader: bool = True,
            jina_search: bool = True,
            max_content_length: int = 8000,
            work_dir: str = None,
    ):
        super().__init__(name="jina_tool")
        self.api_key = api_key or getenv("JINA_API_KEY")
        self.max_content_length = max_content_length
        self.work_dir = work_dir or os.path.curdir
        if self.api_key:
            logger.debug(f"Use JINA_API_KEY: {'*' * 10 + self.api_key[-4:]}")

        if jina_reader:
            self.register(self.jina_url_reader)
        if jina_search:
            self.register(self.jina_search)

    def _get_headers(self) -> dict:
        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _trim_content(self, content: str) -> str:
        """Trim to the maximum allowed length."""
        if len(content) > self.max_content_length:
            truncated = content[: self.max_content_length]
            return truncated + "... (content truncated)"
        return content

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

    def jina_url_reader(self, url: str) -> str:
        """Reads a URL and returns the html text content using Jina Reader API.

        Args:
            url: str, The URL to read.

        Example:
            from agentica.tools.jina_tool import JinaTool
            m = JinaTool(jina_reader=True, jina_search=True)
            url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
            r = m.jina_url_reader(url)
            print(url, '\n\n', r)

        Returns:
            str, The result of the crawling html text content, Markdown format.
        """
        logger.debug(f"Reading URL: {url}")

        try:
            data = {'url': url}
            response = requests.post('https://r.jina.ai/', headers=self._get_headers(), json=data)
            response.raise_for_status()
            content = response.text
            result = self._trim_content(content)
        except Exception as e:
            msg = f"Error reading URL: {str(e)}"
            logger.error(msg)
            result = msg
            content = ''
        if content:
            filename = self._generate_file_name_from_url(url)
            save_path = os.path.realpath(os.path.join(str(self.work_dir), filename))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Url: {url}, saved content to: {save_path}")
        return result

    def jina_search(self, query: str) -> str:
        """Performs a web search using Jina Reader API and returns the search content.

        Args:
            query (str): The query to search for.

        Example:
            from agentica.tools.jina_tool import JinaTool
            m = JinaTool(jina_reader=True, jina_search=True)
            query = "苹果的最新产品是啥？"
            r = m.jina_search(query)
            print(query, '\n\n', r)

        Returns:
            str: The results of the search.
        """
        query = query.strip()
        logger.debug(f"Search query: {query}")
        url = f'https://s.jina.ai/{query}'
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            content = response.text
            result = self._trim_content(content)
        except Exception as e:
            content = ''
            msg = f"Error performing search: {str(e)}"
            logger.error(msg)
            result = msg
        if content:
            filename = self._generate_file_name_from_url(url)
            save_path = os.path.realpath(os.path.join(str(self.work_dir), filename))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Query: {query}, saved content to: {save_path}")
        return result


if __name__ == '__main__':
    m = JinaTool(jina_reader=True, jina_search=True)
    url = "https://raw.githubusercontent.com/shibing624/agentica/refs/heads/main/agentica/tools/base.py"
    r = m.jina_url_reader(url)
    print(r)

    url = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
    r = m.jina_url_reader(url)
    print(url, '\n\n', r)

    query = "苹果的最新产品是啥？"
    r = m.jina_search(query)
    print(query, '\n\n', r)
