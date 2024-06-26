# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Jina 有2个功能：
1）Jina Reader, 从URL获取内容，输出markdown格式的网页内容
2）Jina Search, 输出查询query，结合Reader的检索结果，输出markdown格式的检索网页内容，用于LLM做RAG。
    如：苹果的最新产品是啥？ -> ## 苹果的最新产品是Apple Intelligence，iOS 18。
"""
from os import getenv
from typing import Optional

import requests

from actionflow.tool import Toolkit
from actionflow.utils.log import logger


class JinaTool(Toolkit):
    def __init__(
            self,
            api_key: Optional[str] = None,
            jina_reader: bool = True,
            jina_search: bool = False,
    ):
        super().__init__(name="jina_tool")

        self.api_key = api_key or getenv("JINA_API_KEY")
        if not self.api_key:
            logger.warning("No JINA_API_KEY key provided, faster if you have one.")
        else:
            logger.debug(f"Use JINA_API_KEY: {'*' * 10 + self.api_key[-4:]}")

        if jina_reader:
            self.register(self.jina_url_reader)
        if jina_search:
            self.register(self.jina_search)

    def jina_url_reader(self, url: str, timeout: int = 60) -> str:
        """
        Crawls a website using Jina's website-content-crawler actor.

        :param url: str, The URL to crawl.
        :param timeout: int, The timeout for the crawling.

        :return: str, The result of the crawling, Markdown format.
        """
        if url is None:
            return "No URLs provided"

        logger.debug(f"Crawling URL: {url}")

        result: str = ""
        headers = {'X-Return-Format': 'markdown', 'X-Timeout': f'{timeout}'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        try:
            response = requests.get(f'https://r.jina.ai/{url}', headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch URL. HTTP status code: {response.status_code}")
            else:
                result = response.text
        except Exception as e:
            logger.error(f"Failed to fetch URL. Error: {e}")
        return result

    def jina_search(self, query: str) -> str:
        """
        Search using Jina's web-search actor.

        :param query: The URLs to scrape.

        :return: The results of the search.
        """
        if query is None:
            return "No query provided"
        query = query.strip()
        logger.debug(f"Search query: {query}")
        result: str = ""
        headers = {'X-Return-Format': 'markdown'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        try:
            response = requests.get(f'https://s.jina.ai/{query}', headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch URL. HTTP status code: {response.status_code}")
            else:
                result = response.text
        except Exception as e:
            logger.error(f"Failed to fetch URL. Error: {e}")
        return result


if __name__ == '__main__':
    m = JinaTool(jina_reader=True, jina_search=True)
    text = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
    r = m.jina_url_reader(text)
    print(text, '\n\n', r)

    query = "苹果的最新产品是啥？"
    r = m.jina_search(query)
    print(query, '\n\n', r)
