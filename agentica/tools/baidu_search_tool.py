#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Charles on 2018/10/10
# Function:

import re
import requests
from bs4 import BeautifulSoup
import json
from typing import Optional, List, Dict, Any, Union

from agentica.tools.base import Tool
from agentica.utils.log import logger


def clean_text(text: str) -> str:
    """Clean text by removing control characters, empty lines and extra spaces.
    
    Args:
        text: The raw text
        
    Returns:
        Cleaned text without control characters, empty lines and extra spaces
    """
    if not text:
        return ""
    # Remove control characters (ASCII 0-31 except tab, newline, carriage return)
    # Also remove Unicode control characters like \u000b (vertical tab)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Remove empty lines and collapse multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

ABSTRACT_MAX_LENGTH = 300  # abstract max length

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Ubuntu Chromium/49.0.2623.108 Chrome/49.0.2623.108 Safari/537.36',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; pt-BR) AppleWebKit/533.3 '
    '(KHTML, like Gecko)  QtWeb Internet Browser/3.7 http://www.QtWeb.net',
    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/41.0.2228.0 Safari/537.36',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, '
    'like Gecko) ChromePlus/4.0.222.3 Chrome/4.0.222.3 Safari/532.2',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4pre) '
    'Gecko/20070404 K-Ninja/2.1.3',
    'Mozilla/5.0 (Future Star Technologies Corp.; Star-Blade OS; x86_64; U; '
    'en-US) iNet Browser 4.7',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.13) '
    'Gecko/20080414 Firefox/2.0.0.13 Pogo/2.0.0.13.6866'
]

# 请求头信息
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    "Referer": "https://www.baidu.com/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9"
}

baidu_host_url = "https://www.baidu.com"
baidu_search_url = "https://www.baidu.com/s?ie=utf-8&tn=baidu&wd="

session = requests.Session()
session.headers = HEADERS


def search(keyword, num_results=10, debug=False):
    """
    通过关键字进行搜索
    :param keyword: 关键字
    :param num_results： 指定返回的结果个数
    :param debug: 是否开启debug模式
    :return: 结果列表
    """
    if not keyword:
        return None

    list_result = []
    page = 1

    # 起始搜索的url
    next_url = baidu_search_url + keyword

    # 循环遍历每一页的搜索结果，并返回下一页的url
    while len(list_result) < num_results:
        data, next_url = parse_html(next_url, rank_start=len(list_result))
        if data:
            list_result += data
            if debug:
                logger.debug("---searching[{}], finish parsing page {}, results number={}: ".format(
                    keyword, page, len(data)))
                for d in data:
                    logger.debug(str(d))

        if not next_url:
            if debug:
                logger.debug(u"already search the last page。")
            break
        page += 1

    if debug:
        logger.debug("\n---search [{}] finished. total results number={}！".format(keyword, len(list_result)))
    return list_result[: num_results] if len(list_result) > num_results else list_result


def parse_html(url, rank_start=0, debug=0):
    """
    解析处理结果
    :param url: 需要抓取的 url
    :return:  结果列表，下一页的url
    """
    try:
        res = session.get(url=url)
        res.encoding = "utf-8"
        root = BeautifulSoup(res.text, "lxml")

        list_data = []
        div_contents = root.find("div", id="content_left")
        for div in div_contents.contents:
            if type(div) != type(div_contents):
                continue

            class_list = div.get("class", [])
            if not class_list:
                continue

            if "c-container" not in class_list:
                continue

            title = ''
            url = ''
            abstract = ''
            try:
                # 遍历所有找到的结果，取得标题和概要内容（50字以内）
                if "xpath-log" in class_list:
                    if div.h3:
                        title = div.h3.text.strip()
                        url = div.h3.a['href'].strip()
                    else:
                        title = div.text.strip().split("\n", 1)[0]
                        if div.a:
                            url = div.a['href'].strip()

                    if div.find("div", class_="c-abstract"):
                        abstract = div.find("div", class_="c-abstract").text.strip()
                    elif div.div:
                        abstract = div.div.text.strip()
                    else:
                        abstract = div.text.strip().split("\n", 1)[1].strip()
                elif "result-op" in class_list:
                    if div.h3:
                        title = div.h3.text.strip()
                        url = div.h3.a['href'].strip()
                    else:
                        title = div.text.strip().split("\n", 1)[0]
                        url = div.a['href'].strip()
                    if div.find("div", class_="c-abstract"):
                        abstract = div.find("div", class_="c-abstract").text.strip()
                    elif div.div:
                        abstract = div.div.text.strip()
                    else:
                        # abstract = div.text.strip()
                        abstract = div.text.strip().split("\n", 1)[1].strip()
                else:
                    if div.get("tpl", "") != "se_com_default":
                        if div.get("tpl", "") == "se_st_com_abstract":
                            if len(div.contents) >= 1:
                                title = div.h3.text.strip()
                                if div.find("div", class_="c-abstract"):
                                    abstract = div.find("div", class_="c-abstract").text.strip()
                                elif div.div:
                                    abstract = div.div.text.strip()
                                else:
                                    abstract = div.text.strip()
                        else:
                            if len(div.contents) >= 2:
                                if div.h3:
                                    title = div.h3.text.strip()
                                    url = div.h3.a['href'].strip()
                                else:
                                    title = div.contents[0].text.strip()
                                    url = div.h3.a['href'].strip()
                                # abstract = div.contents[-1].text
                                if div.find("div", class_="c-abstract"):
                                    abstract = div.find("div", class_="c-abstract").text.strip()
                                elif div.div:
                                    abstract = div.div.text.strip()
                                else:
                                    abstract = div.text.strip()
                    else:
                        if div.h3:
                            title = div.h3.text.strip()
                            url = div.h3.a['href'].strip()
                        else:
                            title = div.contents[0].text.strip()
                            url = div.h3.a['href'].strip()
                        if div.find("div", class_="c-abstract"):
                            abstract = div.find("div", class_="c-abstract").text.strip()
                        elif div.div:
                            abstract = div.div.text.strip()
                        else:
                            abstract = div.text.strip()
            except Exception as e:
                if debug:
                    logger.debug("catch exception duration parsing page html, e={}".format(e))
                continue

            if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                abstract = abstract[:ABSTRACT_MAX_LENGTH]

            rank_start += 1
            list_data.append({"title": title, "abstract": abstract, "url": url, "rank": rank_start})

        # 找到下一页按钮
        next_btn = root.find_all("a", class_="n")

        # 已经是最后一页了，没有下一页了，此时只返回数据不再获取下一页的链接
        if len(next_btn) <= 0 or u"上一页" in next_btn[-1].text:
            return list_data, None

        next_url = baidu_host_url + next_btn[-1]["href"]
        return list_data, next_url
    except Exception as e:
        if debug:
            logger.debug(u"catch exception duration parsing page html, e：{}".format(e))
        return None, None


class BaiduSearchTool(Tool):
    """
    BaiduSearch is a toolkit for searching Baidu easily.

    Args:
        num_max_results (Optional[int]): A number of maximum results.
        headers (Optional[Any]): Headers to be used in the search request.
        debug (Optional[bool]): Enable debug output.
    """

    def __init__(
            self,
            num_max_results: Optional[int] = 5,
            headers: Optional[Any] = None,
            timeout: Optional[int] = 10,
            debug: Optional[bool] = False,
    ):
        super().__init__(name="baidusearch")
        self.num_max_results = num_max_results
        self.headers = headers
        self.timeout = timeout
        self.debug = debug
        self.register(self.baidu_search)

    def baidu_search_single_query(self, query: str, max_results: int = 5) -> str:
        """Execute Baidu search and return results

        Args:
            query (str): Search keyword
            max_results (int, optional): Maximum number of results to return, default 5

        Returns:
            str: A JSON formatted string containing the search results.
        """
        max_results = max_results or self.num_max_results
        results = search(keyword=query, num_results=max_results, debug=self.debug)
        res: List[Dict[str, str]] = []
        for idx, item in enumerate(results, 1):
            res.append(
                {
                    "title": clean_text(item.get("title", "")),
                    "url": item.get("url", ""),
                    "content": clean_text(item.get("abstract", "")),
                    "rank": str(idx),
                }
            )
        logger.debug(f"Searching Baidu for: {query}, result: {res}")
        return json.dumps(res, ensure_ascii=False)

    def baidu_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Execute Baidu search for multiple queries and return results

        Args:
            queries (Union[str, List[str]]): Search keyword(s), can be a single string or a list of strings
            max_results (int, optional): Maximum number of results to return for each query, default 5

        Returns:
            str: A JSON formatted string containing the search results.
        """
        logger.debug(f"Searching Baidu for: {queries}, max_results: {max_results}")
        if isinstance(queries, str):
            return self.baidu_search_single_query(queries, max_results=max_results)

        all_results: Dict[str, Any] = {}
        for query in queries:
            result: str = self.baidu_search_single_query(query, max_results=max_results)
            all_results[query] = json.loads(result)
        return json.dumps(all_results, ensure_ascii=False)


if __name__ == '__main__':
    keyword = "NBA"
    results = search(keyword, num_results=5)
    print(results)
    s = BaiduSearchTool()
    r = s.baidu_search(["北京的新闻top3", 'CBA'], max_results=2)
    print(r)
