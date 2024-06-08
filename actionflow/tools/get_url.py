# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os

import requests
from bs4 import BeautifulSoup

from actionflow.config import JINA_API_KEY
from actionflow.output import Output
from actionflow.tool import BaseTool


class GetUrl(BaseTool):
    """
    This class inherits from the BaseFunction class. It defines a function for fetching the contents of a URL.
    """

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",  # "type": "function" indicates that this is a function definition.
            "function": {
                "name": "get_url",
                "description": "Fetch the contents of a URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch content from.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["html", "text", "markdown"],
                            "default": "markdown",
                            "description": (
                                "The format of the returned content. "
                                "If 'html', the full HTML will be returned. If 'text', only the text will be returned."
                                "If 'markdown', the markdown will be returned."
                            ),
                        },
                    },
                    "required": ["url"],
                },
            }
        }

    def execute(self, url: str, format: str = "markdown") -> str:
        """
        Fetches the contents of a URL. The URL and the format of the returned content are provided as parameters.

        :param url: The URL to fetch content from.
        :type url: str
        :param format: The format of the returned content. If 'html', the full HTML will be returned.
            If 'text', only the text will be returned. If 'markdown', the markdown will be returned.
        :type format: str
        :return: The contents of the URL.
        :rtype: str
        """
        if format == "markdown":
            headers = {'X-Return-Format': 'markdown'}
            if JINA_API_KEY and len(JINA_API_KEY) > 0:
                headers['Authorization'] = f'Bearer {JINA_API_KEY}'
            response = requests.get(f'https://r.jina.ai/{url}', headers=headers)
            if response.status_code == 200:
                return response.text
            else:
                raise Exception(f"Failed to fetch URL. HTTP status code: {response.status_code}")
        else:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/58.0.3029.110 Safari/537.3"
                )
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                if format == "html":
                    return response.text
                elif format == "text":
                    soup = BeautifulSoup(response.text, "html.parser")
                    return soup.get_text()
            else:
                raise Exception(f"Failed to fetch URL. HTTP status code: {response.status_code}")


if __name__ == '__main__':
    output = Output('o')
    m = GetUrl(output)
    text = "https://www.jpmorgan.com/insights/global-research/economy/china-economy-cn#section-header#0"
    r = m.execute(text)
    print(text, '\n\n', r)
    os.removedirs('o')
