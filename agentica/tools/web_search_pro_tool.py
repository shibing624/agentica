# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use ZhipuAI web-search-pro tool, free to use
"""

import json
import requests
import uuid
from os import getenv
from typing import Optional, Dict, Any, List

try:
    from zhipuai import ZhipuAI
except ImportError:
    raise ImportError("`zhipuai` not installed. Please install using `pip install zhipuai`.")

from agentica.tools.base import Tool
from agentica.utils.log import logger


class WebSearchProTool(Tool):
    def __init__(self, api_key: Optional[str] = None, timeout: int = 300):
        super().__init__(name="web_search_pro")
        self.api_key = api_key or getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            logger.error("ZHIPUAI_API_KEY not set. Please set the ZHIPUAI_API_KEY environment variable.")
        self.timeout = timeout
        self.register(self.search_web)

    def search_web(self, query: str) -> str:
        """Use this function to search Exa (a web search engine) for a query.

        Args:
            query (str): The query to search for.

        Example:
            from agentica.tools.web_search_pro_tool import WebSearchProTool
            m = WebSearchProTool()
            query = "苹果的最新产品是啥？"
            r = m.search_web(query)
            print(r)

        Returns:
            str: The search results in JSON format.
        """
        if not self.api_key:
            return "Please set the ZHIPUAI_API_KEY"

        try:
            logger.info(f"Searching web for: {query}")
            msg = [{"role": "user", "content": query}]
            tool = "web-search-pro"
            url = "https://open.bigmodel.cn/api/paas/v4/tools"
            data = {
                "request_id": str(uuid.uuid4()),
                "tool": tool,
                "stream": False,
                "messages": msg
            }
            resp = requests.post(
                url,
                json=data,
                headers={'Authorization': self.api_key},
                timeout=self.timeout
            )
            data = json.loads(resp.content.decode())
            # parse data
            results = data['choices'][0]['message']['tool_calls'][1]['search_result']
            parsed_results = json.dumps(results, indent=2, ensure_ascii=False)
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to search exa {e}")
            return f"Error: {e}"


if __name__ == '__main__':
    m = WebSearchProTool()
    query = "苹果的最新产品是啥？"
    r = m.search_web(query)
    print(query, '\n\n', r)
