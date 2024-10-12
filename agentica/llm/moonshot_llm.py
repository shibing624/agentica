# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from os import getenv
from typing import Optional

from agentica.llm.openai_llm import OpenAILLM


class MoonshotLLM(OpenAILLM):
    name: str = "MoonshotLLM"
    model: str = "moonshot-v1-auto"
    api_key: Optional[str] = getenv("MOONSHOT_API_KEY")
    base_url: str = "https://api.moonshot.cn/v1"
