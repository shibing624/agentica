# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from os import getenv
from typing import Optional

from agentica.llm.openai_llm import OpenAILLM


class YiLLM(OpenAILLM):
    name: str = "YiLLM"
    model: str = "yi-lightning"
    api_key: Optional[str] = getenv("YI_API_KEY")
    base_url: str = "https://api.lingyiwanwu.com/v1"
