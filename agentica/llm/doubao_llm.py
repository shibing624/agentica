# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from os import getenv
from typing import Optional

from agentica.llm.openai_llm import OpenAILLM


class DoubaoLLM(OpenAILLM):
    name: str = "DoubaoLLM"
    model: str = "ep-20241012172611-btlgr"
    api_key: Optional[str] = getenv("ARK_API_KEY")
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
