# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from os import getenv
from typing import Optional

from agentica.llm.openai_llm import OpenAILLM


class OpenRouterLLM(OpenAILLM):
    name: str = "OpenRouter"
    model: str = "mistralai/mistral-7b-instruct:free"
    api_key: Optional[str] = getenv("OPENROUTER_API_KEY")
    base_url: str = "https://openrouter.ai/api/v1"
