# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from os import getenv
from typing import Optional, Dict, Any

from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient

from agentica.llm.openai_llm import OpenAILLM


class MoonshotLLM(OpenAILLM):
    name: str = "MoonshotLLM"
    model: str = "moonshot-v1-auto"
    api_key: Optional[str] = getenv("MOONSHOT_API_KEY")
    base_url: str = "https://api.moonshot.cn/v1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    request_params: Optional[Dict[str, Any]] = None
    client_params: Optional[Dict[str, Any]] = None
    # -*- Provide the client manually
    client: Optional[OpenAIClient] = None
    async_client: Optional[AsyncOpenAIClient] = None
