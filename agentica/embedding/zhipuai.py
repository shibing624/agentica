"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from os import getenv
from dataclasses import dataclass
from typing import Optional

from agentica.embedding.openai import OpenAIEmbedding


@dataclass
class ZhipuAIEmbedding(OpenAIEmbedding):
    model: str = "embedding-3"
    dimensions: int = 2048
    api_key: Optional[str] = getenv("ZHIPUAI_API_KEY")
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
