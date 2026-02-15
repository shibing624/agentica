"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from dataclasses import dataclass
from os import getenv
from typing import Optional

from agentica.embedding.openai import OpenAIEmbedding


@dataclass
class FireworksEmbedding(OpenAIEmbedding):
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = 768
    api_key: Optional[str] = getenv("FIREWORKS_API_KEY")
    base_url: str = "https://api.fireworks.ai/inference/v1"
