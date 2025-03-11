# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from dataclasses import dataclass
from os import getenv
from typing import Optional

from agentica.emb.openai_emb import OpenAIEmb


@dataclass
class TogetherEmb(OpenAIEmb):
    model: str = "togethercomputer/m2-bert-80M-32k-retrieval"
    dimensions: int = 768
    api_key: Optional[str] = getenv("TOGETHER_API_KEY")
    base_url: str = "https://api.together.xyz/v1"
