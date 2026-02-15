# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MulanAI Embedding API adapter.
Uses https://api.mulanai.com/v1/text_embedding
"""
from os import getenv
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import requests

from agentica.embedding.base import Embedding
from agentica.utils.log import logger


@dataclass
class MulanAIEmbedding(Embedding):
    """MulanAI Embedding API adapter.

    Args:
        dimensions: Embedding dimensions
        api_key: MulanAI API key (or set MULANAI_API_KEY env var)
        api_url: API endpoint URL

    import requests

    url = "https://api.mulanai.com/v1/text_embedding"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic your_api_token"
    }
    data = {"text": "这个产品真的太棒了，我非常喜欢！"}

    response = requests.post(url, headers=headers, json=data)
    print(response.json())

    """
    dimensions: Optional[int] = 768
    api_key: Optional[str] = None
    api_url: str = "https://api.mulanai.com/v1/text_embedding"

    def _get_api_key(self) -> str:
        return self.api_key or getenv("MULANAI_API_KEY") or ""

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self._get_api_key()}",
        }

    def _request(self, sentences: List[str]) -> Dict[str, Any]:
        payload = {
            "sentences": sentences,
        }
        response = requests.post(
            self.api_url,
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    def get_embedding(self, text: str) -> List[float]:
        try:
            result = self._request([text])
            # print(result)
            return result["output"]['output_embeddings'][0]
        except Exception as e:
            logger.warning(f"Error getting MulanAI embedding: {e}")
            return []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self._request(texts)
            return result["output"]['output_embeddings']
        except Exception as e:
            logger.warning(f"Error getting MulanAI embeddings: {e}")
            return []


if __name__ == "__main__":
    m = MulanAIEmbedding()
    print(m.get_embedding("hello"))
    print(m.get_embeddings(["hello", "world"]))