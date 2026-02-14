# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Jina Embedding API adapter.
Uses https://api.jina.ai/v1/embeddings
"""
from os import getenv
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import requests

from agentica.emb.base import Emb
from agentica.utils.log import logger


@dataclass
class JinaEmb(Emb):
    """Jina Embedding API adapter.

    Args:
        model: Jina embedding model name (default: jina-embeddings-v3)
        dimensions: Embedding dimensions
        api_key: Jina API key (or set JINA_API_KEY env var)
        api_url: API endpoint URL
        task: Task type for the embedding (e.g. "retrieval.query", "retrieval.passage", "text-matching")
    """
    model: str = "jina-embeddings-v3"
    dimensions: Optional[int] = 1024
    api_key: Optional[str] = None
    api_url: str = "https://api.jina.ai/v1/embeddings"
    task: Optional[str] = None

    def _get_api_key(self) -> str:
        return self.api_key or getenv("JINA_API_KEY") or ""

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_api_key()}",
        }

    def _request(self, texts: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        if self.task:
            payload["task"] = self.task

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
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Error getting Jina embedding: {e}")
            return []

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        try:
            result = self._request([text])
            embedding = result["data"][0]["embedding"]
            usage = result.get("usage")
            return embedding, usage
        except Exception as e:
            logger.warning(f"Error getting Jina embedding: {e}")
            return [], None

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self._request(texts)
            data = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.warning(f"Error getting Jina embeddings: {e}")
            return []
