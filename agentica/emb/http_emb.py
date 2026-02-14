# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Generic HTTP Embedding adapter compatible with OpenAI embedding API schema.
Supports self-hosted services like text2vec-inference, vLLM, TEI, etc.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import requests

from agentica.emb.base import Emb
from agentica.utils.log import logger


@dataclass
class HttpEmb(Emb):
    """Generic HTTP Embedding adapter compatible with OpenAI embedding API schema.

    Works with any service that exposes an OpenAI-compatible /v1/embeddings endpoint,
    such as text2vec-inference, vLLM, TEI (Text Embeddings Inference), etc.

    Args:
        api_url: Full URL of the embeddings endpoint (e.g. "http://localhost:8080/v1/embeddings")
        model: Model name to pass in the request body
        dimensions: Embedding dimensions (set to match your model)
        api_key: Optional API key for authenticated endpoints
        request_headers: Optional extra headers to include in requests
        request_params: Optional extra params to include in the request body
    """
    api_url: str = "http://localhost:8080/v1/embeddings"
    model: str = "default"
    dimensions: Optional[int] = 768
    api_key: Optional[str] = None
    request_headers: Optional[Dict[str, str]] = None
    request_params: Optional[Dict[str, Any]] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.request_headers:
            headers.update(self.request_headers)
        return headers

    def _request(self, texts: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }
        if self.request_params:
            payload.update(self.request_params)

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
            logger.warning(f"Error getting embedding: {e}")
            return []

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        try:
            result = self._request([text])
            embedding = result["data"][0]["embedding"]
            usage = result.get("usage")
            return embedding, usage
        except Exception as e:
            logger.warning(f"Error getting embedding: {e}")
            return [], None

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self._request(texts)
            # Sort by index to ensure correct ordering
            data = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.warning(f"Error getting embeddings: {e}")
            return []
