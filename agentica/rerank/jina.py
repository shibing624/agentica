# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Jina Reranker API adapter.
Uses https://api.jina.ai/v1/rerank
"""
from os import getenv
from typing import List, Optional

import requests

from agentica.rerank.base import Rerank
from agentica.document import Document
from agentica.utils.log import logger


class JinaRerank(Rerank):
    """Jina Reranker API adapter.

    Args:
        model: Jina rerank model name
        api_key: Jina API key (or set JINA_API_KEY env var)
        api_url: API endpoint URL
        top_n: Maximum number of documents to return
    """
    model: str = "jina-reranker-v2-base-multilingual"
    api_key: Optional[str] = None
    api_url: str = "https://api.jina.ai/v1/rerank"
    top_n: Optional[int] = None

    def _get_api_key(self) -> str:
        return self.api_key or getenv("JINA_API_KEY") or ""

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        top_n = self.top_n
        if top_n and not (0 < top_n):
            logger.warning(f"top_n should be a positive integer, got {self.top_n}, setting top_n to None")
            top_n = None

        _docs = [doc.content for doc in documents]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_api_key()}",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": _docs,
        }
        if top_n:
            payload["top_n"] = top_n

        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        compressed_docs: list[Document] = []
        for item in result.get("results", []):
            idx = item["index"]
            doc = documents[idx]
            doc.reranking_score = item["relevance_score"]
            compressed_docs.append(doc)

        compressed_docs.sort(
            key=lambda x: x.reranking_score if x.reranking_score is not None else float("-inf"),
            reverse=True,
        )

        if top_n:
            compressed_docs = compressed_docs[:top_n]

        return compressed_docs

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            return self._rerank(query=query, documents=documents)
        except Exception as e:
            logger.error(f"Error reranking documents with Jina: {e}. Returning original documents")
            return documents
