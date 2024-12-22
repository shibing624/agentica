# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, ConfigDict

from agentica.emb.base import Emb


class Document(BaseModel):
    """Model for managing a document"""

    content: str
    id: Optional[str] = None
    name: Optional[str] = None
    meta_data: Dict[str, Any] = {}
    embedder: Optional[Emb] = None
    embedding: Optional[List[float]] = None
    usage: Optional[Dict[str, Any]] = None
    reranking_score: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def embed(self, embedder: Optional[Emb] = None) -> None:
        """Embed the document using the provided embedder"""

        _embedder = embedder or self.embedder
        if _embedder is None:
            raise ValueError("No embedder provided")
        if hasattr(_embedder, "get_embedding_and_usage"):
            self.embedding, self.usage = _embedder.get_embedding_and_usage(self.content)
        else:
            self.embedding = _embedder.get_embedding(self.content)
            self.usage = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the document"""

        return self.model_dump(include={"name", "meta_data", "content"}, exclude_none=True)

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "Document":
        """Returns a Document object from a dictionary representation"""

        return cls.model_validate(**document)

    @classmethod
    def from_json(cls, document: str) -> "Document":
        """Returns a Document object from a json string representation"""

        return cls.model_validate_json(document)