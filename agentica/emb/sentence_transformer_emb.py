# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

from agentica.emb.base import Emb
import numpy as np
import platform

try:
    from sentence_transformers import SentenceTransformer

    if platform.system() == "Windows":
        numpy_version = np.__version__
        if numpy_version.startswith("2"):
            raise RuntimeError(
                "Incompatible NumPy version detected. Please install NumPy 1.x by running 'pip install numpy<2'."
            )
except ImportError:
    raise ImportError("sentence-transformers not installed, please run `pip install sentence-transformers`")


@dataclass
class SentenceTransformerEmb(Emb):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: Optional[int] = 384
    client: SentenceTransformer = None
    client_params: Optional[Dict[str, Any]] = None

    @property
    def get_client(self) -> SentenceTransformer:
        if self.client:
            return self.client

        _client_params: Dict[str, Any] = {}
        if self.model:
            _client_params["model_name_or_path"] = self.model
        if self.client_params:
            _client_params.update(self.client_params)

        self.client = SentenceTransformer(**_client_params)
        dim = self.client.get_sentence_embedding_dimension()
        if dim:
            self.dimensions = dim
        return self.client

    def get_embedding(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        # Calculate emb of the text
        return self.get_client.encode([text], normalize_embeddings=normalize_embeddings).tolist()[0]

    def get_embeddings(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        # Calculate emb of the texts
        return self.get_client.encode(texts, normalize_embeddings=normalize_embeddings).tolist()
