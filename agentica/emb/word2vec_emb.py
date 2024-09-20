# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import List, Optional, Dict, Any

try:
    from text2vec import Word2Vec
except ImportError:
    raise ImportError(
        "`text2vec` not installed. Please install it with `pip install text2vec`"
    )
from agentica.emb.base import Emb


class Word2VecEmb(Emb):
    """Word2Vec embedding model, using the `text2vec` library"""
    model: str = "w2v-light-tencent-chinese"
    dimensions: int = 256
    client: Word2Vec = None
    client_params: Optional[Dict[str, Any]] = None

    @property
    def get_client(self) -> Word2Vec:
        if self.client:
            return self.client

        _client_params: Dict[str, Any] = {}
        if self.model:
            _client_params["model_name_or_path"] = self.model
        if self.client_params:
            _client_params.update(self.client_params)
        self.client = Word2Vec(**_client_params)
        return self.client

    def get_embedding(self, text: str) -> List[float]:
        # Calculate emb of the text
        return self.get_client.encode([text]).tolist()[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Calculate emb of the texts
        ndarr = self.get_client.encode(texts)
        # convert numpy array to list
        return ndarr.tolist()  # type: ignore
