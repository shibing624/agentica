# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import List, Tuple, Optional, Dict, Any

try:
    from text2vec import Word2Vec
except ImportError:
    raise ImportError(
        "`text2vec` not installed. Please install it with `pip install text2vec`"
    )
from actionflow.emb.base import Emb


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

        return Word2Vec(**_client_params)

    def get_embedding(self, text: str) -> List[float]:
        # Calculate emb of the text
        return self.get_client.encode([text])[0]

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        usage = None
        embedding = self.get_embedding(text)
        return embedding, usage
