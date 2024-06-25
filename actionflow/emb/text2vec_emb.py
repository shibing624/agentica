# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import List, Optional, Dict, Any

try:
    from text2vec import SentenceModel
except ImportError:
    raise ImportError(
        "`text2vec` not installed. Please install it with `pip install text2vec`"
    )
from actionflow.emb.base import Emb


class Text2VecEmb(Emb):
    """Text2Vec embedding model(SBert), using the `text2vec` library"""
    model: str = "shibing624/text2vec-base-multilingual"
    dimensions: int = 384
    client: SentenceModel = None
    client_params: Optional[Dict[str, Any]] = None

    @property
    def get_client(self) -> SentenceModel:
        if self.client:
            return self.client

        _client_params: Dict[str, Any] = {}
        if self.model:
            _client_params["model_name_or_path"] = self.model
        if self.client_params:
            _client_params.update(self.client_params)

        self.client = SentenceModel(**_client_params)
        return self.client

    def get_embedding(self, text: str) -> List[float]:
        # Calculate emb of the text
        return self.get_client.encode([text])[0]
