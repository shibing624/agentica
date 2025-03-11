# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

try:
    from text2vec import SentenceModel
except ImportError:
    raise ImportError(
        "`text2vec` not installed. Please install it with `pip install text2vec`"
    )
from agentica.emb.base import Emb


@dataclass
class Text2VecEmb(Emb):
    """Text2Vec embedding model(SBert), using the `text2vec` library"""
    model: str = "shibing624/text2vec-base-multilingual"
    dimensions: Optional[int] = 384
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
        dim = self.client.get_sentence_embedding_dimension()
        if dim:
            self.dimensions = dim
        return self.client

    def get_embedding(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        # Calculate emb of the text
        return self.get_client.encode([text], normalize_embeddings=normalize_embeddings)[0]

    def get_embeddings(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        # Calculate emb of the texts
        return self.get_client.encode(texts, normalize_embeddings=normalize_embeddings).tolist()


if __name__ == '__main__':
    emb = Text2VecEmb()
    text = "I love you"
    r = emb.get_embedding(text)
    print(r)
    texts = ["I love you", "我喜欢你"]
    r = emb.get_embeddings(texts)
    print(r)
