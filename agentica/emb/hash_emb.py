# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from dataclasses import dataclass
import hashlib
from typing import List

from agentica.emb.base import Emb


@dataclass
class HashEmb(Emb):
    """A Literal Hash Embedding Function, which hashes the input text as a list of floats."""
    model: str = "literal_hash_emb"
    dimensions: int = 32

    def get_embedding(self, text: str) -> List[float]:
        # Calculate the SHA-256 hash of the text
        hash_object = hashlib.sha256(text.encode())
        hash_list = list(hash_object.digest())
        float_list = [float(x) / 255.0 for x in hash_list]
        return float_list

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]


if __name__ == '__main__':
    emb = HashEmb()
    text = "I love you"
    r = emb.get_embedding(text)
    print(r)
    texts = ["I love you", "我喜欢你"]
    r = emb.get_embeddings(texts)
    print(r)
