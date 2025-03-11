# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from os import getenv
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Literal

from openai import OpenAI as OpenAIClient
from openai.types.create_embedding_response import CreateEmbeddingResponse

from agentica.emb.base import Emb
from agentica.utils.log import logger


@dataclass
class OpenAIEmb(Emb):
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    encoding_format: Literal["float", "base64"] = "float"
    user: Optional[str] = None
    api_key: Optional[str] = getenv("OPENAI_API_KEY")
    organization: Optional[str] = None
    base_url: Optional[str] = getenv("OPENAI_BASE_URL")
    request_params: Optional[Dict[str, Any]] = None
    client_params: Optional[Dict[str, Any]] = None
    openai_client: Optional[OpenAIClient] = None


    @property
    def client(self) -> OpenAIClient:
        if self.openai_client:
            return self.openai_client

        _client_params: Dict[str, Any] = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.organization:
            _client_params["organization"] = self.organization
        if self.base_url:
            _client_params["base_url"] = self.base_url
        if self.client_params:
            _client_params.update(self.client_params)
        self.openai_client = OpenAIClient(**_client_params)
        return self.openai_client

    def _response(self, text: str) -> CreateEmbeddingResponse:
        _request_params: Dict[str, Any] = {
            "input": text,
            "model": self.model,
            "encoding_format": self.encoding_format,
        }
        if self.user is not None:
            _request_params["user"] = self.user
        if self.model.startswith("text-embedding-3"):
            _request_params["dimensions"] = self.dimensions
        if self.request_params:
            _request_params.update(self.request_params)
        return self.client.embeddings.create(**_request_params)

    def get_embedding(self, text: str) -> List[float]:
        response: CreateEmbeddingResponse = self._response(text=text)
        try:
            return response.data[0].embedding
        except Exception as e:
            logger.warning(e)
            return []

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        response: CreateEmbeddingResponse = self._response(text=text)

        embedding = response.data[0].embedding
        usage = response.usage
        if usage:
            return embedding, usage.model_dump()
        return embedding, None

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]


if __name__ == '__main__':
    emb = OpenAIEmb()
    text = "I love you"
    r = emb.get_embedding(text)
    print(r)
    r = emb.get_embedding_and_usage(text)
    print(r)
    texts = ["I love you", "我喜欢你"]
    r = emb.get_embeddings(texts)
    print(r)
