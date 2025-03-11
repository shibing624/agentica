# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple

from agentica.emb.base import Emb
from agentica.utils.log import logger


@dataclass
class OllamaEmb(Emb):
    model: str = "quentinz/bge-base-zh-v1.5"
    dimensions: int = 384
    host: Optional[str] = "http://localhost:11434"
    timeout: Optional[Any] = None
    options: Optional[Any] = None
    client_kwargs: Optional[Dict[str, Any]] = None
    ollama_client: Optional[Any] = None

    @property
    def client(self) -> Any:
        try:
            from ollama import Client as OllamaClient
        except ImportError:
            raise ImportError("`ollama` not installed, please run `pip install ollama`")

        if self.ollama_client:
            return self.ollama_client

        _ollama_params: Dict[str, Any] = {}
        if self.host:
            _ollama_params["host"] = self.host
        if self.timeout:
            _ollama_params["timeout"] = self.timeout
        if self.client_kwargs:
            _ollama_params.update(self.client_kwargs)
        self.ollama_client = OllamaClient(**_ollama_params)
        return self.ollama_client

    def _response(self, text: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.options is not None:
            kwargs["options"] = self.options

        return self.client.embeddings(prompt=text, model=self.model, **kwargs)  # type: ignore

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self._response(text=text)
            if response is None:
                return []
            return response.get("embedding", [])
        except Exception as e:
            logger.warning(e)
            return []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]
