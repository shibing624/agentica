"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from typing import Optional, Dict, List, Tuple, Any, Union

from agentica.emb.base import Emb
from agentica.utils.log import logger

try:
    from huggingface_hub import InferenceClient, SentenceSimilarityInput
except ImportError:
    raise ImportError("`huggingface-hub` not installed, please run `pip install huggingface-hub`")


class HuggingfaceEmb(Emb):
    """Huggingface Custom Embedder"""

    model: str = "jinaai/jina-embeddings-v2-base-code"
    api_key: Optional[str] = None
    client_params: Optional[Dict[str, Any]] = None
    huggingface_client: Optional[InferenceClient] = None

    @property
    def client(self) -> InferenceClient:
        if self.huggingface_client:
            return self.huggingface_client
        _client_params: Dict[str, Any] = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.client_params:
            _client_params.update(self.client_params)
        self.huggingface_client = InferenceClient(**_client_params)
        return self.huggingface_client

    def _response(self, text: str):
        _request_params: SentenceSimilarityInput = {
            "json": {"inputs": text},
            "model": self.model,
        }
        return self.client.post(**_request_params)

    def get_embedding(self, text: str) -> List[float]:
        resp = []
        try:
            resp = self._response(text=text)
        except Exception as e:
            logger.warning(e)
        return resp

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        return super().get_embedding_and_usage(text)
