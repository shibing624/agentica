"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from typing import Optional, Dict, List, Tuple, Any, Union

from agentica.emb.base import Emb
from agentica.utils.log import logger

try:
    import google.generativeai as genai
    from google.generativeai.types.text_types import EmbeddingDict, BatchEmbeddingDict
except ImportError:
    raise ImportError("`google-generativeai` not installed. Please install it using `pip install google-generativeai`")


class GeminiEmb(Emb):
    model: str = "models/embedding-001"
    task_type: str = "RETRIEVAL_QUERY"
    title: Optional[str] = None
    dimensions: Optional[int] = None
    api_key: Optional[str] = None
    request_params: Optional[Dict[str, Any]] = None
    client_params: Optional[Dict[str, Any]] = None
    gemini_client: Optional[genai.embed_content] = None

    @property
    def client(self):
        if self.gemini_client:
            return self.gemini_client
        _client_params: Dict[str, Any] = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.client_params:
            _client_params.update(self.client_params)
        self.gemini_client = genai
        self.gemini_client.configure(**_client_params)
        return self.gemini_client

    def _response(self, text: str) -> Union[EmbeddingDict, BatchEmbeddingDict]:
        _request_params: Dict[str, Any] = {
            "content": text,
            "model": self.model,
            "output_dimensionality": self.dimensions,
            "task_type": self.task_type,
            "title": self.title,
        }
        if self.request_params:
            _request_params.update(self.request_params)
        return self.client.embed_content(**_request_params)

    def get_embedding(self, text: str) -> List[float]:
        response = self._response(text=text)
        try:
            return response.get("embedding", [])
        except Exception as e:
            logger.warning(e)
            return []

