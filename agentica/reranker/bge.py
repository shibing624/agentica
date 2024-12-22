# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import List, Dict, Any, Optional
from agentica.reranker.base import Reranker
from agentica.document import Document
from agentica.utils.log import logger

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    raise ImportError("transformers not installed, please run `pip install transformers`")


class BgeReranker(Reranker):
    model: str = "BAAI/bge-reranker-base"
    _client = None
    top_n: Optional[int] = None

    @property
    def client(self):
        if self._client is None:
            self._client = {
                "tokenizer": AutoTokenizer.from_pretrained(self.model),
                "model": AutoModelForSequenceClassification.from_pretrained(self.model).to(self.device)
            }
        return self._client

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # Validate input documents and top_n
        if not documents:
            return []

        top_n = self.top_n
        if top_n and not (0 < top_n):
            logger.warning(f"top_n should be a positive integer, got {self.top_n}, setting top_n to None")
            top_n = None

        compressed_docs: List[Document] = []
        _docs = [doc.content for doc in documents]
        pairs = [[query, doc] for doc in _docs]

        with torch.no_grad():
            inputs = self.client["tokenizer"](pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            scores = self.client["model"](**inputs_on_device, return_dict=True).logits.view(-1, ).float()

        for idx, score in enumerate(scores):
            doc = documents[idx]
            doc.reranking_score = score.item()
            compressed_docs.append(doc)

        # Order by relevance score
        compressed_docs.sort(
            key=lambda x: x.reranking_score if x.reranking_score is not None else float("-inf"),
            reverse=True,
        )

        # Limit to top_n if specified
        if top_n:
            compressed_docs = compressed_docs[:top_n]

        return compressed_docs

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            return self._rerank(query=query, documents=documents)
        except Exception as e:
            logger.error(f"Error reranking documents: {e}. Returning original documents")
            return documents


if __name__ == '__main__':
    reranker = BgeReranker()
    query = "大语言模型微调有啥好处?"
    documents = [
        Document(content="Finetuning LLM can improve performance on specific tasks."),
        Document(content="It allows the model to adapt to domain-specific language."),
        Document(content="Finetuning can reduce the need for large amounts of labeled data."),
        Document(content="It can enhance the model's ability to generalize from limited data."),
        Document(content="大语言模型微调后可以完成指定的任务"),
    ]

    # Perform reranking
    reranked_documents = reranker.rerank(query=query, documents=documents)

    # Print the reranked documents
    print(f"Query: {query}\n\n")
    for doc in reranked_documents:
        print(f"Content: {doc.content}, Score: {doc.reranking_score}")
