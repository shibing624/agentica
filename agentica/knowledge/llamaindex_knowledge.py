# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""
from typing import List, Optional, Callable, Any

from agentica.document import Document
from agentica.knowledge.base import Knowledge
from agentica.utils.log import logger


class LlamaIndexKnowledge(Knowledge):
    retriever: Any
    loader: Optional[Callable] = None

    def search(self, query: str, num_documents: Optional[int] = None, filters=None) -> List[Document]:
        """Returns relevant documents matching the query."""
        try:
            from llama_index.core.schema import NodeWithScore
            from llama_index.core.retrievers import BaseRetriever
        except ImportError:
            raise ImportError(
                "The `llama-index-core` package is not installed. Please install it via `pip install llama-index-core`."
            )
        if not isinstance(self.retriever, BaseRetriever):
            raise ValueError(f"Retriever is not of type BaseRetriever: {self.retriever}")

        lc_documents: List[NodeWithScore] = self.retriever.retrieve(query)
        if num_documents is not None:
            lc_documents = lc_documents[:num_documents]
        documents = []
        for lc_doc in lc_documents:
            documents.append(
                Document(
                    content=lc_doc.text,
                    meta_data=lc_doc.metadata,
                )
            )
        return documents

    def load(self, recreate: bool = False, upsert: bool = True, skip_existing: bool = True, filters=None) -> None:
        if self.loader is None:
            logger.error("No loader provided for LlamaIndexKnowledgeBase")
            return
        self.loader()

    def exists(self) -> bool:
        logger.warning("LlamaIndexKnowledgeBase.exists() not supported - please check the vectorstore manually.")
        return True
