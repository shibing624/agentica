# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""

import json
from hashlib import md5
from typing import List, Optional, Dict, Any

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    raise ImportError("`lancedb` not installed, please install it via `pip install lancedb`.")

from agentica.document import Document
from agentica.emb.base import Emb
from agentica.vectordb.base import VectorDb, Distance, SearchType
from agentica.emb.openai_emb import OpenAIEmb
from agentica.reranker.base import Reranker
from agentica.utils.log import logger


class LanceDb(VectorDb):
    def __init__(
            self,
            embedder: Emb = OpenAIEmb(),
            distance: Distance = Distance.cosine,
            connection: Optional[lancedb.db.LanceTable] = None,
            uri: Optional[str] = "tmp/lancedb",
            table: Optional[lancedb.db.LanceTable] = None,
            table_name: Optional[str] = "agentica",
            nprobes: Optional[int] = None,
            search_type: SearchType = SearchType.vector,
            reranker: Optional[Reranker] = None,
            use_tantivy: bool = True,
    ):
        # Embedder for embedding the document contents
        self.embedder: Emb = embedder
        self.dimensions: int = self.embedder.dimensions

        # Distance metric
        self.distance: Distance = distance

        # Connection to lancedb table, can also be provided to use an existing connection
        self.uri = uri
        self.connection: lancedb.DBConnection = connection or lancedb.connect(uri=self.uri)

        # LanceDB table details
        self.table: lancedb.db.LanceTable
        self.table_name: str
        if table:
            if not isinstance(table, lancedb.db.LanceTable):
                raise ValueError(
                    "table should be an instance of lancedb.db.LanceTable, ",
                    f"got {type(table)}",
                )
            self.table = table
            self.table_name = self.table.name
            self._vector_col = self.table.schema.names[0]
            self._id = self.tbl.schema.names[1]  # type: ignore
        else:
            if not table_name:
                raise ValueError("Either table or table_name should be provided.")
            self.table_name = table_name
            self._id = "id"
            self._vector_col = "vector"
            self.table = self._init_table()

        self.reranker: Optional[Reranker] = reranker
        self.nprobes: Optional[int] = nprobes
        self.search_type = search_type
        self.fts_index_exists = False
        self.use_tantivy = use_tantivy

        if self.use_tantivy:
            try:
                import tantivy  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Please install tantivy-py `pip install tantivy` to use the full text search feature."
                )

        logger.debug(f"Initialized LanceDb with table: '{self.table_name}'")

    def create(self) -> None:
        """Create the table if it does not exist."""
        if not self.exists():
            self.connection = self._init_table()  # Connection update is needed

    def _init_table(self) -> lancedb.db.LanceTable:
        schema = pa.schema(
            [
                pa.field(
                    self._vector_col,
                    pa.list_(
                        pa.float32(),
                        len(self.embedder.get_embedding("test")),  # type: ignore
                    ),
                ),
                pa.field(self._id, pa.string()),
                pa.field("payload", pa.string()),
            ]
        )

        logger.info(f"Creating table: {self.table_name}")
        tbl = self.connection.create_table(self.table_name, schema=schema, mode="overwrite")
        return tbl  # type: ignore

    def doc_exists(self, document: Document) -> bool:
        """
        Validating if the document exists or not

        Args:
            document (Document): Document to validate
        """
        if self.table:
            cleaned_content = document.content.replace("\x00", "\ufffd")
            doc_id = md5(cleaned_content.encode()).hexdigest()
            result = self.table.search().where(f"{self._id}='{doc_id}'").to_arrow()
            return len(result) > 0
        return False

    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert documents into the database.

        Args:
            documents (List[Document]): List of documents to insert
            filters (Optional[Dict[str, Any]]): Filters to apply while inserting documents
        """
        logger.debug(f"Inserting {len(documents)} documents")
        data = []
        for document in documents:
            document.embed(embedder=self.embedder)
            cleaned_content = document.content.replace("\x00", "\ufffd")
            doc_id = str(md5(cleaned_content.encode()).hexdigest())
            payload = {
                "name": document.name,
                "meta_data": document.meta_data,
                "content": cleaned_content,
                "usage": document.usage,
            }
            data.append(
                {
                    "id": doc_id,
                    "vector": document.embedding,
                    "payload": json.dumps(payload, ensure_ascii=False),
                }
            )
            logger.debug(f"Inserted document: {document.name} ({document.to_dict()})")

        self.table.add(data)
        logger.debug(f"Upsert {len(data)} documents")

    def upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Upsert documents into the database.

        Args:
            documents (List[Document]): List of documents to upsert
            filters (Optional[Dict[str, Any]]): Filters to apply while upserting
        """
        logger.debug("Redirecting the request to insert")
        self.insert(documents)

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if self.search_type == SearchType.vector:
            return self.vector_search(query, limit)
        elif self.search_type == SearchType.keyword:
            return self.keyword_search(query, limit)
        elif self.search_type == SearchType.hybrid:
            return self.hybrid_search(query, limit)
        else:
            logger.error(f"Invalid search type '{self.search_type}'.")
            return []

    def vector_search(self, query: str, limit: int = 5) -> List[Document]:
        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        results = self.table.search(
            query=query_embedding,
            vector_column_name=self._vector_col,
        ).limit(limit)

        if self.nprobes:
            results.nprobes(self.nprobes)

        results = results.to_pandas()
        search_results = self._build_search_results(results)

        if self.reranker:
            search_results = self.reranker.rerank(query=query, documents=search_results)

        return search_results

    def hybrid_search(self, query: str, limit: int = 5) -> List[Document]:
        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []
        if not self.fts_index_exists:
            self.table.create_fts_index("payload", replace=True)
            self.fts_index_exists = True

        results = (
            self.table.search(
                vector_column_name=self._vector_col,
                query_type="hybrid",
            )
            .vector(query_embedding)
            .text(query)
            .limit(limit)
        )

        if self.nprobes:
            results.nprobes(self.nprobes)

        results = results.to_pandas()

        search_results = self._build_search_results(results)

        if self.reranker:
            search_results = self.reranker.rerank(query=query, documents=search_results)

        return search_results

    def keyword_search(self, query: str, limit: int = 5) -> List[Document]:
        if not self.fts_index_exists:
            self.table.create_fts_index("payload", replace=True)
            self.fts_index_exists = True

        results = (
            self.table.search(
                query=query,
                query_type="fts",
            )
            .limit(limit)
            .to_pandas()
        )
        search_results = self._build_search_results(results)

        if self.reranker:
            search_results = self.reranker.rerank(query=query, documents=search_results)
        return search_results

    def _build_search_results(self, results) -> List[Document]:  # TODO: typehint pandas?
        search_results: List[Document] = []
        try:
            for _, item in results.iterrows():
                payload = json.loads(item["payload"])
                search_results.append(
                    Document(
                        name=payload["name"],
                        meta_data=payload["meta_data"],
                        content=payload["content"],
                        embedder=self.embedder,
                        embedding=item["vector"],
                        usage=payload["usage"],
                    )
                )

        except Exception as e:
            logger.error(f"Error building search results: {e}")

        return search_results

    def drop(self) -> None:
        if self.exists():
            logger.debug(f"Deleting collection: {self.table_name}")
            self.connection.drop_table(self.table_name)

    def exists(self) -> bool:
        if self.connection:
            if self.table_name in self.connection.table_names():
                return True
        return False

    def get_count(self) -> int:
        if self.exists():
            return self.table.count_rows()
        return 0

    def delete(self) -> bool:
        return False

    def name_exists(self, name: str) -> bool:
        raise NotImplementedError
