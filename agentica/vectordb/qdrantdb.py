import os
from hashlib import md5
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from agentica.document import Document
from agentica.emb.base import Emb
from agentica.emb.openai_emb import OpenAIEmb
from agentica.vectordb.base import VectorDb, Distance
from agentica.utils.log import logger
from agentica.reranker.base import Reranker


class QdrantDb(VectorDb):
    """
    Qdrant-based Vector Database with semantic search support.

    Features:
    - Vector-based semantic search using embeddings
    - Local disk storage (on_disk=True by default)
    - Support for remote Qdrant server
    """

    def __init__(
            self,
            collection: str = "qdrant_vec_db",
            embedder: Emb = None,
            distance: Distance = Distance.cosine,
            path: Optional[str] = None,
            on_disk: bool = True,
            location: Optional[str] = None,
            url: Optional[str] = None,
            port: Optional[int] = 6333,
            grpc_port: int = 6334,
            prefer_grpc: bool = False,
            https: Optional[bool] = None,
            api_key: Optional[str] = None,
            prefix: Optional[str] = None,
            timeout: Optional[float] = None,
            host: Optional[str] = None,
            reranker: Optional[Reranker] = None,
            **kwargs,
    ):
        """
        Initialize QdrantDb.

        Args:
            collection: Name of the Qdrant collection
            embedder: Embedding model for semantic search (default: OpenAIEmb)
            distance: Distance metric for vector similarity
            path: Path for local disk storage (used when on_disk=True)
            on_disk: Whether to use local disk storage (default: True)
            location: Qdrant server location (":memory:" for in-memory)
            url: Qdrant server URL (for remote server)
            port: Qdrant server port
            grpc_port: Qdrant gRPC port
            prefer_grpc: Whether to prefer gRPC over HTTP
            https: Whether to use HTTPS
            api_key: API key for Qdrant Cloud
            prefix: URL prefix for Qdrant server
            timeout: Request timeout
            host: Qdrant server host
            reranker: Reranker for search results
            **kwargs: Additional arguments passed to QdrantClient
        """
        # Collection attributes
        self.collection: str = collection

        # Embedder for embedding the document contents (default to OpenAIEmb)
        self.embedder: Emb = embedder if embedder is not None else OpenAIEmb()
        self.dimensions: Optional[int] = self.embedder.dimensions

        # Distance metric
        self.distance: Distance = distance

        # Storage mode
        self.on_disk: bool = on_disk

        # Setup storage path for local disk storage
        if on_disk and path is None and url is None and location is None:
            path = os.path.join(os.path.expanduser("~"), ".agentica", "qdrant_db")
        if path:
            path = os.path.expanduser(path)
        self.path: Optional[str] = path

        # Create storage directory if needed
        if self.path:
            os.makedirs(self.path, exist_ok=True)

        # Qdrant client arguments
        self.location: Optional[str] = location
        self.url: Optional[str] = url
        self.port: Optional[int] = port
        self.grpc_port: int = grpc_port
        self.prefer_grpc: bool = prefer_grpc
        self.https: Optional[bool] = https
        self.api_key: Optional[str] = api_key
        self.prefix: Optional[str] = prefix
        self.timeout: Optional[float] = timeout
        self.host: Optional[str] = host
        self.reranker: Optional[Reranker] = reranker
        self.kwargs = kwargs

        # Qdrant client instance (lazy initialization)
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.debug("Creating Qdrant Client")
            if self.url:
                # Remote Qdrant server
                self._client = QdrantClient(
                    url=self.url,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    https=self.https,
                    api_key=self.api_key,
                    prefix=self.prefix,
                    timeout=self.timeout,
                    host=self.host,
                    **self.kwargs,
                )
            elif self.on_disk and self.path:
                # Local disk storage
                self._client = QdrantClient(path=self.path, **self.kwargs)
            elif self.location:
                # Custom location (e.g., ":memory:")
                self._client = QdrantClient(location=self.location, **self.kwargs)
            else:
                # In-memory storage as fallback
                self._client = QdrantClient(location=":memory:", **self.kwargs)
        return self._client

    def create(self) -> None:
        # Collection distance
        _distance = models.Distance.COSINE
        if self.distance == Distance.l2:
            _distance = models.Distance.EUCLID
        elif self.distance == Distance.max_inner_product:
            _distance = models.Distance.DOT

        if not self.exists():
            logger.debug(f"Creating collection: {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dimensions, distance=_distance),
            )

    def doc_exists(self, document: Document) -> bool:
        """
        Validating if the document exists or not

        Args:
            document (Document): Document to validate
        """
        if self.client:
            cleaned_content = document.content.replace("\x00", "\ufffd")
            doc_id = md5(cleaned_content.encode()).hexdigest()
            collection_points = self.client.retrieve(
                collection_name=self.collection,
                ids=[doc_id],
            )
            return len(collection_points) > 0
        return False

    def name_exists(self, name: str) -> bool:
        """
        Validates if a document with the given name exists in the collection.

        Args:
            name (str): The name of the document to check.

        Returns:
            bool: True if a document with the given name exists, False otherwise.
        """
        if self.client:
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="name", match=models.MatchValue(value=name))]
                ),
                limit=1,
            )
            return len(scroll_result[0]) > 0
        return False

    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None, batch_size: int = 10) -> None:
        """
        Insert documents into the database.

        Args:
            documents (List[Document]): List of documents to insert
            filters (Optional[Dict[str, Any]]): Filters to apply while inserting documents
            batch_size (int): Batch size for inserting documents
        """
        logger.debug(f"Inserting {len(documents)} documents")
        points = []
        for document in documents:
            document.embed(embedder=self.embedder)
            cleaned_content = document.content.replace("\x00", "\ufffd")
            doc_id = md5(cleaned_content.encode()).hexdigest()
            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=document.embedding,
                    payload={
                        "name": document.name,
                        "meta_data": document.meta_data,
                        "content": cleaned_content,
                        "usage": document.usage,
                    },
                )
            )
            logger.debug(f"Inserted document: {document.name} ({document.meta_data})")
        if len(points) > 0:
            self.client.upsert(collection_name=self.collection, wait=False, points=points)
        logger.debug(f"Upsert {len(points)} documents")

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
        """
        Search for documents in the database.

        Args:
            query (str): Query to search for
            limit (int): Number of search results to return
            filters (Optional[Dict[str, Any]]): Filters to apply while searching
        """
        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )

        # Build search results
        search_results: List[Document] = []
        for result in results:
            if result.payload is None:
                continue
            search_results.append(
                Document(
                    name=result.payload["name"],
                    meta_data=result.payload["meta_data"],
                    content=result.payload["content"],
                    embedder=self.embedder,
                    embedding=result.vector,
                    usage=result.payload["usage"],
                )
            )

        if self.reranker:
            search_results = self.reranker.rerank(query=query, documents=search_results)

        return search_results

    def drop(self) -> None:
        if self.exists():
            logger.debug(f"Deleting collection: {self.collection}")
            self.client.delete_collection(self.collection)

    def exists(self) -> bool:
        if self.client:
            collections_response: models.CollectionsResponse = self.client.get_collections()
            collections: List[models.CollectionDescription] = collections_response.collections
            for collection in collections:
                if collection.name == self.collection:
                    # collection.status == models.CollectionStatus.GREEN
                    return True
        return False

    def get_count(self) -> int:
        count_result: models.CountResult = self.client.count(collection_name=self.collection, exact=True)
        return count_result.count

    def optimize(self) -> None:
        pass

    def delete(self) -> bool:
        return False
