# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Qdrant-based Memory Database with semantic search support
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import md5
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

from agentica.memorydb import MemoryDb, MemoryRow
from agentica.emb.base import Emb
from agentica.utils.log import logger


class QdrantMemoryDb(MemoryDb):
    """
    Qdrant-based Memory Database with semantic search support.
    
    Features:
    - Vector-based semantic search using embeddings
    - Fallback to keyword search when embedder is unavailable
    - Local disk storage (on_disk=True by default)
    """
    
    def __init__(
        self,
        collection: str = "qdrant_memory",
        embedder: Optional[Emb] = None,
        path: Optional[str] = None,
        on_disk: bool = True,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize QdrantMemoryDb.
        
        Args:
            collection: Name of the Qdrant collection
            embedder: Embedding model for semantic search (optional, falls back to keyword search if None)
            path: Path for local disk storage (used when on_disk=True)
            on_disk: Whether to use local disk storage (default: True)
            location: Qdrant server location (":memory:" for in-memory, path for disk)
            url: Qdrant server URL (for remote server)
            port: Qdrant server port
            api_key: API key for Qdrant Cloud
            **kwargs: Additional arguments passed to QdrantClient
        """
        self.collection = collection
        self.embedder = embedder
        self.on_disk = on_disk
        self._models = models
        
        # Determine dimensions from embedder or use default
        self.dimensions = 1536
        if embedder is not None and hasattr(embedder, 'dimensions'):
            self.dimensions = embedder.dimensions
        
        # Setup storage path
        if on_disk and path is None:
            path = os.path.join(os.path.expanduser("~"), ".agentica", "qdrant_memory")
        # Expand ~ in path
        if path:
            path = os.path.expanduser(path)
        self.path = path
        
        # Create storage directory if needed
        if self.path:
            os.makedirs(self.path, exist_ok=True)
        
        # Initialize Qdrant client
        if url:
            # Remote Qdrant server
            self._client = QdrantClient(
                url=url,
                port=port,
                api_key=api_key,
                **kwargs
            )
        elif on_disk and self.path:
            # Local disk storage
            self._client = QdrantClient(path=self.path, **kwargs)
        else:
            # In-memory storage
            self._client = QdrantClient(location=":memory:", **kwargs)
        
        # Track if embedder is available and working
        self._embedder_available = self._check_embedder()
        
        # Create collection if it doesn't exist
        self.create()
    
    def _check_embedder(self) -> bool:
        """Check if embedder is available and working."""
        if self.embedder is None:
            # logger.debug("No embedder provided, will use keyword search")
            return False
        
        try:
            # Try to get a test embedding
            test_embedding = self.embedder.get_embedding("test")
            if test_embedding and len(test_embedding) > 0:
                self.dimensions = len(test_embedding)
                logger.debug(f"Embedder available, dimensions: {self.dimensions}")
                return True
        except Exception as e:
            logger.warning(f"Embedder check failed: {e}, falling back to keyword search")
        
        return False
    
    @property
    def client(self):
        return self._client
    
    def create(self) -> None:
        """Create the collection if it doesn't exist."""
        if not self.table_exists():
            logger.debug(f"Creating Qdrant collection: {self.collection}")
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=self._models.VectorParams(
                    size=self.dimensions,
                    distance=self._models.Distance.COSINE
                )
            )
    
    def table_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            collections = self._client.get_collections().collections
            return any(c.name == self.collection for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, return None if embedder unavailable."""
        if not self._embedder_available or self.embedder is None:
            return None
        
        try:
            embedding = self.embedder.get_embedding(text)
            return embedding if embedding else None
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            self._embedder_available = False
            return None
    
    def _generate_id(self, memory: Dict[str, Any]) -> str:
        """Generate a unique ID for a memory."""
        memory_str = json.dumps(memory, sort_keys=True, ensure_ascii=False)
        cleaned = memory_str.replace(" ", "").replace("\n", "").replace("\t", "")
        return md5(cleaned.encode()).hexdigest()
    
    def memory_exists(self, memory: MemoryRow) -> bool:
        """Check if a memory exists in the collection."""
        try:
            result = self._client.retrieve(
                collection_name=self.collection,
                ids=[memory.id]
            )
            return len(result) > 0
        except Exception as e:
            logger.debug(f"Error checking memory existence: {e}")
            return False
    
    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None
    ) -> List[MemoryRow]:
        """Read memories from the collection."""
        memories: List[MemoryRow] = []
        
        try:
            # Build filter
            filter_conditions = []
            if user_id is not None:
                filter_conditions.append(
                    self._models.FieldCondition(
                        key="user_id",
                        match=self._models.MatchValue(value=user_id)
                    )
                )
            
            scroll_filter = None
            if filter_conditions:
                scroll_filter = self._models.Filter(must=filter_conditions)
            
            # Scroll through all matching points
            results, _ = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=limit or 100,
                with_payload=True,
                with_vectors=False
            )
            
            for point in results:
                if point.payload:
                    memory_row = MemoryRow(
                        id=str(point.id),
                        user_id=point.payload.get("user_id"),
                        memory=point.payload.get("memory", {}),
                        created_at=datetime.fromisoformat(point.payload["created_at"]) if point.payload.get("created_at") else None,
                        updated_at=datetime.fromisoformat(point.payload["updated_at"]) if point.payload.get("updated_at") else None
                    )
                    memories.append(memory_row)
            
            # Sort if requested
            if sort == "asc":
                memories.sort(key=lambda x: x.created_at or datetime.min)
            elif sort == "desc":
                memories.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
            
        except Exception as e:
            logger.debug(f"Error reading memories: {e}")
        
        return memories
    
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        """Insert or update a memory in the collection."""
        try:
            # Get memory text for embedding
            memory_text = ""
            if isinstance(memory.memory, dict):
                memory_text = memory.memory.get("memory", "")
                if not memory_text:
                    memory_text = json.dumps(memory.memory, ensure_ascii=False)
            else:
                memory_text = str(memory.memory)
            
            # Get embedding or use zero vector
            embedding = self._get_embedding(memory_text)
            if embedding is None:
                # Use zero vector as placeholder when embedder unavailable
                embedding = [0.0] * self.dimensions
            
            # Prepare payload
            payload = {
                "user_id": memory.user_id,
                "memory": memory.memory,
                "memory_text": memory_text,  # Store text for keyword search
                "created_at": memory.created_at.isoformat() if memory.created_at else datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Upsert point
            self._client.upsert(
                collection_name=self.collection,
                points=[
                    self._models.PointStruct(
                        id=memory.id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.debug(f"Memory upserted: {memory.id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error upserting memory: {e}")
            return None
    
    def delete_memory(self, id: str) -> None:
        """Delete a memory from the collection."""
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=self._models.PointIdsList(points=[id])
            )
            logger.debug(f"Memory deleted: {id}")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
    
    def drop_table(self) -> None:
        """Drop the collection."""
        try:
            if self.table_exists():
                self._client.delete_collection(self.collection)
                logger.debug(f"Collection dropped: {self.collection}")
        except Exception as e:
            logger.error(f"Error dropping collection: {e}")
    
    def clear(self) -> bool:
        """Clear all memories from the collection."""
        try:
            self.drop_table()
            self.create()
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[MemoryRow]:
        """
        Search memories using semantic search with keyword fallback.
        
        Args:
            query: Search query
            user_id: Filter by user ID
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of matching MemoryRow objects
        """
        # Try semantic search first
        if self._embedder_available:
            results = self._semantic_search(query, user_id, limit, score_threshold)
            if results:
                return results
            logger.debug("Semantic search returned no results, trying keyword search")
        
        # Fallback to keyword search
        return self._keyword_search(query, user_id, limit)
    
    def _semantic_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[MemoryRow]:
        """Perform semantic search using embeddings."""
        memories: List[MemoryRow] = []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return memories
            
            # Build filter
            filter_conditions = []
            if user_id is not None:
                filter_conditions.append(
                    self._models.FieldCondition(
                        key="user_id",
                        match=self._models.MatchValue(value=user_id)
                    )
                )
            
            search_filter = None
            if filter_conditions:
                search_filter = self._models.Filter(must=filter_conditions)
            
            # Search using query_points (new API) or search (old API)
            if hasattr(self._client, 'query_points'):
                # New qdrant-client API (>=1.7.0)
                results = self._client.query_points(
                    collection_name=self.collection,
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True
                )
                points = results.points if hasattr(results, 'points') else results
            else:
                # Old qdrant-client API
                points = self._client.search(
                    collection_name=self.collection,
                    query_vector=query_embedding,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True
                )
            
            for point in points:
                if point.payload:
                    memory_row = MemoryRow(
                        id=str(point.id),
                        user_id=point.payload.get("user_id"),
                        memory=point.payload.get("memory", {}),
                        created_at=datetime.fromisoformat(point.payload["created_at"]) if point.payload.get("created_at") else None,
                        updated_at=datetime.fromisoformat(point.payload["updated_at"]) if point.payload.get("updated_at") else None
                    )
                    memories.append(memory_row)
            
            logger.debug(f"Semantic search found {len(memories)} memories")
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        return memories
    
    def _keyword_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryRow]:
        """Perform keyword-based search as fallback."""
        memories: List[MemoryRow] = []
        
        try:
            # Get all memories for the user
            all_memories = self.read_memories(user_id=user_id, limit=1000)
            
            # Simple keyword matching
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            scored_memories = []
            for memory in all_memories:
                # Get memory text
                memory_text = ""
                if isinstance(memory.memory, dict):
                    memory_text = memory.memory.get("memory", "")
                    if not memory_text:
                        memory_text = json.dumps(memory.memory, ensure_ascii=False)
                else:
                    memory_text = str(memory.memory)
                
                memory_lower = memory_text.lower()
                
                # Calculate simple relevance score
                score = 0
                # Exact substring match (highest priority)
                if query_lower in memory_lower:
                    score += 10
                
                # Word overlap
                memory_words = set(memory_lower.split())
                overlap = len(query_words & memory_words)
                score += overlap * 2
                
                # Partial word match (for compound words)
                for qword in query_words:
                    if len(qword) >= 3:  # Only check words with 3+ chars
                        for mword in memory_words:
                            if qword in mword or mword in qword:
                                score += 1
                
                if score > 0:
                    scored_memories.append((score, memory))
            
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            memories = [m for _, m in scored_memories[:limit]]
            
            logger.debug(f"Keyword search found {len(memories)} memories")
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
        
        return memories
    
    def get_relevant_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> str:
        """
        Get relevant memories as formatted string for prompt injection.
        
        Args:
            query: Search query
            user_id: Filter by user ID
            limit: Maximum number of results
        
        Returns:
            Formatted string of relevant memories
        """
        memories = self.search_memories(query, user_id, limit)
        
        if not memories:
            return ""
        
        memory_texts = []
        for memory in memories:
            if isinstance(memory.memory, dict):
                text = memory.memory.get("memory", "")
            else:
                text = str(memory.memory)
            if text:
                memory_texts.append(f"- {text}")
        
        if memory_texts:
            return "Relevant memories:\n" + "\n".join(memory_texts)
        
        return ""
