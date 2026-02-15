# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Workspace memory search (keyword + vector hybrid search)
"""

import math
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from agentica.utils.log import logger


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class MemoryChunk(BaseModel):
    """A chunk of memory content for search results."""

    content: str = Field(description="The memory content")
    file_path: str = Field(description="Path to the source file")
    start_line: int = Field(default=1, description="Starting line number in the file")
    end_line: int = Field(default=1, description="Ending line number in the file")
    score: float = Field(default=0.0, description="Relevance score")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def chunk_key(self) -> str:
        """Unique key for this chunk."""
        return f"{self.file_path}:{self.start_line}"


class WorkspaceMemorySearch(BaseModel):
    """
    Workspace memory search for searching Markdown files in a workspace.

    Supports keyword search and vector search (hybrid search) when an
    embedding model is provided.

    Example:
        >>> from agentica.memory import WorkspaceMemorySearch
        >>> from pathlib import Path
        >>>
        >>> searcher = WorkspaceMemorySearch(workspace_path=Path("~/.agentica/workspace"))
        >>> searcher.index()
        >>> results = searcher.search("project deadline")
        >>>
        >>> # With vector search (requires OpenAI API key)
        >>> results = searcher.search_hybrid("project deadline")
    """

    workspace_path: str = Field(description="Path to the workspace directory")
    chunk_lines: int = Field(default=20, description="Number of lines per chunk")
    overlap_lines: int = Field(default=4, description="Number of overlap lines between chunks")
    chunks: List[MemoryChunk] = Field(default_factory=list, description="Indexed memory chunks")
    _embeddings: Dict[int, List[float]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Initialize per-instance _embeddings to avoid class-variable sharing."""
        self._embeddings = {}

    def _get_path(self) -> "Path":
        """Get workspace path as Path object."""
        from pathlib import Path
        return Path(self.workspace_path).expanduser().resolve()

    def index(self, patterns: Optional[List[str]] = None) -> int:
        """Index Markdown files in the workspace.

        Searches both workspace root and users/ subdirectories for memory files.
        """
        if patterns is None:
            patterns = ["MEMORY.md", "memory/*.md"]

        self.chunks = []
        self._embeddings = {}
        workspace = self._get_path()

        # Search workspace root level
        for pattern in patterns:
            for file_path in workspace.glob(pattern):
                if file_path.is_file() and file_path.suffix == ".md":
                    self._index_file(file_path)

        # Search users/ subdirectories (multi-user workspace structure)
        users_dir = workspace / "users"
        if users_dir.exists():
            for user_dir in users_dir.iterdir():
                if user_dir.is_dir():
                    for pattern in patterns:
                        for file_path in user_dir.glob(pattern):
                            if file_path.is_file() and file_path.suffix == ".md":
                                self._index_file(file_path)

        logger.debug(f"Indexed {len(self.chunks)} chunks from workspace {workspace}")
        return len(self.chunks)

    def _index_file(self, file_path: "Path"):
        """Index a single file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            new_chunks = self._chunk_lines(
                lines,
                str(file_path.relative_to(self._get_path()))
            )
            self.chunks.extend(new_chunks)
        except Exception as e:
            logger.warning(f"Failed to index file {file_path}: {e}")

    def _chunk_lines(self, lines: List[str], file_path: str) -> List[MemoryChunk]:
        """Split file into chunks by lines."""
        chunks = []

        i = 0
        while i < len(lines):
            end = min(i + self.chunk_lines, len(lines))
            chunk_content = "\n".join(lines[i:end])

            if chunk_content.strip():
                chunks.append(MemoryChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end,
                    score=0.0,
                ))

            i = end - self.overlap_lines if end < len(lines) else len(lines)

        return chunks

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> List[MemoryChunk]:
        """Search memory using keyword matching."""
        if not self.chunks:
            return []

        query_lower = query.lower()
        query_words = query_lower.split()
        results = []

        for chunk in self.chunks:
            content_lower = chunk.content.lower()

            score = 0.0
            for word in query_words:
                if word in content_lower:
                    score += 1.0 / len(query_words)

            if score >= min_score:
                result = MemoryChunk(
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=score,
                )
                results.append(result)

        results.sort(key=lambda x: -x.score)
        return results[:limit]

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        embedding: Optional[Any] = None,
    ) -> List[MemoryChunk]:
        """Hybrid search combining vector similarity and keyword matching."""
        if not self.chunks:
            return []

        keyword_results = self.search(query, limit=len(self.chunks), min_score=0.0)
        keyword_scores = {r.chunk_key: r.score for r in keyword_results}

        vector_scores: Dict[str, float] = {}
        try:
            if embedding is None:
                from agentica.embedding.openai import OpenAIEmbedding
                embedding = OpenAIEmbedding()

            query_embedding = embedding.get_embedding(query)
            if not query_embedding:
                logger.warning("Failed to get query embedding, falling back to keyword search")
                return self.search(query, limit)

            self._compute_embeddings(embedding)

            for i, chunk in enumerate(self.chunks):
                if i in self._embeddings:
                    similarity = cosine_similarity(query_embedding, self._embeddings[i])
                    vector_scores[chunk.chunk_key] = (similarity + 1) / 2

        except ImportError:
            logger.warning("OpenAI not available, falling back to keyword search")
            return self.search(query, limit)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}, falling back to keyword search")
            return self.search(query, limit)

        results = []
        for chunk in self.chunks:
            kw_score = keyword_scores.get(chunk.chunk_key, 0.0)
            vec_score = vector_scores.get(chunk.chunk_key, 0.0)

            combined_score = (keyword_weight * kw_score) + (vector_weight * vec_score)

            if combined_score > 0:
                result = MemoryChunk(
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=combined_score,
                )
                results.append(result)

        results.sort(key=lambda x: -x.score)
        return results[:limit]

    def _compute_embeddings(self, embedding: Any) -> None:
        """Compute and cache embeddings for all chunks."""
        for i, chunk in enumerate(self.chunks):
            if i not in self._embeddings:
                try:
                    emb = embedding.get_embedding(chunk.content)
                    if emb:
                        self._embeddings[i] = emb
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for chunk {i}: {e}")

    def clear(self):
        """Clear all indexed chunks and embeddings cache."""
        self.chunks = []
        self._embeddings = {}

    def get_context(self, query: str, max_chars: int = 2000) -> str:
        """Get relevant memory context for a query."""
        results = self.search(query, limit=10)
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for chunk in results:
            if total_chars + len(chunk.content) > max_chars:
                break
            context_parts.append(f"[{chunk.file_path}:{chunk.start_line}]\n{chunk.content}")
            total_chars += len(chunk.content)

        return "\n\n---\n\n".join(context_parts)
