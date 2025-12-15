# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base database class for unified storage

This module provides a unified database abstraction for storing:
- Sessions: Agent/Workflow session history
- Memories: User memories (long-term memory)
- Metrics: Usage metrics and statistics
- Knowledge: RAG knowledge documents
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from hashlib import md5
import json

from pydantic import BaseModel, Field, ConfigDict, model_validator


class SessionRow(BaseModel):
    """Session record stored in database"""
    session_id: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None
    agent_data: Optional[Dict[str, Any]] = None
    user_data: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class MemoryRow(BaseModel):
    """Memory record stored in database"""
    id: Optional[str] = None
    user_id: Optional[str] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

    def to_dict(self) -> Dict[str, Any]:
        return self.serializable_dict()

    @model_validator(mode="after")
    def generate_id(self) -> "MemoryRow":
        if self.id is None:
            memory_str = json.dumps(self.memory, sort_keys=True, ensure_ascii=False)
            cleaned_memory = memory_str.replace(" ", "").replace("\n", "").replace("\t", "")
            self.id = md5(cleaned_memory.encode()).hexdigest()
        return self


class MetricsRow(BaseModel):
    """Metrics record stored in database"""
    id: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class KnowledgeRow(BaseModel):
    """Knowledge document record stored in database for RAG"""
    id: str
    name: str
    description: str = ""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    doc_type: Optional[str] = None  # file type: pdf, txt, url, etc.
    size: Optional[int] = None  # file size in bytes
    status: Optional[str] = None  # processing status
    status_message: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class BaseDb(ABC):
    """
    Base class for unified database storage.

    Manages multiple tables:
    - sessions: Agent/Workflow session history
    - memories: User memories (long-term memory)
    - metrics: Usage metrics and statistics
    - knowledge: RAG knowledge documents

    Example:
        >>> from agentica.db.sqlite import SqliteDb
        >>> db = SqliteDb(db_file="agent.db")
        >>> # Use with Agent (auto creates tables)
        >>> from agentica import Agent
        >>> agent = Agent(db=db)
        >>>
        >>> # Use for RAG knowledge
        >>> contents_db = SqliteDb(db_file="data.db", knowledge_table="knowledge_contents")
    """

    def __init__(
        self,
        session_table: Optional[str] = None,
        memory_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
    ):
        self.session_table_name = session_table or "agentica_sessions"
        self.memory_table_name = memory_table or "agentica_memories"
        self.metrics_table_name = metrics_table or "agentica_metrics"
        self.knowledge_table_name = knowledge_table  # Optional, None means not used

    # ==================== Session Operations ====================

    @abstractmethod
    def create_session_table(self) -> None:
        """Create session table if not exists"""
        raise NotImplementedError

    @abstractmethod
    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]:
        """Read a session by ID"""
        raise NotImplementedError

    @abstractmethod
    def upsert_session(self, session: SessionRow) -> Optional[SessionRow]:
        """Insert or update a session"""
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session"""
        raise NotImplementedError

    @abstractmethod
    def get_all_session_ids(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[str]:
        """Get all session IDs"""
        raise NotImplementedError

    @abstractmethod
    def get_all_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[SessionRow]:
        """Get all sessions"""
        raise NotImplementedError

    # ==================== Memory Operations ====================

    @abstractmethod
    def create_memory_table(self) -> None:
        """Create memory table if not exists"""
        raise NotImplementedError

    @abstractmethod
    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None  # "asc" or "desc"
    ) -> List[MemoryRow]:
        """Read memories for a user"""
        raise NotImplementedError

    @abstractmethod
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        """Insert or update a memory"""
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        """Delete a memory"""
        raise NotImplementedError

    @abstractmethod
    def memory_exists(self, memory: MemoryRow) -> bool:
        """Check if memory exists"""
        raise NotImplementedError

    @abstractmethod
    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        """Clear all memories for a user"""
        raise NotImplementedError

    # ==================== Metrics Operations ====================

    @abstractmethod
    def create_metrics_table(self) -> None:
        """Create metrics table if not exists"""
        raise NotImplementedError

    @abstractmethod
    def insert_metrics(self, metrics: MetricsRow) -> None:
        """Insert metrics record"""
        raise NotImplementedError

    @abstractmethod
    def get_metrics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MetricsRow]:
        """Get metrics records"""
        raise NotImplementedError

    # ==================== Knowledge Operations ====================

    @abstractmethod
    def create_knowledge_table(self) -> None:
        """Create knowledge table if not exists"""
        raise NotImplementedError

    @abstractmethod
    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]:
        """Insert or update a knowledge document"""
        raise NotImplementedError

    @abstractmethod
    def read_knowledge(self, knowledge_id: str) -> Optional[KnowledgeRow]:
        """Read a knowledge document by ID"""
        raise NotImplementedError

    @abstractmethod
    def get_all_knowledge(
        self,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[KnowledgeRow]:
        """Get all knowledge documents"""
        raise NotImplementedError

    @abstractmethod
    def delete_knowledge(self, knowledge_id: str) -> None:
        """Delete a knowledge document"""
        raise NotImplementedError

    @abstractmethod
    def clear_knowledge(self) -> bool:
        """Clear all knowledge documents"""
        raise NotImplementedError

    # ==================== Lifecycle ====================

    def create(self) -> None:
        """Create all tables"""
        self.create_session_table()
        self.create_memory_table()
        self.create_metrics_table()
        if self.knowledge_table_name:
            self.create_knowledge_table()

    @abstractmethod
    def drop(self) -> None:
        """Drop all tables"""
        raise NotImplementedError

    @abstractmethod
    def upgrade_schema(self) -> None:
        """Upgrade database schema if needed"""
        raise NotImplementedError
