# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: In-memory implementation of BaseDb
"""
import time
from typing import Optional, List, Dict
from datetime import datetime
from copy import deepcopy

from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow, KnowledgeRow
from agentica.utils.log import logger


class InMemoryDb(BaseDb):
    """
    In-memory implementation of unified database storage.
    
    Data is stored in memory and will be lost when the process exits.
    Useful for testing, development, or temporary sessions.

    Example:
        >>> db = InMemoryDb()
        >>> 
        >>> # Store session
        >>> from agentica.db.base import SessionRow
        >>> session = SessionRow(session_id="123", agent_id="agent1")
        >>> db.upsert_session(session)
        >>> 
        >>> # Store memory
        >>> from agentica.db.base import MemoryRow
        >>> memory = MemoryRow(user_id="user1", memory={"text": "Remember this"})
        >>> db.upsert_memory(memory)
    """

    def __init__(
        self,
        session_table: Optional[str] = None,
        memory_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
    ):
        """
        Initialize InMemoryDb.

        Args:
            session_table: Name of sessions table (for namespacing)
            memory_table: Name of memories table (for namespacing)
            metrics_table: Name of metrics table (for namespacing)
            knowledge_table: Name of knowledge table (for namespacing)
        """
        super().__init__(session_table, memory_table, metrics_table, knowledge_table)

        # In-memory storage
        self._sessions: Dict[str, SessionRow] = {}
        self._memories: Dict[str, MemoryRow] = {}
        self._metrics: List[MetricsRow] = []
        self._knowledge: Dict[str, KnowledgeRow] = {}

        # Auto-create tables
        self.create()

    # ==================== Session Operations ====================

    def create_session_table(self) -> None:
        """Create session table (no-op for in-memory)"""
        logger.debug(f"In-memory session storage initialized: {self.session_table_name}")

    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]:
        session = self._sessions.get(session_id)
        if session and user_id and session.user_id != user_id:
            return None
        return deepcopy(session) if session else None

    def upsert_session(self, session_row: SessionRow) -> Optional[SessionRow]:
        now = int(time.time())
        
        existing = self._sessions.get(session_row.session_id)
        if existing:
            # Update existing session
            session_row.created_at = existing.created_at
            session_row.updated_at = now
        else:
            # New session
            session_row.created_at = session_row.created_at or now
            session_row.updated_at = now
        
        self._sessions[session_row.session_id] = deepcopy(session_row)
        logger.debug(f"Session upserted: {session_row.session_id}")
        return session_row

    def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Session deleted: {session_id}")

    def get_all_session_ids(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[str]:
        sessions = list(self._sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        
        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at or 0, reverse=True)
        return [s.session_id for s in sessions]

    def get_all_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[SessionRow]:
        sessions = list(self._sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        
        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at or 0, reverse=True)
        return [deepcopy(s) for s in sessions]

    # ==================== Memory Operations ====================

    def create_memory_table(self) -> None:
        """Create memory table (no-op for in-memory)"""
        logger.debug(f"In-memory memory storage initialized: {self.memory_table_name}")

    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories = list(self._memories.values())
        
        if user_id:
            memories = [m for m in memories if m.user_id == user_id]
        
        # Sort by created_at
        reverse = sort != "asc"
        memories.sort(key=lambda m: m.created_at or datetime.min, reverse=reverse)
        
        if limit:
            memories = memories[:limit]
        
        return [deepcopy(m) for m in memories]

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        now = datetime.now()
        
        existing = self._memories.get(memory.id)
        if existing:
            memory.created_at = existing.created_at
            memory.updated_at = now
        else:
            memory.created_at = memory.created_at or now
            memory.updated_at = now
        
        self._memories[memory.id] = deepcopy(memory)
        logger.debug(f"Memory upserted: {memory.id}")
        return memory

    def delete_memory(self, memory_id: str) -> None:
        if memory_id in self._memories:
            del self._memories[memory_id]
            logger.debug(f"Memory deleted: {memory_id}")

    def memory_exists(self, memory: MemoryRow) -> bool:
        return memory.id in self._memories

    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        if user_id:
            to_delete = [mid for mid, m in self._memories.items() if m.user_id == user_id]
            for mid in to_delete:
                del self._memories[mid]
        else:
            self._memories.clear()
        logger.debug(f"Memories cleared for user: {user_id}")
        return True

    # ==================== Metrics Operations ====================

    def create_metrics_table(self) -> None:
        """Create metrics table (no-op for in-memory)"""
        logger.debug(f"In-memory metrics storage initialized: {self.metrics_table_name}")

    def insert_metrics(self, metrics: MetricsRow) -> None:
        metrics.created_at = metrics.created_at or datetime.now()
        self._metrics.append(deepcopy(metrics))
        logger.debug(f"Metrics inserted: {metrics.id}")

    def get_metrics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MetricsRow]:
        metrics = self._metrics.copy()
        
        if agent_id:
            metrics = [m for m in metrics if m.agent_id == agent_id]
        if session_id:
            metrics = [m for m in metrics if m.session_id == session_id]
        
        # Sort by created_at descending
        metrics.sort(key=lambda m: m.created_at or datetime.min, reverse=True)
        
        if limit:
            metrics = metrics[:limit]
        
        return [deepcopy(m) for m in metrics]

    # ==================== Knowledge Operations ====================

    def create_knowledge_table(self) -> None:
        """Create knowledge table (no-op for in-memory)"""
        if self.knowledge_table_name:
            logger.debug(f"In-memory knowledge storage initialized: {self.knowledge_table_name}")

    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            logger.warning("Knowledge table not configured")
            return None
        now = int(time.time())
        
        existing = self._knowledge.get(knowledge.id)
        if existing:
            knowledge.created_at = existing.created_at
            knowledge.updated_at = now
        else:
            knowledge.created_at = knowledge.created_at or now
            knowledge.updated_at = now
        
        self._knowledge[knowledge.id] = deepcopy(knowledge)
        logger.debug(f"Knowledge upserted: {knowledge.id}")
        return knowledge

    def read_knowledge(self, knowledge_id: str) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            return None
        knowledge = self._knowledge.get(knowledge_id)
        return deepcopy(knowledge) if knowledge else None

    def get_all_knowledge(
        self,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[KnowledgeRow]:
        if not self.knowledge_table_name:
            return []
        knowledge_list = list(self._knowledge.values())
        
        if doc_type:
            knowledge_list = [k for k in knowledge_list if k.doc_type == doc_type]
        if status:
            knowledge_list = [k for k in knowledge_list if k.status == status]
        
        # Sort by created_at descending
        knowledge_list.sort(key=lambda k: k.created_at or 0, reverse=True)
        
        if limit:
            knowledge_list = knowledge_list[:limit]
        
        return [deepcopy(k) for k in knowledge_list]

    def delete_knowledge(self, knowledge_id: str) -> None:
        if not self.knowledge_table_name:
            return
        if knowledge_id in self._knowledge:
            del self._knowledge[knowledge_id]
            logger.debug(f"Knowledge deleted: {knowledge_id}")

    def clear_knowledge(self) -> bool:
        if not self.knowledge_table_name:
            return False
        self._knowledge.clear()
        logger.debug("All knowledge cleared")
        return True

    # ==================== Lifecycle ====================

    def drop(self) -> None:
        """Drop all tables (clear all data)"""
        self._sessions.clear()
        self._memories.clear()
        self._metrics.clear()
        self._knowledge.clear()
        logger.debug("All in-memory data cleared")

    def upgrade_schema(self) -> None:
        """Upgrade database schema (no-op for in-memory)"""
        pass

    def __deepcopy__(self, memo):
        """Create a deep copy of the InMemoryDb instance"""
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj

        for k, v in self.__dict__.items():
            setattr(copied_obj, k, deepcopy(v, memo))

        return copied_obj
