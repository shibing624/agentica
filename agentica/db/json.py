# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: JSON file implementation of BaseDb
"""
import json
import time
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow, KnowledgeRow
from agentica.utils.log import logger


class JsonDb(BaseDb):
    """
    JSON file implementation of unified database storage.
    
    Data is persisted to a JSON file. Suitable for lightweight storage needs
    where SQLite might be overkill.

    Example:
        >>> db = JsonDb(db_file="outputs/agent_data.json")
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
        db_file: str = "agentica_db.json",
        session_table: Optional[str] = None,
        memory_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
    ):
        """
        Initialize JsonDb.

        Args:
            db_file: Path to JSON database file
            session_table: Name of sessions table (for namespacing)
            memory_table: Name of memories table (for namespacing)
            metrics_table: Name of metrics table (for namespacing)
            knowledge_table: Name of knowledge table (for namespacing)
        """
        super().__init__(session_table, memory_table, metrics_table, knowledge_table)

        self.db_file = Path(db_file).resolve()
        self._lock = threading.RLock()
        
        # In-memory cache
        self._data: Dict[str, Any] = {
            "sessions": {},
            "memories": {},
            "metrics": [],
            "knowledge": {},
        }
        
        # Load existing data if file exists
        self._load()
        
        # Auto-create tables
        self.create()

    def _load(self) -> None:
        """Load data from JSON file"""
        if self.db_file.exists():
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                    # Ensure all keys exist
                    if "sessions" not in self._data:
                        self._data["sessions"] = {}
                    if "memories" not in self._data:
                        self._data["memories"] = {}
                    if "metrics" not in self._data:
                        self._data["metrics"] = []
                    if "knowledge" not in self._data:
                        self._data["knowledge"] = {}
                logger.debug(f"Loaded data from {self.db_file}")
            except Exception as e:
                logger.warning(f"Error loading JSON file: {e}, starting with empty data")
                self._data = {"sessions": {}, "memories": {}, "metrics": [], "knowledge": {}}

    def _save(self) -> None:
        """Save data to JSON file"""
        try:
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Saved data to {self.db_file}")
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")

    # ==================== Session Operations ====================

    def create_session_table(self) -> None:
        """Create session table (ensure sessions key exists)"""
        with self._lock:
            if "sessions" not in self._data:
                self._data["sessions"] = {}
                self._save()
        logger.debug(f"JSON session storage initialized: {self.session_table_name}")

    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]:
        with self._lock:
            session_data = self._data["sessions"].get(session_id)
            if session_data:
                if user_id and session_data.get("user_id") != user_id:
                    return None
                return SessionRow(**session_data)
        return None

    def upsert_session(self, session_row: SessionRow) -> Optional[SessionRow]:
        now = int(time.time())
        
        with self._lock:
            existing = self._data["sessions"].get(session_row.session_id)
            
            session_dict = session_row.model_dump()
            if existing:
                session_dict["created_at"] = existing.get("created_at", now)
                session_dict["updated_at"] = now
            else:
                session_dict["created_at"] = session_row.created_at or now
                session_dict["updated_at"] = now
            
            self._data["sessions"][session_row.session_id] = session_dict
            self._save()
        
        logger.debug(f"Session upserted: {session_row.session_id}")
        return SessionRow(**session_dict)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._data["sessions"]:
                del self._data["sessions"][session_id]
                self._save()
                logger.debug(f"Session deleted: {session_id}")

    def get_all_session_ids(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[str]:
        with self._lock:
            sessions = list(self._data["sessions"].values())
        
        if user_id:
            sessions = [s for s in sessions if s.get("user_id") == user_id]
        if agent_id:
            sessions = [s for s in sessions if s.get("agent_id") == agent_id]
        
        # Sort by created_at descending
        sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return [s["session_id"] for s in sessions]

    def get_all_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[SessionRow]:
        with self._lock:
            sessions = list(self._data["sessions"].values())
        
        if user_id:
            sessions = [s for s in sessions if s.get("user_id") == user_id]
        if agent_id:
            sessions = [s for s in sessions if s.get("agent_id") == agent_id]
        
        # Sort by created_at descending
        sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return [SessionRow(**s) for s in sessions]

    # ==================== Memory Operations ====================

    def create_memory_table(self) -> None:
        """Create memory table (ensure memories key exists)"""
        with self._lock:
            if "memories" not in self._data:
                self._data["memories"] = {}
                self._save()
        logger.debug(f"JSON memory storage initialized: {self.memory_table_name}")

    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None
    ) -> List[MemoryRow]:
        with self._lock:
            memories = list(self._data["memories"].values())
        
        if user_id:
            memories = [m for m in memories if m.get("user_id") == user_id]
        
        # Sort by created_at
        reverse = sort != "asc"
        memories.sort(
            key=lambda m: m.get("created_at", ""),
            reverse=reverse
        )
        
        if limit:
            memories = memories[:limit]
        
        result = []
        for m in memories:
            m_copy = m.copy()
            # Parse datetime strings
            if isinstance(m_copy.get("created_at"), str):
                try:
                    m_copy["created_at"] = datetime.fromisoformat(m_copy["created_at"])
                except (ValueError, TypeError):
                    m_copy["created_at"] = None
            if isinstance(m_copy.get("updated_at"), str):
                try:
                    m_copy["updated_at"] = datetime.fromisoformat(m_copy["updated_at"])
                except (ValueError, TypeError):
                    m_copy["updated_at"] = None
            result.append(MemoryRow(**m_copy))
        
        return result

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        now = datetime.now()
        
        with self._lock:
            existing = self._data["memories"].get(memory.id)
            
            memory_dict = {
                "id": memory.id,
                "user_id": memory.user_id,
                "memory": memory.memory,
            }
            
            if existing:
                memory_dict["created_at"] = existing.get("created_at", now.isoformat())
                memory_dict["updated_at"] = now.isoformat()
            else:
                memory_dict["created_at"] = memory.created_at.isoformat() if memory.created_at else now.isoformat()
                memory_dict["updated_at"] = now.isoformat()
            
            self._data["memories"][memory.id] = memory_dict
            self._save()
        
        logger.debug(f"Memory upserted: {memory.id}")
        return memory

    def delete_memory(self, memory_id: str) -> None:
        with self._lock:
            if memory_id in self._data["memories"]:
                del self._data["memories"][memory_id]
                self._save()
                logger.debug(f"Memory deleted: {memory_id}")

    def memory_exists(self, memory: MemoryRow) -> bool:
        with self._lock:
            return memory.id in self._data["memories"]

    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        with self._lock:
            if user_id:
                to_delete = [
                    mid for mid, m in self._data["memories"].items()
                    if m.get("user_id") == user_id
                ]
                for mid in to_delete:
                    del self._data["memories"][mid]
            else:
                self._data["memories"] = {}
            self._save()
        logger.debug(f"Memories cleared for user: {user_id}")
        return True

    # ==================== Metrics Operations ====================

    def create_metrics_table(self) -> None:
        """Create metrics table (ensure metrics key exists)"""
        with self._lock:
            if "metrics" not in self._data:
                self._data["metrics"] = []
                self._save()
        logger.debug(f"JSON metrics storage initialized: {self.metrics_table_name}")

    def insert_metrics(self, metrics: MetricsRow) -> None:
        now = datetime.now()
        
        metrics_dict = {
            "id": metrics.id,
            "agent_id": metrics.agent_id,
            "session_id": metrics.session_id,
            "run_id": metrics.run_id,
            "metrics": metrics.metrics,
            "created_at": metrics.created_at.isoformat() if metrics.created_at else now.isoformat(),
        }
        
        with self._lock:
            self._data["metrics"].append(metrics_dict)
            self._save()
        
        logger.debug(f"Metrics inserted: {metrics.id}")

    def get_metrics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MetricsRow]:
        with self._lock:
            metrics = self._data["metrics"].copy()
        
        if agent_id:
            metrics = [m for m in metrics if m.get("agent_id") == agent_id]
        if session_id:
            metrics = [m for m in metrics if m.get("session_id") == session_id]
        
        # Sort by created_at descending
        metrics.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        
        if limit:
            metrics = metrics[:limit]
        
        result = []
        for m in metrics:
            m_copy = m.copy()
            if isinstance(m_copy.get("created_at"), str):
                try:
                    m_copy["created_at"] = datetime.fromisoformat(m_copy["created_at"])
                except (ValueError, TypeError):
                    m_copy["created_at"] = None
            result.append(MetricsRow(**m_copy))
        
        return result

    # ==================== Knowledge Operations ====================

    def create_knowledge_table(self) -> None:
        """Create knowledge table (ensure knowledge key exists)"""
        if not self.knowledge_table_name:
            return
        with self._lock:
            if "knowledge" not in self._data:
                self._data["knowledge"] = {}
                self._save()
        logger.debug(f"JSON knowledge storage initialized: {self.knowledge_table_name}")

    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            logger.warning("Knowledge table not configured")
            return None
        now = int(time.time())
        
        with self._lock:
            existing = self._data["knowledge"].get(knowledge.id)
            
            knowledge_dict = {
                "id": knowledge.id,
                "name": knowledge.name,
                "description": knowledge.description,
                "content": knowledge.content,
                "metadata": knowledge.metadata,
                "doc_type": knowledge.doc_type,
                "size": knowledge.size,
                "status": knowledge.status,
                "status_message": knowledge.status_message,
            }
            
            if existing:
                knowledge_dict["created_at"] = existing.get("created_at", now)
                knowledge_dict["updated_at"] = now
            else:
                knowledge_dict["created_at"] = knowledge.created_at or now
                knowledge_dict["updated_at"] = now
            
            self._data["knowledge"][knowledge.id] = knowledge_dict
            self._save()
        
        logger.debug(f"Knowledge upserted: {knowledge.id}")
        return KnowledgeRow(**knowledge_dict)

    def read_knowledge(self, knowledge_id: str) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            return None
        with self._lock:
            knowledge_data = self._data["knowledge"].get(knowledge_id)
            if knowledge_data:
                return KnowledgeRow(**knowledge_data)
        return None

    def get_all_knowledge(
        self,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[KnowledgeRow]:
        if not self.knowledge_table_name:
            return []
        with self._lock:
            knowledge_list = list(self._data["knowledge"].values())
        
        if doc_type:
            knowledge_list = [k for k in knowledge_list if k.get("doc_type") == doc_type]
        if status:
            knowledge_list = [k for k in knowledge_list if k.get("status") == status]
        
        # Sort by created_at descending
        knowledge_list.sort(key=lambda k: k.get("created_at", 0), reverse=True)
        
        if limit:
            knowledge_list = knowledge_list[:limit]
        
        return [KnowledgeRow(**k) for k in knowledge_list]

    def delete_knowledge(self, knowledge_id: str) -> None:
        if not self.knowledge_table_name:
            return
        with self._lock:
            if knowledge_id in self._data["knowledge"]:
                del self._data["knowledge"][knowledge_id]
                self._save()
                logger.debug(f"Knowledge deleted: {knowledge_id}")

    def clear_knowledge(self) -> bool:
        if not self.knowledge_table_name:
            return False
        with self._lock:
            self._data["knowledge"] = {}
            self._save()
        logger.debug("All knowledge cleared")
        return True

    # ==================== Lifecycle ====================

    def drop(self) -> None:
        """Drop all tables (clear all data and delete file)"""
        with self._lock:
            self._data = {"sessions": {}, "memories": {}, "metrics": [], "knowledge": {}}
            if self.db_file.exists():
                try:
                    self.db_file.unlink()
                    logger.debug(f"JSON file deleted: {self.db_file}")
                except Exception as e:
                    logger.error(f"Error deleting JSON file: {e}")

    def upgrade_schema(self) -> None:
        """Upgrade database schema (no-op for JSON)"""
        pass

    def __deepcopy__(self, memo):
        """Create a deep copy of the JsonDb instance"""
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj

        for k, v in self.__dict__.items():
            if k == "_lock":
                copied_obj._lock = threading.RLock()
            else:
                setattr(copied_obj, k, deepcopy(v, memo))

        return copied_obj
