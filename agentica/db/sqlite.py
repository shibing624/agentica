# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite implementation of BaseDb
"""
import json
import time
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from sqlalchemy import create_engine, Column, String, Integer, Text, BigInteger
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.schema import MetaData, Table
from sqlalchemy.sql.expression import select, delete
from sqlalchemy.inspection import inspect

from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow, KnowledgeRow
from agentica.utils.log import logger


class SqliteDb(BaseDb):
    """
    SQLite implementation of unified database storage.

    Tables are auto-created on first use, no need to call db.create() explicitly.

    Example:
        >>> db = SqliteDb(db_file="outputs/agent.db")
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
        >>>
        >>> # For RAG knowledge storage
        >>> contents_db = SqliteDb(db_file="data.db", knowledge_table="knowledge_contents")
        >>> from agentica.db.base import KnowledgeRow
        >>> doc = KnowledgeRow(id="doc1", name="test.pdf", description="Test document")
        >>> contents_db.upsert_knowledge(doc)
    """

    def __init__(
        self,
        db_file: Optional[str] = None,
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        session_table: Optional[str] = None,
        memory_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        schema_version: int = 1,
        auto_upgrade_schema: bool = False,
    ):
        """
        Initialize SqliteDb.

        Args:
            db_file: Path to SQLite database file
            db_url: SQLAlchemy database URL
            db_engine: Existing SQLAlchemy engine
            session_table: Name of sessions table
            memory_table: Name of memories table
            metrics_table: Name of metrics table
            knowledge_table: Name of knowledge table (for RAG)
            schema_version: Schema version number
            auto_upgrade_schema: Whether to auto-upgrade schema
        """
        super().__init__(session_table, memory_table, metrics_table, knowledge_table)

        # Create engine
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            db_path = Path(db_file).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{db_path}")
        elif _engine is None:
            # Use in-memory SQLite if nothing provided
            _engine = create_engine("sqlite://")

        self.db_file = db_file
        self.db_url = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()
        self.inspector = inspect(self.db_engine)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))

        # Schema version
        self.schema_version = schema_version
        self.auto_upgrade_schema = auto_upgrade_schema

        # Initialize tables
        self._session_table: Optional[Table] = None
        self._memory_table: Optional[Table] = None
        self._metrics_table: Optional[Table] = None
        self._knowledge_table: Optional[Table] = None

        # Auto-create tables
        self.create()

    # ==================== Session Operations ====================

    def _get_session_table(self) -> Table:
        if self._session_table is None:
            self._session_table = Table(
                self.session_table_name,
                self.metadata,
                Column("session_id", String, primary_key=True),
                Column("agent_id", String),
                Column("user_id", String),
                Column("memory", Text),  # JSON string
                Column("agent_data", Text),
                Column("user_data", Text),
                Column("session_data", Text),
                Column("created_at", Integer),
                Column("updated_at", Integer),
                extend_existing=True,
            )
        return self._session_table

    def create_session_table(self) -> None:
        table = self._get_session_table()
        if not self.inspector.has_table(table.name):
            logger.debug(f"Creating session table: {table.name}")
            table.create(self.db_engine, checkfirst=True)

    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]:
        table = self._get_session_table()
        try:
            with self.Session() as session:
                stmt = select(table).where(table.c.session_id == session_id)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                row = session.execute(stmt).first()
                if row:
                    return SessionRow(
                        session_id=row.session_id,
                        agent_id=row.agent_id,
                        user_id=row.user_id,
                        memory=json.loads(row.memory) if row.memory else None,
                        agent_data=json.loads(row.agent_data) if row.agent_data else None,
                        user_data=json.loads(row.user_data) if row.user_data else None,
                        session_data=json.loads(row.session_data) if row.session_data else None,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
        except Exception as e:
            logger.debug(f"Error reading session: {e}")
            self.create_session_table()
        return None

    def upsert_session(self, session_row: SessionRow) -> Optional[SessionRow]:
        table = self._get_session_table()
        now = int(time.time())

        try:
            with self.Session() as session:
                existing = session.execute(
                    select(table).where(table.c.session_id == session_row.session_id)
                ).first()

                values = {
                    "agent_id": session_row.agent_id,
                    "user_id": session_row.user_id,
                    "memory": json.dumps(session_row.memory, ensure_ascii=False) if session_row.memory else None,
                    "agent_data": json.dumps(session_row.agent_data, ensure_ascii=False) if session_row.agent_data else None,
                    "user_data": json.dumps(session_row.user_data, ensure_ascii=False) if session_row.user_data else None,
                    "session_data": json.dumps(session_row.session_data, ensure_ascii=False) if session_row.session_data else None,
                    "updated_at": now,
                }

                if existing:
                    stmt = table.update().where(
                        table.c.session_id == session_row.session_id
                    ).values(**values)
                else:
                    values["session_id"] = session_row.session_id
                    values["created_at"] = session_row.created_at or now
                    stmt = table.insert().values(**values)

                session.execute(stmt)
                session.commit()
                logger.debug(f"Session upserted: {session_row.session_id}")
        except Exception as e:
            logger.debug(f"Error upserting session: {e}")
            self.create_session_table()
            return self.upsert_session(session_row)

        return session_row

    def delete_session(self, session_id: str) -> None:
        table = self._get_session_table()
        try:
            with self.Session() as session:
                stmt = delete(table).where(table.c.session_id == session_id)
                session.execute(stmt)
                session.commit()
                logger.debug(f"Session deleted: {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")

    def get_all_session_ids(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[str]:
        table = self._get_session_table()
        try:
            with self.Session() as session:
                stmt = select(table.c.session_id)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                if agent_id:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                stmt = stmt.order_by(table.c.created_at.desc())
                rows = session.execute(stmt).fetchall()
                return [row.session_id for row in rows]
        except Exception as e:
            logger.debug(f"Error getting session IDs: {e}")
            self.create_session_table()
        return []

    def get_all_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[SessionRow]:
        table = self._get_session_table()
        try:
            with self.Session() as session:
                stmt = select(table)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                if agent_id:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                stmt = stmt.order_by(table.c.created_at.desc())
                rows = session.execute(stmt).fetchall()
                return [
                    SessionRow(
                        session_id=row.session_id,
                        agent_id=row.agent_id,
                        user_id=row.user_id,
                        memory=json.loads(row.memory) if row.memory else None,
                        agent_data=json.loads(row.agent_data) if row.agent_data else None,
                        user_data=json.loads(row.user_data) if row.user_data else None,
                        session_data=json.loads(row.session_data) if row.session_data else None,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.debug(f"Error getting sessions: {e}")
            self.create_session_table()
        return []

    # ==================== Memory Operations ====================

    def _get_memory_table(self) -> Table:
        if self._memory_table is None:
            self._memory_table = Table(
                self.memory_table_name,
                self.metadata,
                Column("id", String, primary_key=True),
                Column("user_id", String, index=True),
                Column("memory", Text),  # JSON string
                Column("created_at", Text),  # ISO format
                Column("updated_at", Text),
                extend_existing=True,
            )
        return self._memory_table

    def create_memory_table(self) -> None:
        table = self._get_memory_table()
        if not self.inspector.has_table(table.name):
            logger.debug(f"Creating memory table: {table.name}")
            table.create(self.db_engine, checkfirst=True)

    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None
    ) -> List[MemoryRow]:
        table = self._get_memory_table()
        memories: List[MemoryRow] = []
        try:
            with self.Session() as session:
                stmt = select(table)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                if sort == "asc":
                    stmt = stmt.order_by(table.c.created_at.asc())
                elif sort == "desc":
                    stmt = stmt.order_by(table.c.created_at.desc())
                else:
                    stmt = stmt.order_by(table.c.created_at.desc())
                if limit:
                    stmt = stmt.limit(limit)

                rows = session.execute(stmt).fetchall()
                for row in rows:
                    memories.append(MemoryRow(
                        id=row.id,
                        user_id=row.user_id,
                        memory=json.loads(row.memory) if row.memory else {},
                        created_at=datetime.fromisoformat(row.created_at) if row.created_at else None,
                        updated_at=datetime.fromisoformat(row.updated_at) if row.updated_at else None,
                    ))
        except Exception as e:
            logger.debug(f"Error reading memories: {e}")
            self.create_memory_table()
        return memories

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        table = self._get_memory_table()
        now = datetime.now().isoformat()

        try:
            with self.Session() as session:
                existing = session.execute(
                    select(table).where(table.c.id == memory.id)
                ).first()

                if existing:
                    stmt = table.update().where(table.c.id == memory.id).values(
                        user_id=memory.user_id,
                        memory=json.dumps(memory.memory, ensure_ascii=False),
                        updated_at=now,
                    )
                else:
                    stmt = table.insert().values(
                        id=memory.id,
                        user_id=memory.user_id,
                        memory=json.dumps(memory.memory, ensure_ascii=False),
                        created_at=memory.created_at.isoformat() if memory.created_at else now,
                        updated_at=now,
                    )

                session.execute(stmt)
                session.commit()
                logger.debug(f"Memory upserted: {memory.id}")
        except Exception as e:
            logger.debug(f"Error upserting memory: {e}")
            self.create_memory_table()
            return self.upsert_memory(memory)

        return memory

    def delete_memory(self, memory_id: str) -> None:
        table = self._get_memory_table()
        try:
            with self.Session() as session:
                stmt = delete(table).where(table.c.id == memory_id)
                session.execute(stmt)
                session.commit()
                logger.debug(f"Memory deleted: {memory_id}")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")

    def memory_exists(self, memory: MemoryRow) -> bool:
        table = self._get_memory_table()
        try:
            with self.Session() as session:
                stmt = select(table.c.id).where(table.c.id == memory.id)
                return session.execute(stmt).first() is not None
        except Exception as e:
            logger.debug(f"Error checking memory existence: {e}")
        return False

    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        table = self._get_memory_table()
        try:
            with self.Session() as session:
                stmt = delete(table)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                session.execute(stmt)
                session.commit()
                logger.debug(f"Memories cleared for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False

    # ==================== Metrics Operations ====================

    def _get_metrics_table(self) -> Table:
        if self._metrics_table is None:
            self._metrics_table = Table(
                self.metrics_table_name,
                self.metadata,
                Column("id", String, primary_key=True),
                Column("agent_id", String, index=True),
                Column("session_id", String, index=True),
                Column("run_id", String),
                Column("metrics", Text),  # JSON string
                Column("created_at", Text),
                extend_existing=True,
            )
        return self._metrics_table

    def create_metrics_table(self) -> None:
        table = self._get_metrics_table()
        if not self.inspector.has_table(table.name):
            logger.debug(f"Creating metrics table: {table.name}")
            table.create(self.db_engine, checkfirst=True)

    def insert_metrics(self, metrics: MetricsRow) -> None:
        table = self._get_metrics_table()
        try:
            with self.Session() as session:
                stmt = table.insert().values(
                    id=metrics.id,
                    agent_id=metrics.agent_id,
                    session_id=metrics.session_id,
                    run_id=metrics.run_id,
                    metrics=json.dumps(metrics.metrics, ensure_ascii=False),
                    created_at=datetime.now().isoformat(),
                )
                session.execute(stmt)
                session.commit()
                logger.debug(f"Metrics inserted: {metrics.id}")
        except Exception as e:
            logger.debug(f"Error inserting metrics: {e}")
            self.create_metrics_table()
            self.insert_metrics(metrics)

    def get_metrics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MetricsRow]:
        table = self._get_metrics_table()
        try:
            with self.Session() as session:
                stmt = select(table)
                if agent_id:
                    stmt = stmt.where(table.c.agent_id == agent_id)
                if session_id:
                    stmt = stmt.where(table.c.session_id == session_id)
                stmt = stmt.order_by(table.c.created_at.desc())
                if limit:
                    stmt = stmt.limit(limit)

                rows = session.execute(stmt).fetchall()
                return [
                    MetricsRow(
                        id=row.id,
                        agent_id=row.agent_id,
                        session_id=row.session_id,
                        run_id=row.run_id,
                        metrics=json.loads(row.metrics) if row.metrics else {},
                        created_at=datetime.fromisoformat(row.created_at) if row.created_at else None,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.debug(f"Error getting metrics: {e}")
            self.create_metrics_table()
        return []

    # ==================== Knowledge Operations ====================

    def _get_knowledge_table(self) -> Table:
        if self._knowledge_table is None:
            table_name = self.knowledge_table_name or "agentica_knowledge"
            self._knowledge_table = Table(
                table_name,
                self.metadata,
                Column("id", String, primary_key=True),
                Column("name", String, nullable=False),
                Column("description", Text),
                Column("content", Text),  # Document content
                Column("metadata", Text),  # JSON string
                Column("doc_type", String),  # file type
                Column("size", BigInteger),  # file size
                Column("status", String),  # processing status
                Column("status_message", Text),
                Column("created_at", BigInteger),
                Column("updated_at", BigInteger),
                extend_existing=True,
            )
        return self._knowledge_table

    def create_knowledge_table(self) -> None:
        if not self.knowledge_table_name:
            return  # Skip if knowledge table not configured
        table = self._get_knowledge_table()
        if not self.inspector.has_table(table.name):
            logger.debug(f"Creating knowledge table: {table.name}")
            table.create(self.db_engine, checkfirst=True)

    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            logger.warning("Knowledge table not configured")
            return None
        table = self._get_knowledge_table()
        now = int(time.time())

        try:
            with self.Session() as session:
                existing = session.execute(
                    select(table).where(table.c.id == knowledge.id)
                ).first()

                values = {
                    "name": knowledge.name,
                    "description": knowledge.description,
                    "content": knowledge.content,
                    "metadata": json.dumps(knowledge.metadata, ensure_ascii=False) if knowledge.metadata else None,
                    "doc_type": knowledge.doc_type,
                    "size": knowledge.size,
                    "status": knowledge.status,
                    "status_message": knowledge.status_message,
                    "updated_at": now,
                }

                if existing:
                    stmt = table.update().where(table.c.id == knowledge.id).values(**values)
                else:
                    values["id"] = knowledge.id
                    values["created_at"] = knowledge.created_at or now
                    stmt = table.insert().values(**values)

                session.execute(stmt)
                session.commit()
                logger.debug(f"Knowledge upserted: {knowledge.id}")
        except Exception as e:
            logger.debug(f"Error upserting knowledge: {e}")
            self.create_knowledge_table()
            return self.upsert_knowledge(knowledge)

        return knowledge

    def read_knowledge(self, knowledge_id: str) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            return None
        table = self._get_knowledge_table()
        try:
            with self.Session() as session:
                stmt = select(table).where(table.c.id == knowledge_id)
                row = session.execute(stmt).first()
                if row:
                    return KnowledgeRow(
                        id=row.id,
                        name=row.name,
                        description=row.description or "",
                        content=row.content,
                        metadata=json.loads(row.metadata) if row.metadata else None,
                        doc_type=row.doc_type,
                        size=row.size,
                        status=row.status,
                        status_message=row.status_message,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
        except Exception as e:
            logger.debug(f"Error reading knowledge: {e}")
            self.create_knowledge_table()
        return None

    def get_all_knowledge(
        self,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[KnowledgeRow]:
        if not self.knowledge_table_name:
            return []
        table = self._get_knowledge_table()
        try:
            with self.Session() as session:
                stmt = select(table)
                if doc_type:
                    stmt = stmt.where(table.c.doc_type == doc_type)
                if status:
                    stmt = stmt.where(table.c.status == status)
                stmt = stmt.order_by(table.c.created_at.desc())
                if limit:
                    stmt = stmt.limit(limit)

                rows = session.execute(stmt).fetchall()
                return [
                    KnowledgeRow(
                        id=row.id,
                        name=row.name,
                        description=row.description or "",
                        content=row.content,
                        metadata=json.loads(row.metadata) if row.metadata else None,
                        doc_type=row.doc_type,
                        size=row.size,
                        status=row.status,
                        status_message=row.status_message,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.debug(f"Error getting knowledge: {e}")
            self.create_knowledge_table()
        return []

    def delete_knowledge(self, knowledge_id: str) -> None:
        if not self.knowledge_table_name:
            return
        table = self._get_knowledge_table()
        try:
            with self.Session() as session:
                stmt = delete(table).where(table.c.id == knowledge_id)
                session.execute(stmt)
                session.commit()
                logger.debug(f"Knowledge deleted: {knowledge_id}")
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")

    def clear_knowledge(self) -> bool:
        if not self.knowledge_table_name:
            return False
        table = self._get_knowledge_table()
        try:
            with self.Session() as session:
                stmt = delete(table)
                session.execute(stmt)
                session.commit()
                logger.debug("All knowledge cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge: {e}")
            return False

    # ==================== Lifecycle ====================

    def drop(self) -> None:
        """Drop all tables"""
        for table in [self._session_table, self._memory_table, self._metrics_table, self._knowledge_table]:
            if table is not None:
                try:
                    if self.inspector.has_table(table.name):
                        table.drop(self.db_engine)
                        logger.debug(f"Table dropped: {table.name}")
                except Exception as e:
                    logger.error(f"Error dropping table {table.name}: {e}")

    def upgrade_schema(self) -> None:
        """Upgrade database schema if needed"""
        # TODO: Implement schema migration logic
        pass

    def __deepcopy__(self, memo):
        """Create a deep copy of the SqliteDb instance"""
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj

        for k, v in self.__dict__.items():
            if k in {"metadata", "_session_table", "_memory_table", "_metrics_table", "_knowledge_table"}:
                continue
            elif k in {"db_engine", "Session", "inspector"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))

        # Recreate metadata and tables
        copied_obj.metadata = MetaData()
        copied_obj._session_table = None
        copied_obj._memory_table = None
        copied_obj._metrics_table = None
        copied_obj._knowledge_table = None

        return copied_obj

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.Session.remove()
        except Exception:
            pass
