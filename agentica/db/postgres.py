# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: PostgreSQL implementation of BaseDb
"""
import time
from typing import Optional, List
from copy import deepcopy

from sqlalchemy import create_engine, Column, String, BigInteger, DateTime, Text, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.schema import MetaData, Table
from sqlalchemy.sql.expression import select, delete
from sqlalchemy.inspection import inspect

from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow, KnowledgeRow
from agentica.utils.log import logger


class PostgresDb(BaseDb):
    """
    PostgreSQL implementation of unified database storage.

    Tables are auto-created on first use, no need to call db.create() explicitly.

    Example:
        >>> db = PostgresDb(db_url="postgresql://user:pass@localhost:5432/mydb")
        >>> 
        >>> # Store session
        >>> from agentica.db.base import SessionRow
        >>> session = SessionRow(session_id="123", agent_id="agent1")
        >>> db.upsert_session(session)
        >>>
        >>> # For RAG knowledge storage
        >>> contents_db = PostgresDb(db_url="...", knowledge_table="knowledge_contents")
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        schema: Optional[str] = "ai",
        session_table: Optional[str] = None,
        memory_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
        schema_version: int = 1,
        auto_upgrade_schema: bool = False,
    ):
        """
        Initialize PostgresDb.

        Args:
            db_url: PostgreSQL connection URL
            db_engine: Existing SQLAlchemy engine
            schema: Database schema name (default: "ai")
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

        if _engine is None:
            raise ValueError("Must provide db_url or db_engine")

        self.db_url = db_url
        self.schema = schema
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)
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

        logger.debug(f"PostgresDb initialized with schema: {self.schema}")
        
        # Auto-create tables
        self.create()

    def _create_schema(self) -> None:
        """Create schema if not exists"""
        if self.schema is not None:
            try:
                with self.Session() as sess, sess.begin():
                    sess.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))
                    logger.debug(f"Schema created: {self.schema}")
            except Exception as e:
                logger.debug(f"Schema may already exist: {e}")

    # ==================== Session Operations ====================

    def _get_session_table(self) -> Table:
        if self._session_table is None:
            self._session_table = Table(
                self.session_table_name,
                self.metadata,
                Column("session_id", String, primary_key=True),
                Column("agent_id", String),
                Column("user_id", String),
                Column("memory", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("agent_data", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("user_data", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("session_data", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("created_at", BigInteger),
                Column("updated_at", BigInteger),
                extend_existing=True,
            )
        return self._session_table

    def create_session_table(self) -> None:
        self._create_schema()
        table = self._get_session_table()
        if not self.inspector.has_table(table.name, schema=self.schema):
            logger.debug(f"Creating session table: {self.schema}.{table.name}")
            table.create(self.db_engine, checkfirst=True)

    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]:
        table = self._get_session_table()
        try:
            with self.Session() as session, session.begin():
                stmt = select(table).where(table.c.session_id == session_id)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                row = session.execute(stmt).first()
                if row:
                    return SessionRow(
                        session_id=row.session_id,
                        agent_id=row.agent_id,
                        user_id=row.user_id,
                        memory=row.memory,
                        agent_data=row.agent_data,
                        user_data=row.user_data,
                        session_data=row.session_data,
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
            with self.Session() as session, session.begin():
                stmt = postgresql.insert(table).values(
                    session_id=session_row.session_id,
                    agent_id=session_row.agent_id,
                    user_id=session_row.user_id,
                    memory=session_row.memory,
                    agent_data=session_row.agent_data,
                    user_data=session_row.user_data,
                    session_data=session_row.session_data,
                    created_at=session_row.created_at or now,
                    updated_at=now,
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["session_id"],
                    set_=dict(
                        agent_id=stmt.excluded.agent_id,
                        user_id=stmt.excluded.user_id,
                        memory=stmt.excluded.memory,
                        agent_data=stmt.excluded.agent_data,
                        user_data=stmt.excluded.user_data,
                        session_data=stmt.excluded.session_data,
                        updated_at=now,
                    ),
                )

                session.execute(stmt)
                logger.debug(f"Session upserted: {session_row.session_id}")
        except Exception as e:
            logger.debug(f"Error upserting session: {e}")
            self.create_session_table()
            return self.upsert_session(session_row)

        return session_row

    def delete_session(self, session_id: str) -> None:
        table = self._get_session_table()
        try:
            with self.Session() as session, session.begin():
                stmt = delete(table).where(table.c.session_id == session_id)
                session.execute(stmt)
                logger.debug(f"Session deleted: {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")

    def get_all_session_ids(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[str]:
        table = self._get_session_table()
        try:
            with self.Session() as session, session.begin():
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
            with self.Session() as session, session.begin():
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
                        memory=row.memory,
                        agent_data=row.agent_data,
                        user_data=row.user_data,
                        session_data=row.session_data,
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
                Column("memory", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("created_at", DateTime(timezone=True), server_default=text("now()")),
                Column("updated_at", DateTime(timezone=True), onupdate=text("now()")),
                extend_existing=True,
            )
        return self._memory_table

    def create_memory_table(self) -> None:
        self._create_schema()
        table = self._get_memory_table()
        if not self.inspector.has_table(table.name, schema=self.schema):
            logger.debug(f"Creating memory table: {self.schema}.{table.name}")
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
            with self.Session() as session, session.begin():
                stmt = select(table)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                if sort == "asc":
                    stmt = stmt.order_by(table.c.created_at.asc())
                else:
                    stmt = stmt.order_by(table.c.created_at.desc())
                if limit:
                    stmt = stmt.limit(limit)

                rows = session.execute(stmt).fetchall()
                for row in rows:
                    memories.append(MemoryRow(
                        id=row.id,
                        user_id=row.user_id,
                        memory=row.memory or {},
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    ))
        except Exception as e:
            logger.debug(f"Error reading memories: {e}")
            self.create_memory_table()
        return memories

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        table = self._get_memory_table()

        try:
            with self.Session() as session, session.begin():
                stmt = postgresql.insert(table).values(
                    id=memory.id,
                    user_id=memory.user_id,
                    memory=memory.memory,
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=dict(
                        user_id=stmt.excluded.user_id,
                        memory=stmt.excluded.memory,
                    ),
                )

                session.execute(stmt)
                logger.debug(f"Memory upserted: {memory.id}")
        except Exception as e:
            logger.debug(f"Error upserting memory: {e}")
            self.create_memory_table()
            return self.upsert_memory(memory)

        return memory

    def delete_memory(self, memory_id: str) -> None:
        table = self._get_memory_table()
        try:
            with self.Session() as session, session.begin():
                stmt = delete(table).where(table.c.id == memory_id)
                session.execute(stmt)
                logger.debug(f"Memory deleted: {memory_id}")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")

    def memory_exists(self, memory: MemoryRow) -> bool:
        table = self._get_memory_table()
        try:
            with self.Session() as session, session.begin():
                stmt = select(table.c.id).where(table.c.id == memory.id)
                return session.execute(stmt).first() is not None
        except Exception as e:
            logger.debug(f"Error checking memory existence: {e}")
        return False

    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        table = self._get_memory_table()
        try:
            with self.Session() as session, session.begin():
                stmt = delete(table)
                if user_id:
                    stmt = stmt.where(table.c.user_id == user_id)
                session.execute(stmt)
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
                Column("metrics", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("created_at", DateTime(timezone=True), server_default=text("now()")),
                extend_existing=True,
            )
        return self._metrics_table

    def create_metrics_table(self) -> None:
        self._create_schema()
        table = self._get_metrics_table()
        if not self.inspector.has_table(table.name, schema=self.schema):
            logger.debug(f"Creating metrics table: {self.schema}.{table.name}")
            table.create(self.db_engine, checkfirst=True)

    def insert_metrics(self, metrics: MetricsRow) -> None:
        table = self._get_metrics_table()
        try:
            with self.Session() as session, session.begin():
                stmt = table.insert().values(
                    id=metrics.id,
                    agent_id=metrics.agent_id,
                    session_id=metrics.session_id,
                    run_id=metrics.run_id,
                    metrics=metrics.metrics,
                )
                session.execute(stmt)
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
            with self.Session() as session, session.begin():
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
                        metrics=row.metrics or {},
                        created_at=row.created_at,
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
                Column("content", Text),
                Column("metadata", postgresql.JSONB, server_default=text("'{}'::jsonb")),
                Column("doc_type", String),
                Column("size", BigInteger),
                Column("status", String),
                Column("status_message", Text),
                Column("created_at", BigInteger),
                Column("updated_at", BigInteger),
                extend_existing=True,
            )
        return self._knowledge_table

    def create_knowledge_table(self) -> None:
        if not self.knowledge_table_name:
            return
        self._create_schema()
        table = self._get_knowledge_table()
        if not self.inspector.has_table(table.name, schema=self.schema):
            logger.debug(f"Creating knowledge table: {self.schema}.{table.name}")
            table.create(self.db_engine, checkfirst=True)

    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            logger.warning("Knowledge table not configured")
            return None
        table = self._get_knowledge_table()
        now = int(time.time())

        try:
            with self.Session() as session, session.begin():
                stmt = postgresql.insert(table).values(
                    id=knowledge.id,
                    name=knowledge.name,
                    description=knowledge.description,
                    content=knowledge.content,
                    metadata=knowledge.metadata,
                    doc_type=knowledge.doc_type,
                    size=knowledge.size,
                    status=knowledge.status,
                    status_message=knowledge.status_message,
                    created_at=knowledge.created_at or now,
                    updated_at=now,
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=dict(
                        name=stmt.excluded.name,
                        description=stmt.excluded.description,
                        content=stmt.excluded.content,
                        metadata=stmt.excluded.metadata,
                        doc_type=stmt.excluded.doc_type,
                        size=stmt.excluded.size,
                        status=stmt.excluded.status,
                        status_message=stmt.excluded.status_message,
                        updated_at=now,
                    ),
                )

                session.execute(stmt)
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
            with self.Session() as session, session.begin():
                stmt = select(table).where(table.c.id == knowledge_id)
                row = session.execute(stmt).first()
                if row:
                    return KnowledgeRow(
                        id=row.id,
                        name=row.name,
                        description=row.description or "",
                        content=row.content,
                        metadata=row.metadata,
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
            with self.Session() as session, session.begin():
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
                        metadata=row.metadata,
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
            with self.Session() as session, session.begin():
                stmt = delete(table).where(table.c.id == knowledge_id)
                session.execute(stmt)
                logger.debug(f"Knowledge deleted: {knowledge_id}")
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")

    def clear_knowledge(self) -> bool:
        if not self.knowledge_table_name:
            return False
        table = self._get_knowledge_table()
        try:
            with self.Session() as session, session.begin():
                stmt = delete(table)
                session.execute(stmt)
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
                    if self.inspector.has_table(table.name, schema=self.schema):
                        table.drop(self.db_engine)
                        logger.debug(f"Table dropped: {self.schema}.{table.name}")
                except Exception as e:
                    logger.error(f"Error dropping table {table.name}: {e}")

    def upgrade_schema(self) -> None:
        """Upgrade database schema if needed"""
        # TODO: Implement schema migration logic
        pass

    def __deepcopy__(self, memo):
        """Create a deep copy of the PostgresDb instance"""
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
        copied_obj.metadata = MetaData(schema=copied_obj.schema)
        copied_obj._session_table = None
        copied_obj._memory_table = None
        copied_obj._metrics_table = None
        copied_obj._knowledge_table = None

        return copied_obj
