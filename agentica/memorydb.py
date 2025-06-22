from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Optional, List, Dict, Any
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator, Field
from sqlalchemy.dialects import postgresql
from sqlalchemy import (
    create_engine,
    inspect,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.schema import MetaData, Table, Column
from sqlalchemy.sql.expression import text, select, delete
from sqlalchemy.types import DateTime, String

from agentica.utils.log import logger


class MemoryDb(ABC):
    """Base class for the Memory Database."""

    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def memory_exists(self, memory: MemoryRow) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        raise NotImplementedError

    @abstractmethod
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_table(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def table_exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> bool:
        raise NotImplementedError


class MemoryRow(BaseModel):
    """Memory Row that is stored in the database"""

    memory: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    # id for this memory, auto-generated from the memory
    id: Optional[str] = None

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


class CsvMemoryDb(MemoryDb):
    def __init__(self, csv_file_path: str = None):
        self.file_path = csv_file_path if csv_file_path else f"memory_{datetime.now().isoformat()}.csv"
        dirname = os.path.dirname(self.file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.memories = []

    def create(self) -> None:
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
        self.memories = []

    def memory_exists(self, memory: MemoryRow) -> bool:
        with open(self.file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == memory.id:
                    return True
        return False

    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories = []
        if not os.path.exists(self.file_path):
            return memories
        with open(self.file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if user_id is not None and row[1] != user_id:
                    continue
                memory = MemoryRow(
                    id=row[0],
                    user_id=row[1],
                    memory=json.loads(row[2]),
                    created_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    updated_at=datetime.fromisoformat(row[4]) if row[4] else None
                )
                memories.append(memory)
                if limit is not None and len(memories) >= limit:
                    break
        self.memories = memories
        return memories

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        memories = self.read_memories()
        memory_exists = False
        for i, m in enumerate(memories):
            if m and m.id == memory.id:
                memories[i] = memory
                memory_exists = True
                break
        if not memory_exists:
            memories.append(memory)
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
            for memory in memories:
                writer.writerow([
                    memory.id, memory.user_id, json.dumps(memory.memory, ensure_ascii=False),
                    memory.created_at.isoformat() if memory.created_at else '',
                    memory.updated_at.isoformat() if memory.updated_at else ''
                ])
            logger.debug(f"Memory {memory.id} upserted, memories size: {len(memories)}, saved: {self.file_path}")
        self.memories = memories
        return memory

    def delete_memory(self, id: str) -> None:
        memories = self.read_memories()
        memories = [m for m in memories if m.id != id]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
            for memory in memories:
                writer.writerow([
                    memory.id, memory.user_id, json.dumps(memory.memory, ensure_ascii=False),
                    memory.created_at.isoformat() if memory.created_at else '',
                    memory.updated_at.isoformat() if memory.updated_at else ''
                ])
            logger.debug(f"Memory {id} deleted, memories size: {len(memories)}, saved: {self.file_path}")
        self.memories = memories

    def drop_table(self) -> None:
        os.remove(self.file_path)
        self.memories = []

    def table_exists(self) -> bool:
        return os.path.exists(self.file_path)

    def clear(self) -> bool:
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'user_id', 'memory', 'created_at', 'updated_at'])
        self.memories = []
        return True


class InMemoryDb(MemoryDb):
    def __init__(self):
        self.memories = []

    def create(self) -> None:
        self.memories = []

    def memory_exists(self, memory: MemoryRow) -> bool:
        for m in self.memories:
            if m.id == memory.id:
                return True
        return False

    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        results = []
        for memory in self.memories:
            if user_id and memory.user_id == user_id:
                results.append(memory)

        if sort == "asc":
            results = sorted(results, key=lambda x: x.created_at)
        elif sort == "desc":
            results = sorted(results, key=lambda x: x.created_at, reverse=True)

        if limit:
            results = results[:limit]

        return results

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        if self.memory_exists(memory):
            self.delete_memory(memory.id)
        self.memories.append(memory)
        return memory

    def delete_memory(self, id: str) -> None:
        for i, memory in enumerate(self.memories):
            if memory.id == id:
                del self.memories[i]
                break

    def drop_table(self) -> None:
        self.create()

    def table_exists(self) -> bool:
        return True

    def clear(self) -> bool:
        self.create()
        return True


class PgMemoryDb(MemoryDb):
    def __init__(
            self,
            table_name: str,
            schema: Optional[str] = "ai",
            db_url: Optional[str] = None,
            db_engine: Optional[Engine] = None,
    ):
        """
        This class provides a memory store backed by a postgres table.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url to create the engine

        Args:
            table_name (str): The name of the table to store memory rows.
            schema (Optional[str]): The schema to store the table in. Defaults to "ai".
            db_url (Optional[str]): The database URL to connect to. Defaults to None.
            db_engine (Optional[Engine]): The database engine to use. Defaults to None.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)

        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")

        self.table_name: str = table_name
        self.schema: Optional[str] = schema
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.inspector = inspect(self.db_engine)
        self.metadata: MetaData = MetaData(schema=self.schema)
        self.Session: scoped_session = scoped_session(sessionmaker(bind=self.db_engine))
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String),
            Column("memory", postgresql.JSONB, server_default=text("'{}'::jsonb")),
            Column("created_at", DateTime(timezone=True), server_default=text("now()")),
            Column("updated_at", DateTime(timezone=True), onupdate=text("now()")),
            extend_existing=True,
        )

    def create(self) -> None:
        if not self.table_exists():
            try:
                with self.Session() as sess, sess.begin():
                    if self.schema is not None:
                        logger.debug(f"Creating schema: {self.schema}")
                        sess.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))
                logger.debug(f"Creating table: {self.table_name}")
                self.table.create(self.db_engine, checkfirst=True)
            except Exception as e:
                logger.error(f"Error creating table '{self.table.fullname}': {e}")
                raise

    def memory_exists(self, memory: MemoryRow) -> bool:
        columns = [self.table.c.id]
        with self.Session() as sess, sess.begin():
            stmt = select(*columns).where(self.table.c.id == memory.id)
            result = sess.execute(stmt).first()
            return result is not None

    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories: List[MemoryRow] = []
        try:
            with self.Session() as sess, sess.begin():
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if limit is not None:
                    stmt = stmt.limit(limit)

                if sort == "asc":
                    stmt = stmt.order_by(self.table.c.created_at.asc())
                else:
                    stmt = stmt.order_by(self.table.c.created_at.desc())

                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None:
                        memories.append(MemoryRow.model_validate(row))
        except Exception as e:
            logger.debug(f"Exception reading from table: {e}")
            logger.debug(f"Table does not exist: {self.table.name}")
            logger.debug("Creating table for future transactions")
            self.create()
        return memories

    def upsert_memory(self, memory: MemoryRow, create_and_retry: bool = True) -> None:
        """Create a new memory if it does not exist, otherwise update the existing memory"""

        try:
            with self.Session() as sess, sess.begin():
                # Create an insert statement
                stmt = postgresql.insert(self.table).values(
                    id=memory.id,
                    user_id=memory.user_id,
                    memory=memory.memory,
                )

                # Define the upsert if the memory already exists
                # See: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#postgresql-insert-on-conflict
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_=dict(
                        user_id=stmt.excluded.user_id,
                        memory=stmt.excluded.memory,
                    ),
                )

                sess.execute(stmt)
        except Exception as e:
            logger.debug(f"Exception upserting into table: {e}")
            logger.debug(f"Table does not exist: {self.table.name}")
            logger.debug("Creating table for future transactions")
            self.create()
            if create_and_retry:
                return self.upsert_memory(memory, create_and_retry=False)
            return None

    def delete_memory(self, id: str) -> None:
        with self.Session() as sess, sess.begin():
            stmt = delete(self.table).where(self.table.c.id == id)
            sess.execute(stmt)

    def drop_table(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name, schema=self.schema)
        except Exception as e:
            logger.error(e)
            return False

    def clear(self) -> bool:
        with self.Session() as sess, sess.begin():
            stmt = delete(self.table)
            sess.execute(stmt)
            return True

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the PgMemoryDb instance, handling unpickleable attributes.

        Args:
            memo (dict): A dictionary of objects already copied during the current copying pass.

        Returns:
            PgMemoryDb: A deep-copied instance of PgMemoryDb.
        """
        from copy import deepcopy

        # Create a new instance without calling __init__
        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj

        # Deep copy attributes
        for k, v in self.__dict__.items():
            if k in {"metadata", "table"}:
                continue
            # Reuse db_engine and Session without copying
            elif k in {"db_engine", "Session"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))

        # Recreate metadata and table for the copied instance
        copied_obj.metadata = MetaData(schema=copied_obj.schema)
        copied_obj.table = copied_obj.get_table()

        return copied_obj


class SqliteMemoryDb(MemoryDb):
    def __init__(
            self,
            table_name: str = "memory",
            db_url: Optional[str] = None,
            db_file: Optional[str] = None,
            db_engine: Optional[Engine] = None,
    ):
        """
        This class provides a memory store backed by a SQLite table.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url
            3. Use the db_file
            4. Create a new in-memory database

        Args:
            table_name: The name of the table to store Agent sessions.
            db_url: The database URL to connect to.
            db_file: The database file to connect to.
            db_engine: The database engine to use.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            # Use the db_file to create the engine
            db_path = Path(db_file).resolve()
            # Ensure the directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{db_path}")
        else:
            _engine = create_engine("sqlite://")

        if _engine is None:
            raise ValueError("Must provide either db_url, db_file or db_engine")

        # Database attributes
        self.table_name: str = table_name
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()
        self.inspector = inspect(self.db_engine)

        # Database session
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        # Database table for memories
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String),
            Column("memory", String),
            Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
            Column(
                "updated_at", DateTime, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
            ),
            extend_existing=True,
        )

    def create(self) -> None:
        if not self.table_exists():
            try:
                logger.debug(f"Creating table: {self.table_name}")
                self.table.create(self.db_engine, checkfirst=True)
            except Exception as e:
                logger.error(f"Error creating table '{self.table_name}': {e}")
                raise

    def memory_exists(self, memory: MemoryRow) -> bool:
        with self.Session() as session:
            stmt = select(self.table.c.id).where(self.table.c.id == memory.id)
            result = session.execute(stmt).first()
            return result is not None

    def read_memories(
            self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories: List[MemoryRow] = []
        try:
            with self.Session() as session:
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)

                if sort == "asc":
                    stmt = stmt.order_by(self.table.c.created_at.asc())
                else:
                    stmt = stmt.order_by(self.table.c.created_at.desc())

                if limit is not None:
                    stmt = stmt.limit(limit)

                result = session.execute(stmt)
                for row in result:
                    memories.append(MemoryRow(id=row.id, user_id=row.user_id, memory=eval(row.memory)))
        except SQLAlchemyError as e:
            logger.debug(f"Exception reading from table: {e}")
            logger.debug(f"Table does not exist: {self.table_name}")
            logger.debug("Creating table for future transactions")
            self.create()
        return memories

    def upsert_memory(self, memory: MemoryRow, create_and_retry: bool = True) -> None:
        try:
            with self.Session() as session:
                # Check if the memory already exists
                existing = session.execute(select(self.table).where(self.table.c.id == memory.id)).first()

                if existing:
                    # Update existing memory
                    stmt = (
                        self.table.update()
                        .where(self.table.c.id == memory.id)
                        .values(user_id=memory.user_id, memory=str(memory.memory), updated_at=text("CURRENT_TIMESTAMP"))
                    )
                else:
                    # Insert new memory
                    stmt = self.table.insert().values(id=memory.id, user_id=memory.user_id,
                                                      memory=str(memory.memory))  # type: ignore

                session.execute(stmt)
                session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Exception upserting into table: {e}")
            if not self.table_exists():
                logger.info(f"Table does not exist: {self.table_name}")
                logger.info("Creating table for future transactions")
                self.create()
                if create_and_retry:
                    return self.upsert_memory(memory, create_and_retry=False)
            else:
                raise

    def delete_memory(self, id: str) -> None:
        with self.Session() as session:
            stmt = delete(self.table).where(self.table.c.id == id)
            session.execute(stmt)
            session.commit()

    def drop_table(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return self.inspector.has_table(self.table.name)
        except Exception as e:
            logger.error(e)
            return False

    def clear(self) -> bool:
        with self.Session() as session:
            stmt = delete(self.table)
            session.execute(stmt)
            session.commit()
        return True

    def __del__(self):
        self.Session.remove()
