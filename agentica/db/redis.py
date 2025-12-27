# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Redis implementation of BaseDb
"""
import json
import time
from typing import Optional, List
from datetime import datetime

from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow, KnowledgeRow
from agentica.utils.log import logger

try:
    import redis
except ImportError:
    redis = None


class RedisDb(BaseDb):
    """
    Redis implementation of unified database storage.

    Uses Redis hash and sorted sets for efficient storage and retrieval.

    Example:
        >>> db = RedisDb(
        ...     host="localhost",
        ...     port=6379,
        ...     password=None,
        ...     db=0
        ... )
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
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        redis_url: Optional[str] = None,
        redis_client: Optional["redis.Redis"] = None,
        key_prefix: str = "agentica",
        session_table: Optional[str] = None,
        memory_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        knowledge_table: Optional[str] = None,
    ):
        """
        Initialize RedisDb.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            redis_url: Redis URL (overrides host/port/password/db)
            redis_client: Existing Redis client
            key_prefix: Prefix for all Redis keys
            session_table: Name suffix for sessions keys
            memory_table: Name suffix for memories keys
            metrics_table: Name suffix for metrics keys
            knowledge_table: Name suffix for knowledge keys
        """
        if redis is None:
            raise ImportError("redis package is required. Install with: pip install redis")

        super().__init__(session_table, memory_table, metrics_table, knowledge_table)

        # Create Redis client
        if redis_client is not None:
            self.client = redis_client
        elif redis_url is not None:
            self.client = redis.from_url(redis_url, decode_responses=True)
        else:
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,
            )

        self.host = host
        self.port = port
        self.db = db
        self.key_prefix = key_prefix

        # Key patterns
        self._session_key_prefix = f"{key_prefix}:{self.session_table_name}"
        self._memory_key_prefix = f"{key_prefix}:{self.memory_table_name}"
        self._metrics_key_prefix = f"{key_prefix}:{self.metrics_table_name}"
        self._knowledge_key_prefix = f"{key_prefix}:{self.knowledge_table_name or 'knowledge'}"

        # Index keys (sorted sets for ordering)
        self._session_index = f"{self._session_key_prefix}:index"
        self._memory_index = f"{self._memory_key_prefix}:index"
        self._metrics_index = f"{self._metrics_key_prefix}:index"
        self._knowledge_index = f"{self._knowledge_key_prefix}:index"

    def _session_key(self, session_id: str) -> str:
        return f"{self._session_key_prefix}:{session_id}"

    def _memory_key(self, memory_id: str) -> str:
        return f"{self._memory_key_prefix}:{memory_id}"

    def _metrics_key(self, metrics_id: str) -> str:
        return f"{self._metrics_key_prefix}:{metrics_id}"

    def _knowledge_key(self, knowledge_id: str) -> str:
        return f"{self._knowledge_key_prefix}:{knowledge_id}"

    def _user_memory_index(self, user_id: str) -> str:
        return f"{self._memory_key_prefix}:user:{user_id}"

    # ==================== Session Operations ====================

    def create_session_table(self) -> None:
        """No-op for Redis - tables are created on demand"""
        pass

    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]:
        try:
            key = self._session_key(session_id)
            data = self.client.hgetall(key)
            if not data:
                return None

            # Check user_id if provided
            if user_id and data.get("user_id") != user_id:
                return None

            return SessionRow(
                session_id=data.get("session_id"),
                agent_id=data.get("agent_id"),
                user_id=data.get("user_id"),
                memory=json.loads(data["memory"]) if data.get("memory") else None,
                agent_data=json.loads(data["agent_data"]) if data.get("agent_data") else None,
                user_data=json.loads(data["user_data"]) if data.get("user_data") else None,
                session_data=json.loads(data["session_data"]) if data.get("session_data") else None,
                created_at=int(data["created_at"]) if data.get("created_at") else None,
                updated_at=int(data["updated_at"]) if data.get("updated_at") else None,
            )
        except Exception as e:
            logger.debug(f"Error reading session: {e}")
        return None

    def upsert_session(self, session_row: SessionRow) -> Optional[SessionRow]:
        try:
            key = self._session_key(session_row.session_id)
            now = int(time.time())

            # Get existing created_at or use current time
            existing = self.client.hget(key, "created_at")
            created_at = int(existing) if existing else (session_row.created_at or now)

            data = {
                "session_id": session_row.session_id,
                "agent_id": session_row.agent_id or "",
                "user_id": session_row.user_id or "",
                "memory": json.dumps(session_row.memory, ensure_ascii=False) if session_row.memory else "",
                "agent_data": json.dumps(session_row.agent_data, ensure_ascii=False) if session_row.agent_data else "",
                "user_data": json.dumps(session_row.user_data, ensure_ascii=False) if session_row.user_data else "",
                "session_data": json.dumps(session_row.session_data, ensure_ascii=False) if session_row.session_data else "",
                "created_at": str(created_at),
                "updated_at": str(now),
            }

            self.client.hset(key, mapping=data)
            # Add to index sorted set (score = created_at for ordering)
            self.client.zadd(self._session_index, {session_row.session_id: created_at})

            logger.debug(f"Session upserted: {session_row.session_id}")
            return session_row
        except Exception as e:
            logger.error(f"Error upserting session: {e}")
        return None

    def delete_session(self, session_id: str) -> None:
        try:
            key = self._session_key(session_id)
            self.client.delete(key)
            self.client.zrem(self._session_index, session_id)
            logger.debug(f"Session deleted: {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")

    def get_all_session_ids(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[str]:
        try:
            # Get all session IDs from index (ordered by created_at desc)
            session_ids = self.client.zrevrange(self._session_index, 0, -1)

            if not user_id and not agent_id:
                return session_ids

            # Filter by user_id and/or agent_id
            result = []
            for sid in session_ids:
                key = self._session_key(sid)
                data = self.client.hmget(key, "user_id", "agent_id")
                if user_id and data[0] != user_id:
                    continue
                if agent_id and data[1] != agent_id:
                    continue
                result.append(sid)
            return result
        except Exception as e:
            logger.debug(f"Error getting session IDs: {e}")
        return []

    def get_all_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[SessionRow]:
        session_ids = self.get_all_session_ids(user_id, agent_id)
        sessions = []
        for sid in session_ids:
            session = self.read_session(sid)
            if session:
                sessions.append(session)
        return sessions

    # ==================== Memory Operations ====================

    def create_memory_table(self) -> None:
        """No-op for Redis - tables are created on demand"""
        pass

    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None
    ) -> List[MemoryRow]:
        try:
            # Get memory IDs from index
            if user_id:
                index_key = self._user_memory_index(user_id)
            else:
                index_key = self._memory_index

            # Get IDs ordered by created_at
            if sort == "asc":
                memory_ids = self.client.zrange(index_key, 0, -1)
            else:
                memory_ids = self.client.zrevrange(index_key, 0, -1)

            if limit:
                memory_ids = memory_ids[:limit]

            memories = []
            for mid in memory_ids:
                key = self._memory_key(mid)
                data = self.client.hgetall(key)
                if data:
                    memories.append(MemoryRow(
                        id=data.get("id"),
                        user_id=data.get("user_id"),
                        memory=json.loads(data["memory"]) if data.get("memory") else {},
                        created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
                        updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
                    ))
            return memories
        except Exception as e:
            logger.debug(f"Error reading memories: {e}")
        return []

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        try:
            key = self._memory_key(memory.id)
            now = datetime.now()

            # Get existing created_at or use current time
            existing = self.client.hget(key, "created_at")
            created_at = existing if existing else (memory.created_at.isoformat() if memory.created_at else now.isoformat())

            data = {
                "id": memory.id,
                "user_id": memory.user_id or "",
                "memory": json.dumps(memory.memory, ensure_ascii=False),
                "created_at": created_at,
                "updated_at": now.isoformat(),
            }

            self.client.hset(key, mapping=data)

            # Add to global index
            score = datetime.fromisoformat(created_at).timestamp()
            self.client.zadd(self._memory_index, {memory.id: score})

            # Add to user-specific index
            if memory.user_id:
                self.client.zadd(self._user_memory_index(memory.user_id), {memory.id: score})

            logger.debug(f"Memory upserted: {memory.id}")
            return memory
        except Exception as e:
            logger.error(f"Error upserting memory: {e}")
        return None

    def delete_memory(self, memory_id: str) -> None:
        try:
            key = self._memory_key(memory_id)
            # Get user_id before deleting
            user_id = self.client.hget(key, "user_id")

            self.client.delete(key)
            self.client.zrem(self._memory_index, memory_id)
            if user_id:
                self.client.zrem(self._user_memory_index(user_id), memory_id)

            logger.debug(f"Memory deleted: {memory_id}")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")

    def memory_exists(self, memory: MemoryRow) -> bool:
        try:
            key = self._memory_key(memory.id)
            return self.client.exists(key) > 0
        except Exception as e:
            logger.debug(f"Error checking memory existence: {e}")
        return False

    def clear_memories(self, user_id: Optional[str] = None) -> bool:
        try:
            if user_id:
                # Clear memories for specific user
                index_key = self._user_memory_index(user_id)
                memory_ids = self.client.zrange(index_key, 0, -1)
                for mid in memory_ids:
                    self.client.delete(self._memory_key(mid))
                    self.client.zrem(self._memory_index, mid)
                self.client.delete(index_key)
            else:
                # Clear all memories
                memory_ids = self.client.zrange(self._memory_index, 0, -1)
                for mid in memory_ids:
                    key = self._memory_key(mid)
                    user_id_val = self.client.hget(key, "user_id")
                    self.client.delete(key)
                    if user_id_val:
                        self.client.zrem(self._user_memory_index(user_id_val), mid)
                self.client.delete(self._memory_index)

            logger.debug(f"Memories cleared for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False

    # ==================== Metrics Operations ====================

    def create_metrics_table(self) -> None:
        """No-op for Redis - tables are created on demand"""
        pass

    def insert_metrics(self, metrics: MetricsRow) -> None:
        try:
            key = self._metrics_key(metrics.id)
            now = datetime.now()

            data = {
                "id": metrics.id,
                "agent_id": metrics.agent_id or "",
                "session_id": metrics.session_id or "",
                "run_id": metrics.run_id or "",
                "metrics": json.dumps(metrics.metrics, ensure_ascii=False),
                "created_at": now.isoformat(),
            }

            self.client.hset(key, mapping=data)
            self.client.zadd(self._metrics_index, {metrics.id: now.timestamp()})

            logger.debug(f"Metrics inserted: {metrics.id}")
        except Exception as e:
            logger.error(f"Error inserting metrics: {e}")

    def get_metrics(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MetricsRow]:
        try:
            # Get all metrics IDs from index (ordered by created_at desc)
            metrics_ids = self.client.zrevrange(self._metrics_index, 0, -1)

            result = []
            for mid in metrics_ids:
                key = self._metrics_key(mid)
                data = self.client.hgetall(key)
                if not data:
                    continue

                # Filter by agent_id and/or session_id
                if agent_id and data.get("agent_id") != agent_id:
                    continue
                if session_id and data.get("session_id") != session_id:
                    continue

                result.append(MetricsRow(
                    id=data.get("id"),
                    agent_id=data.get("agent_id") or None,
                    session_id=data.get("session_id") or None,
                    run_id=data.get("run_id") or None,
                    metrics=json.loads(data["metrics"]) if data.get("metrics") else {},
                    created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
                ))

                if limit and len(result) >= limit:
                    break

            return result
        except Exception as e:
            logger.debug(f"Error getting metrics: {e}")
        return []

    # ==================== Knowledge Operations ====================

    def create_knowledge_table(self) -> None:
        """No-op for Redis - tables are created on demand"""
        pass

    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            logger.warning("Knowledge table not configured")
            return None

        try:
            key = self._knowledge_key(knowledge.id)
            now = int(time.time())

            # Get existing created_at or use current time
            existing = self.client.hget(key, "created_at")
            created_at = int(existing) if existing else (knowledge.created_at or now)

            data = {
                "id": knowledge.id,
                "name": knowledge.name,
                "description": knowledge.description or "",
                "content": knowledge.content or "",
                "metadata": json.dumps(knowledge.metadata, ensure_ascii=False) if knowledge.metadata else "",
                "doc_type": knowledge.doc_type or "",
                "size": str(knowledge.size) if knowledge.size else "0",
                "status": knowledge.status or "",
                "status_message": knowledge.status_message or "",
                "created_at": str(created_at),
                "updated_at": str(now),
            }

            self.client.hset(key, mapping=data)
            self.client.zadd(self._knowledge_index, {knowledge.id: created_at})

            logger.debug(f"Knowledge upserted: {knowledge.id}")
            return knowledge
        except Exception as e:
            logger.error(f"Error upserting knowledge: {e}")
        return None

    def read_knowledge(self, knowledge_id: str) -> Optional[KnowledgeRow]:
        if not self.knowledge_table_name:
            return None

        try:
            key = self._knowledge_key(knowledge_id)
            data = self.client.hgetall(key)
            if not data:
                return None

            return KnowledgeRow(
                id=data.get("id"),
                name=data.get("name", ""),
                description=data.get("description", ""),
                content=data.get("content") or None,
                metadata=json.loads(data["metadata"]) if data.get("metadata") else None,
                doc_type=data.get("doc_type") or None,
                size=int(data["size"]) if data.get("size") else None,
                status=data.get("status") or None,
                status_message=data.get("status_message") or None,
                created_at=int(data["created_at"]) if data.get("created_at") else None,
                updated_at=int(data["updated_at"]) if data.get("updated_at") else None,
            )
        except Exception as e:
            logger.debug(f"Error reading knowledge: {e}")
        return None

    def get_all_knowledge(
        self,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[KnowledgeRow]:
        if not self.knowledge_table_name:
            return []

        try:
            # Get all knowledge IDs from index (ordered by created_at desc)
            knowledge_ids = self.client.zrevrange(self._knowledge_index, 0, -1)

            result = []
            for kid in knowledge_ids:
                knowledge = self.read_knowledge(kid)
                if not knowledge:
                    continue

                # Filter by doc_type and/or status
                if doc_type and knowledge.doc_type != doc_type:
                    continue
                if status and knowledge.status != status:
                    continue

                result.append(knowledge)

                if limit and len(result) >= limit:
                    break

            return result
        except Exception as e:
            logger.debug(f"Error getting knowledge: {e}")
        return []

    def delete_knowledge(self, knowledge_id: str) -> None:
        if not self.knowledge_table_name:
            return

        try:
            key = self._knowledge_key(knowledge_id)
            self.client.delete(key)
            self.client.zrem(self._knowledge_index, knowledge_id)
            logger.debug(f"Knowledge deleted: {knowledge_id}")
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")

    def clear_knowledge(self) -> bool:
        if not self.knowledge_table_name:
            return False

        try:
            knowledge_ids = self.client.zrange(self._knowledge_index, 0, -1)
            for kid in knowledge_ids:
                self.client.delete(self._knowledge_key(kid))
            self.client.delete(self._knowledge_index)
            logger.debug("All knowledge cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge: {e}")
            return False

    # ==================== Lifecycle ====================

    def create(self) -> None:
        """No-op for Redis - data structures are created on demand"""
        pass

    def drop(self) -> None:
        """Drop all data with the key prefix"""
        try:
            # Find and delete all keys with the prefix
            pattern = f"{self.key_prefix}:*"
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
            logger.debug(f"All keys with prefix '{self.key_prefix}' dropped")
        except Exception as e:
            logger.error(f"Error dropping data: {e}")

    def upgrade_schema(self) -> None:
        """No-op for Redis - schema-less"""
        pass

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.client.close()
        except Exception:
            pass
