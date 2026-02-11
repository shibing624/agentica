# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Long-term memory demo with multiple database backends

DEPRECATED: This approach is deprecated for new projects.
For persistent user memories, use Workspace instead:

    from agentica import Agent
    from agentica.workspace import Workspace

    workspace = Workspace("~/.agentica/workspace")
    workspace.save_memory("User prefers concise responses")

    agent = Agent(workspace=workspace, model=...)

This example is preserved for legacy support and multi-user web applications
where database-based storage is required.

Database backends supported:
1. SqliteDb - SQLite file-based storage
2. MysqlDb - MySQL database storage
3. RedisDb - Redis key-value storage
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import (
    Agent, OpenAIChat, AgentMemory,
    MemoryClassifier, MemoryRetrieval
)
from agentica.db.sqlite import SqliteDb
from agentica.db.base import MemoryRow
from agentica.memory import Memory


def demo_sqlite_memory():
    """Demo with SqliteDb for long-term memory storage."""
    print("=" * 50)
    print("Demo 1: SqliteDb for long-term memory")
    print("=" * 50)

    db_file = "outputs/demo_memory.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    db = SqliteDb(db_file=db_file)
    memory = AgentMemory(
        db=db,
        user_id="test_user",
        num_memories=5,
        retrieval=MemoryRetrieval.last_n,
        create_user_memories=True,
        update_user_memories_after_run=True,
        classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
    )

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        memory=memory,
        # debug_mode=True,
    )

    print("\n--- First conversation ---")
    agent.print_response_sync("My name is Alice and I work as a software engineer at Google.")

    print("\n--- Loading memories ---")
    memory.load_user_memories()
    if memory.memories:
        print(f"Found {len(memory.memories)} memories")
        for m in memory.memories:
            print(f"  - {m.memory}")
    else:
        print("No memories found")


def demo_mysql_memory():
    """Demo with MysqlDb for long-term memory storage."""
    print("\n" + "=" * 50)
    print("Demo 2: MysqlDb for long-term memory")
    print("=" * 50)

    try:
        from agentica.db.mysql import MysqlDb
    except ImportError:
        print("MySQL support requires pymysql. Install with: pip install pymysql")
        return

    # MySQL configuration from environment or defaults
    mysql_host = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user = os.getenv("MYSQL_USER", "test_user")
    mysql_password = os.getenv("MYSQL_PASSWORD", "test_password.123")
    mysql_database = os.getenv("MYSQL_DATABASE", "agentica")

    try:
        db = MysqlDb(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )

        memory = AgentMemory(
            db=db,
            user_id="mysql_test_user",
            num_memories=5,
            retrieval=MemoryRetrieval.last_n,
            create_user_memories=True,
            update_user_memories_after_run=True,
            classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
        )

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            memory=memory,
        )

        print("\n--- First conversation (MySQL) ---")
        agent.print_response_sync("My name is Bob and I'm a data scientist at Microsoft.")

        print("\n--- Loading memories from MySQL ---")
        memory.load_user_memories()
        if memory.memories:
            print(f"Found {len(memory.memories)} memories")
            for m in memory.memories:
                print(f"  - {m.memory}")
        else:
            print("No memories found")

    except Exception as e:
        print(f"MySQL connection failed: {e}")
        print("Make sure MySQL is running and credentials are correct.")


def demo_redis_memory():
    """Demo with RedisDb for long-term memory storage."""
    print("\n" + "=" * 50)
    print("Demo 3: RedisDb for long-term memory")
    print("=" * 50)

    try:
        from agentica.db.redis import RedisDb
    except ImportError:
        print("Redis support requires redis package. Install with: pip install redis")
        return

    # Redis configuration from environment or defaults
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    redis_db = int(os.getenv("REDIS_DB", "0"))

    try:
        db = RedisDb(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
        )

        memory = AgentMemory(
            db=db,
            user_id="redis_test_user",
            num_memories=5,
            retrieval=MemoryRetrieval.last_n,
            create_user_memories=True,
            update_user_memories_after_run=True,
            classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
        )

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            memory=memory,
        )

        print("\n--- First conversation (Redis) ---")
        agent.print_response_sync("My name is Charlie and I'm a ML engineer at Meta.")

        print("\n--- Loading memories from Redis ---")
        memory.load_user_memories()
        if memory.memories:
            print(f"Found {len(memory.memories)} memories")
            for m in memory.memories:
                print(f"  - {m.memory}")
        else:
            print("No memories found")

    except Exception as e:
        print(f"Redis connection failed: {e}")
        print("Make sure Redis is running and credentials are correct.")


def demo_direct_db_operations():
    """Demo using database directly for memory operations."""
    print("\n" + "=" * 50)
    print("Demo 4: Direct database operations")
    print("=" * 50)

    db_file = "outputs/direct_demo.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    db = SqliteDb(db_file=db_file)

    memories_to_insert = [
        "The capital of France is Paris.",
        "Python was created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence.",
        "Tokyo is the capital of Japan.",
    ]

    for content in memories_to_insert:
        memory_data = Memory(memory=content, input_text=content).to_dict()
        row = MemoryRow(user_id="demo_user", memory=memory_data)
        db.upsert_memory(row)

    print(f"Inserted {len(memories_to_insert)} memories")

    print("\n--- Reading memories ---")
    results = db.read_memories(user_id="demo_user", limit=3)
    for row in results:
        print(f"  - {row.memory.get('memory', '')}")


def demo_agent_with_session():
    """Demo using Agent with db for session persistence."""
    print("\n" + "=" * 50)
    print("Demo 5: Agent with session persistence")
    print("=" * 50)

    db_file = "outputs/agent_session.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    db = SqliteDb(db_file=db_file)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        db=db,
        memory=AgentMemory(db=db, create_user_memories=True),
        add_history_to_messages=True,
        # debug_mode=True,
    )

    session_id = agent.load_session()
    print(f"Session ID: {session_id}")

    print("\n--- Conversation ---")
    agent.print_response_sync("Hi, I'm David. I love programming in Python.")
    agent.print_response_sync("What's my name and what do I like?")

    print("\n--- Stored sessions ---")
    session_ids = db.get_all_session_ids()
    print(f"Found {len(session_ids)} sessions: {session_ids}")


if __name__ == "__main__":
    # Run SQLite demo (always works)
    demo_sqlite_memory()

    # Run MySQL demo (requires MySQL server)
    demo_mysql_memory()

    # Run Redis demo (requires Redis server)
    demo_redis_memory()

    # Run direct database operations demo
    demo_direct_db_operations()

    # Run session persistence demo
    demo_agent_with_session()
