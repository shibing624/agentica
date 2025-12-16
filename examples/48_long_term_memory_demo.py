# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Long-term memory demo with SqliteDb
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    # Create SqliteDb
    db_file = "outputs/demo_memory.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    
    db = SqliteDb(db_file=db_file)
    # Initialize memory with SqliteDb
    memory = AgentMemory(
        db=db,
        user_id="test_user",
        num_memories=5,
        retrieval=MemoryRetrieval.last_n,
        create_user_memories=True,
        update_user_memories_after_run=True,
        classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
    )
    
    # Create agent with memory
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        memory=memory,
        debug_mode=True,
    )
    
    # First conversation
    print("\n--- First conversation ---")
    agent.print_response("My name is Alice and I work as a software engineer at Google.")
    
    # Load memories
    print("\n--- Loading memories ---")
    memory.load_user_memories()
    if memory.memories:
        print(f"Found {len(memory.memories)} memories")
        for m in memory.memories:
            print(f"  - {m.memory}")
    else:
        print("No memories found")


def demo_direct_sqlite():
    """Demo using SqliteDb directly for memory operations."""
    print("\n" + "=" * 50)
    print("Demo 2: Using SqliteDb directly")
    print("=" * 50)
    
    # Create SqliteDb directly
    db_file = "outputs/direct_demo.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    
    db = SqliteDb(db_file=db_file)
    
    # Insert memories
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
    
    # Read memories
    print("\n--- Reading memories ---")
    results = db.read_memories(user_id="demo_user", limit=3)
    for row in results:
        print(f"  - {row.memory.get('memory', '')}")


def demo_agent_with_session():
    """Demo using Agent with db for session persistence."""
    print("\n" + "=" * 50)
    print("Demo 3: Agent with session persistence")
    print("=" * 50)
    
    db_file = "outputs/agent_session.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    
    db = SqliteDb(db_file=db_file)
    
    # Create agent with db for session storage
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        db=db,
        memory=AgentMemory(db=db, create_user_memories=True),
        add_history_to_messages=True,
        debug_mode=True,
    )
    
    # Load or create session
    session_id = agent.load_session()
    print(f"Session ID: {session_id}")
    
    # Have a conversation
    print("\n--- Conversation ---")
    agent.print_response("Hi, I'm Bob. I love programming in Python.")
    agent.print_response("What's my name and what do I like?")
    
    # Check stored sessions
    print("\n--- Stored sessions ---")
    session_ids = db.get_all_session_ids()
    print(f"Found {len(session_ids)} sessions: {session_ids}")


if __name__ == "__main__":
    # Run demos
    
    # Demo 1: SqliteDb with AgentMemory
    demo_sqlite_memory()
    
    # Demo 2: Direct SqliteDb usage
    demo_direct_sqlite()
    
    # Demo 3: Agent with session persistence
    demo_agent_with_session()
