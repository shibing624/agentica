# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Long-term memory demo with semantic search and keyword fallback

Usage:
1. QdrantMemoryDb with embedder: semantic search
2. QdrantMemoryDb without embedder: keyword search fallback
3. CsvMemoryDb/SqliteMemoryDb: traditional retrieval (last_n, first_n)
"""
import sys
sys.path.append('..')
sys.path.append('../..')

from agentica import (
    Agent, OpenAIChat, AgentMemory, QdrantMemoryDb, OpenAIEmb,
    MemoryClassifier, MemoryRetrieval
)


def demo_qdrant_with_embedding():
    """Demo with QdrantMemoryDb + embedding model for semantic search."""
    print("=" * 50)
    print("Demo 1: QdrantMemoryDb with semantic search")
    print("=" * 50)
    
    # Initialize embedding model
    embedder = OpenAIEmb()
    
    # Create QdrantMemoryDb with embedder
    db = QdrantMemoryDb(
        collection="demo_semantic",
        embedder=embedder,
        on_disk=True,
    )
    
    # Initialize memory with QdrantMemoryDb
    memory = AgentMemory(
        db=db,
        user_id="test_user",
        num_memories=5,
        retrieval=MemoryRetrieval.semantic,  # Use semantic retrieval
        create_user_memories=True,
        update_user_memories_after_run=True,
        classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
        semantic_score_threshold=0.5,
    )
    
    # Create agent with memory
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        memory=memory,
        debug=True,
    )
    
    # First conversation
    print("\n--- First conversation ---")
    agent.print_response("My name is Alice and I work as a software engineer at Google.")
    
    # Search memories
    print("\n--- Searching memories ---")
    relevant = memory.search_memories("What does Alice do?")
    print(f"Found {len(relevant)} relevant memories")
    for m in relevant:
        print(f"  - {m.memory}")


def demo_qdrant_keyword_fallback():
    """Demo with QdrantMemoryDb without embedder - keyword search fallback."""
    print("\n" + "=" * 50)
    print("Demo 2: QdrantMemoryDb with keyword search (no embedder)")
    print("=" * 50)
    
    # Create QdrantMemoryDb without embedder (keyword search mode)
    # Use a different path to avoid concurrent access conflict
    db = QdrantMemoryDb(
        collection="demo_keyword",
        embedder=None,  # No embedder, will use keyword search
        on_disk=True,
        path="outputs/qdrant_memory_keyword",  # Use different path
    )
    
    # Initialize memory
    memory = AgentMemory(
        db=db,
        user_id="test_user_2",
        num_memories=5,
        retrieval=MemoryRetrieval.semantic,  # Will fallback to keyword search
        create_user_memories=True,
        update_user_memories_after_run=True,
    )
    
    # Manually add some memories using the manager
    from agentica.memorydb import MemoryRow
    from agentica.memory import Memory
    
    memories_to_add = [
        "Bob is a data scientist who loves Python.",
        "Bob enjoys hiking and photography.",
        "Bob's favorite programming language is Python.",
    ]
    
    for content in memories_to_add:
        memory_data = Memory(memory=content, input_text=content).to_dict()
        row = MemoryRow(user_id="test_user_2", memory=memory_data)
        db.upsert_memory(row)
    
    print(f"Added {len(memories_to_add)} memories")
    
    # Search using keyword matching
    print("\n--- Searching with keywords ---")
    results = memory.search_memories("Python programming")
    print(f"Found {len(results)} memories matching 'Python programming':")
    for m in results:
        print(f"  - {m.memory}")
    
    # Get formatted string
    formatted = memory.get_relevant_memories_str("Python")
    print(f"\nFormatted memories:\n{formatted}")


def demo_csv_memory():
    """Demo with CsvMemoryDb - traditional retrieval."""
    print("\n" + "=" * 50)
    print("Demo 3: CsvMemoryDb with traditional retrieval")
    print("=" * 50)
    
    from agentica import CsvMemoryDb
    
    # Create CsvMemoryDb
    db = CsvMemoryDb(csv_file_path="outputs/demo_memory.csv")
    
    # Initialize memory (same API as before)
    memory = AgentMemory(
        db=db,
        user_id="test_user_3",
        num_memories=10,
        retrieval=MemoryRetrieval.last_n,  # Traditional retrieval
        create_user_memories=True,
        update_user_memories_after_run=True,
        classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
    )
    
    print("CsvMemoryDb initialized - works the same as before!")
    print(f"Memory file: outputs/demo_memory.csv")


def demo_direct_qdrant():
    """Demo using QdrantMemoryDb directly."""
    print("\n" + "=" * 50)
    print("Demo 4: Using QdrantMemoryDb directly")
    print("=" * 50)
    
    from agentica.memorydb import MemoryRow
    from agentica.memory import Memory
    
    # Create QdrantMemoryDb directly
    # Use a different path to avoid concurrent access conflict
    db = QdrantMemoryDb(
        collection="direct_demo",
        embedder=None,  # Keyword search mode
        on_disk=True,
        path="outputs/qdrant_memory_direct",  # Use different path
    )
    
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
    
    # Search
    print("\n--- Searching for 'capital' ---")
    results = db.search_memories("capital city", user_id="demo_user", limit=3)
    for row in results:
        print(f"  - {row.memory.get('memory', '')}")
    
    # Get formatted string
    formatted = db.get_relevant_memories("programming language", user_id="demo_user")
    print(f"\nFormatted results for 'programming language':\n{formatted}")


if __name__ == "__main__":
    # Run demos (choose based on your setup)
    
    # Demo 1: Requires OpenAI API key for embeddings
    demo_qdrant_with_embedding()
    
    # Demo 2: No embedder needed, uses keyword search
    demo_qdrant_keyword_fallback()
    
    # Demo 3: Traditional CsvMemoryDb
    demo_csv_memory()
    
    # Demo 4: Direct QdrantMemoryDb usage
    demo_direct_qdrant()
