# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Long-term memory demo — persistent user/project memory

Demonstrates:
1. Write long-term memories as Markdown entries
2. Recall relevant memories by query
3. Keep memories isolated per user
4. Configure an Agent to use long-term memory

Workspace is the storage implementation. The user-facing concept is long-term
memory: information that survives across sessions and can be loaded later.
"""
import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.agent.config import WorkspaceMemoryConfig
from agentica.workspace import Workspace


MEMORY_ROOT = Path("outputs") / "long_term_memory_demo"


def reset_demo_memory() -> None:
    """Start each demo run from an empty memory store."""
    if MEMORY_ROOT.exists():
        shutil.rmtree(MEMORY_ROOT)


def create_long_term_memory(user_id: str) -> Workspace:
    """Create the file-backed long-term memory store for one user."""
    memory_store = Workspace(str(MEMORY_ROOT), user_id=user_id)
    memory_store.initialize()
    return memory_store


async def seed_user_memory(memory_store: Workspace) -> None:
    """Write durable facts that should survive across conversations."""
    await memory_store.write_memory_entry(
        title="User Python Preference",
        content="The user prefers Python, pytest, and type hints for backend projects.",
        memory_type="preference",
        description="Python testing and typing preferences",
    )
    await memory_store.write_memory_entry(
        title="Current RAG Project",
        content="The user is building a RAG system and cares about concise technical answers.",
        memory_type="project",
        description="RAG project context",
    )
    await memory_store.write_memory_entry(
        title="Model Provider Preference",
        content="The user often tests examples with ZhipuAI GLM or DeepSeek instead of OpenAI.",
        memory_type="preference",
        description="Preferred model providers",
    )


async def demo_persistent_recall() -> None:
    """Demo 1: Save durable memories and recall the relevant ones."""
    print("=" * 60)
    print("Demo 1: Persistent Long-Term Memory")
    print("=" * 60)

    memory_store = create_long_term_memory("alice")
    await seed_user_memory(memory_store)

    recalled = await memory_store.get_relevant_memories(
        query="How should I write tests for this Python RAG project?",
        limit=3,
    )
    print("\n--- Recalled memories for the next agent run ---")
    print(recalled or "(empty)")

    print("\n--- Search result for 'DeepSeek' ---")
    for item in memory_store.search_memory("DeepSeek", limit=3):
        print(f"[{item['score']:.2f}] {item['file_path']}: {item['content'][:90]}...")


async def demo_multi_user_isolation() -> None:
    """Demo 2: Different users get separate long-term memory spaces."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-User Memory Isolation")
    print("=" * 60)

    alice_memory = create_long_term_memory("alice")
    bob_memory = create_long_term_memory("bob")

    await bob_memory.write_memory_entry(
        title="Bob Frontend Stack",
        content="Bob is a frontend engineer using React and TypeScript.",
        memory_type="profile",
        description="Bob frontend stack",
    )

    alice_results = alice_memory.search_memory("React TypeScript", limit=5, min_score=1.0)
    bob_results = bob_memory.search_memory("React TypeScript", limit=5, min_score=1.0)

    print("\n--- Alice search for Bob's frontend stack ---")
    print(alice_results or "(empty)")
    print("\n--- Bob search for his frontend stack ---")
    for item in bob_results:
        print(f"[{item['score']:.2f}] {item['file_path']}: {item['content'][:90]}...")
    print(f"\nRegistered users: {alice_memory.list_users()}")


async def demo_agent_configuration() -> None:
    """Demo 3: Configure an Agent to load long-term memory automatically."""
    print("\n" + "=" * 60)
    print("Demo 3: Agent Long-Term Memory Configuration")
    print("=" * 60)

    memory_store = create_long_term_memory("alice")

    long_term_memory_options = WorkspaceMemoryConfig(
        load_workspace_context=True,
        load_workspace_memory=True,
        max_memory_entries=3,
        auto_archive=False,
        auto_extract_memory=False,
    )
    agent = Agent(
        name="long-term-memory-agent",
        workspace=memory_store,
        enable_long_term_memory=True,
        long_term_memory_config=long_term_memory_options,
        instructions="Use the user's long-term memory when it is relevant.",
    )

    memory_prompt = await memory_store.get_relevant_memories(query="What testing style should I use?", limit=3)
    print("\n--- Long-term memory the Agent can inject ---")
    print(memory_prompt or "(empty)")
    print("\nAgent is configured with:")
    print(f"  enable_long_term_memory={agent.enable_long_term_memory}")
    print(f"  max_memory_entries={agent.long_term_memory_config.max_memory_entries}")


async def main() -> None:
    reset_demo_memory()
    await demo_persistent_recall()
    await demo_multi_user_isolation()
    await demo_agent_configuration()

    print("\n" + "=" * 60)
    print(f"Demo memory files written to: {MEMORY_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
