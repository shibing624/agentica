# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Workspace memory demo — file-based persistent memory

Demonstrates:
1. Basic Workspace memory — save and read (daily + long-term)
2. Multi-user isolation — different users have separate memory spaces
3. Memory search — keyword search across memory files
4. Agent with Workspace — Agent automatically loads workspace memory into context
5. Memory lifecycle — create, read, clear
"""
import asyncio
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.workspace import Workspace


async def demo_basic_workspace_memory():
    """Demo 1: Basic Workspace memory — save and read."""
    print("=" * 60)
    print("Demo 1: Basic Workspace Memory")
    print("=" * 60)

    workspace_path = "outputs/demo_workspace"
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)

    workspace = Workspace(workspace_path)
    workspace.initialize()

    # Save daily memories (default)
    await workspace.save_memory("User prefers Python for backend development")
    await workspace.save_memory("User is working on an AI agent framework")
    print("Saved 2 daily memories")

    # Save long-term memories
    await workspace.save_memory("User's name is Alice, software engineer at Google", long_term=True)
    await workspace.save_memory("User likes hiking and photography", long_term=True)
    print("Saved 2 long-term memories")

    # Read memory prompt (what gets injected into Agent context)
    memory_prompt = await workspace.get_memory_prompt(days=7)
    print(f"\n--- Memory Prompt (injected into Agent context) ---")
    print(memory_prompt[:500] if memory_prompt else "(empty)")


async def demo_multi_user_memory():
    """Demo 2: Multi-user isolation."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-User Isolation")
    print("=" * 60)

    workspace_path = "outputs/demo_workspace"
    workspace = Workspace(workspace_path)

    # User 1: Alice
    workspace.set_user("alice@example.com")
    workspace.initialize()
    await workspace.save_memory("Alice is a Python developer", long_term=True)
    await workspace.save_memory("Alice asked about async programming")
    print("Saved memories for alice@example.com")

    # User 2: Bob
    workspace.set_user("bob@example.com")
    workspace.initialize()
    await workspace.save_memory("Bob is a data scientist", long_term=True)
    await workspace.save_memory("Bob asked about pandas performance")
    print("Saved memories for bob@example.com")

    # Read memories for each user — they are isolated
    workspace.set_user("alice@example.com")
    alice_memory = await workspace.get_memory_prompt()
    print(f"\n--- Alice's Memory ---")
    print(alice_memory[:300] if alice_memory else "(empty)")

    workspace.set_user("bob@example.com")
    bob_memory = await workspace.get_memory_prompt()
    print(f"\n--- Bob's Memory ---")
    print(bob_memory[:300] if bob_memory else "(empty)")

    # List all users
    users = workspace.list_users()
    print(f"\nRegistered users: {users}")


async def demo_memory_search():
    """Demo 3: Memory search."""
    print("\n" + "=" * 60)
    print("Demo 3: Memory Search")
    print("=" * 60)

    workspace_path = "outputs/demo_workspace"
    workspace = Workspace(workspace_path, user_id="alice@example.com")

    # Search memories by keyword
    results = workspace.search_memory("Python", limit=5)
    print(f"Search 'Python': found {len(results)} results")
    for r in results:
        print(f"  [{r['score']:.1f}] {r['file_path']}: {r['content'][:80]}...")

    results = workspace.search_memory("data science", limit=5)
    print(f"\nSearch 'data science': found {len(results)} results")
    for r in results:
        print(f"  [{r['score']:.1f}] {r['file_path']}: {r['content'][:80]}...")


async def demo_agent_with_workspace():
    """Demo 4: Agent with Workspace — memory auto-loaded into context."""
    print("\n" + "=" * 60)
    print("Demo 4: Agent with Workspace")
    print("=" * 60)

    workspace_path = "outputs/demo_workspace"
    workspace = Workspace(workspace_path, user_id="demo_user")
    workspace.initialize()

    # Pre-populate some memories
    await workspace.save_memory("User's name is David, ML engineer", long_term=True)
    await workspace.save_memory("User prefers concise answers with code examples", long_term=True)
    await workspace.save_memory("Discussed PyTorch vs TensorFlow, user chose PyTorch")

    # Create Agent with Workspace
    # The Agent will automatically load workspace context and memory into its system prompt
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=workspace,
        add_history_to_messages=True,
    )

    print("\n--- Conversation (Agent has workspace memory in context) ---")
    await agent.print_response("What's my name and what framework do I use?")
    await agent.print_response("Give me a quick PyTorch training loop example")

    # Save new memory after conversation
    await workspace.save_memory("User requested a PyTorch training loop example")
    print("\n--- New memory saved ---")

    # Show all memory files
    files = workspace.get_all_memory_files()
    print(f"\nAll memory files for demo_user: {len(files)}")
    for f in files:
        print(f"  - {f.relative_to(workspace.path)}")


async def demo_memory_lifecycle():
    """Demo 5: Memory lifecycle — create, read, clear."""
    print("\n" + "=" * 60)
    print("Demo 5: Memory Lifecycle")
    print("=" * 60)

    workspace_path = "outputs/demo_workspace"
    workspace = Workspace(workspace_path, user_id="lifecycle_user")
    workspace.initialize()

    # Create memories over multiple "days" (simulate with direct file writes)
    await workspace.save_memory("Day 1: Started learning Rust")
    await workspace.save_memory("Important: User is allergic to peanuts", long_term=True)

    files_before = workspace.get_all_memory_files()
    print(f"Memory files: {len(files_before)}")

    # Read long-term memory
    memory_md = workspace._get_user_memory_md()
    if memory_md.exists():
        print(f"\nLong-term memory:\n{memory_md.read_text(encoding='utf-8')}")

    # Clear old daily memories (keep last 7 days)
    workspace.clear_daily_memory(keep_days=7)
    files_after = workspace.get_all_memory_files()
    print(f"\nAfter cleanup: {len(files_after)} memory files")

    # User info
    info = workspace.get_user_info()
    print(f"\nUser info: {info}")


if __name__ == "__main__":
    async def _main():
        await demo_basic_workspace_memory()
        await demo_multi_user_memory()
        await demo_memory_search()
        await demo_agent_with_workspace()
        await demo_memory_lifecycle()

        print("\n" + "=" * 60)
        print("All demos completed!")
        print("=" * 60)

    asyncio.run(_main())
