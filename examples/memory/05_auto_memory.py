# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Recommended memory management — LLM auto-save & auto-load

This example shows how Agentica handles long-term memory **automatically**:

1. **Session 1** — Agent with `BuiltinMemoryTool`:
   - LLM autonomously decides what to remember (user preferences, personal info, etc.)
   - Memories are persisted as Markdown files in the Workspace
   - No manual `agent.save_memory()` calls needed

2. **Session 2** — A fresh Agent loads the same Workspace:
   - Long-term memories are automatically injected into the system prompt
   - Agent "remembers" facts from Session 1 without any code

3. **Multi-user isolation** — Each user has their own memory directory

4. **Session summary** — Compressed conversation history across turns
"""
import asyncio
import shutil
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.memory import WorkingMemory
from agentica.workspace import Workspace
from agentica.deep_tools import BuiltinMemoryTool


def _create_agent(workspace: Workspace, **kwargs) -> Agent:
    """Helper: create an Agent with BuiltinMemoryTool (LLM can auto-save memory)."""
    memory_tool = BuiltinMemoryTool(workspace=workspace)
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=workspace,
        tools=[memory_tool],
        add_history_to_messages=True,
        history_window=5,
        **kwargs,
    )


async def demo_auto_memory():
    """Demo 1: LLM automatically saves & loads long-term memory (core demo)"""
    print("=" * 60)
    print("Demo 1: LLM Auto-Save Memory → New Agent Auto-Load")
    print("=" * 60)

    workspace_path = Path("outputs") / "auto_memory_workspace"
    # Clean up for a fresh demo
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    # ── Session 1: Tell the agent about yourself ──
    print("\n--- Session 1: First conversation (LLM auto-saves memory) ---")
    ws1 = Workspace(str(workspace_path), user_id="alice")
    ws1.initialize()
    agent1 = _create_agent(ws1)

    # The LLM will detect personal info and call save_memory automatically
    await agent1.print_response(
        "Hi! I'm Alice, a machine learning researcher at Stanford. "
        "I prefer Python and PyTorch. Please remember my info."
    )

    await agent1.print_response(
        "I'm working on a multimodal LLM project using Vision Transformers. "
        "Please note this for our future conversations."
    )

    # Show what the LLM saved (without manual intervention)
    print("\n--- What LLM auto-saved to workspace ---")
    memory_content = await ws1.get_memory_prompt(days=7)
    if memory_content:
        print(memory_content)
    else:
        print("(No memory saved — try a model with better tool-calling support)")

    print(f"\nMemory files location: {workspace_path}/users/alice/")

    # ── Session 2: A brand new Agent loads the same workspace ──
    print("\n\n--- Session 2: Fresh Agent auto-loads memory ---")
    ws2 = Workspace(str(workspace_path), user_id="alice")
    agent2 = _create_agent(ws2)

    # This agent has NEVER seen Session 1 messages, but knows about Alice
    await agent2.print_response(
        "What do you know about me? What am I working on?"
    )


async def demo_multi_user():
    """Demo 2: Multi-user workspace with isolated memories"""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-User Memory Isolation")
    print("=" * 60)

    workspace_path = Path("outputs") / "multi_user_workspace"
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    # User 1: Alice
    print("\n--- User: Alice ---")
    ws_alice = Workspace(str(workspace_path), user_id="alice@example.com")
    ws_alice.initialize()
    agent_alice = _create_agent(ws_alice)
    await agent_alice.print_response(
        "I'm Alice, a Python developer. I love FastAPI and async programming. Remember this."
    )

    # User 2: Bob
    print("\n--- User: Bob ---")
    ws_bob = Workspace(str(workspace_path), user_id="bob@test.com")
    ws_bob.initialize()
    agent_bob = _create_agent(ws_bob)
    await agent_bob.print_response(
        "I'm Bob, a frontend engineer using React and TypeScript. Remember my info."
    )

    # Verify isolation: Alice's agent should NOT know about Bob
    print("\n--- Verify isolation: Alice's agent ---")
    ws_alice2 = Workspace(str(workspace_path), user_id="alice@example.com")
    agent_alice2 = _create_agent(ws_alice2)
    await agent_alice2.print_response("What do you know about me?")

    # List all users
    print("\n--- All registered users ---")
    ws = Workspace(str(workspace_path))
    for user in ws.list_users():
        info = ws.get_user_info(user)
        print(f"  {user}: {info['memory_count']} memories")


async def demo_session_summary():
    """Demo 3: Session summary for compressed conversation history"""
    print("\n" + "=" * 60)
    print("Demo 3: Session Summary (compressed history)")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        working_memory=WorkingMemory.with_summary(),
        add_history_to_messages=True,
    )

    await agent.print_response("Hi, I need help with Python decorators")
    await agent.print_response("Can you show me a decorator with arguments?")
    await agent.print_response("How do I stack multiple decorators?")

    print("\n--- Auto-generated session summary ---")
    summary = agent.working_memory.summary
    if summary:
        print(f"Summary: {summary.summary}")
        print(f"Topics: {summary.topics}")
    else:
        print("(Summary not yet generated — run more turns)")


async def demo_combined():
    """Demo 4: Best practice — Workspace + BuiltinMemoryTool + Session Summary"""
    print("\n" + "=" * 60)
    print("Demo 4: Best Practice (Workspace + Auto Memory + Summary)")
    print("=" * 60)

    workspace_path = Path("outputs") / "best_practice_workspace"
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    workspace = Workspace(str(workspace_path), user_id="demo_user")
    workspace.initialize()

    memory_tool = BuiltinMemoryTool(workspace=workspace)
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=workspace,
        tools=[memory_tool],
        working_memory=WorkingMemory.with_summary(),
        add_history_to_messages=True,
    )

    print("Config: Workspace (persistent) + BuiltinMemoryTool (LLM auto-save) + SessionSummary")

    await agent.print_response(
        "I'm a senior engineer building a RAG system with LangChain and Qdrant. "
        "I prefer type-annotated Python code with detailed docstrings. Please remember."
    )

    await agent.print_response("What are the best practices for chunking strategies?")

    print("\n--- Workspace memory ---")
    print(await workspace.get_memory_prompt(days=7) or "(empty)")

    print(f"\n--- Session stats ---")
    print(f"  Messages: {len(agent.working_memory.messages)}")
    print(f"  Runs: {len(agent.working_memory.runs)}")
    if agent.working_memory.summary:
        print(f"  Summary: {agent.working_memory.summary.summary[:100]}...")


async def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         Agentica: Automatic Long-Term Memory                  ║
╠═══════════════════════════════════════════════════════════════╣
║  BuiltinMemoryTool → LLM decides when to save memory         ║
║  Workspace         → Persistent Markdown files (Git-friendly) ║
║  WorkingMemory       → Session history & auto-summary           ║
║                                                               ║
║  Key: No manual save_memory() calls needed!                   ║
║  The LLM autonomously saves important info as a tool call.    ║
╚═══════════════════════════════════════════════════════════════╝
""")

    await demo_auto_memory()
    await demo_multi_user()
    await demo_session_summary()
    await demo_combined()

    print("\n" + "=" * 60)
    print("How It Works")
    print("=" * 60)
    print("""
┌───────────────────────────────────────────────────────────────┐
│  1. Agent receives BuiltinMemoryTool + system prompt          │
│  2. LLM detects important info → calls save_memory() tool     │
│  3. Memory persisted to workspace/users/{user}/MEMORY.md      │
│  4. Next session: Workspace memory auto-injected into prompt  │
│  5. Agent "remembers" without any manual code                 │
└───────────────────────────────────────────────────────────────┘

Memory Architecture:
  workspace/
  ├── AGENT.md            # Agent instructions
  ├── users/
  │   └── {user_id}/
  │       ├── USER.md     # User profile
  │       ├── MEMORY.md   # Long-term memory (permanent)
  │       └── memory/
  │           └── {date}.md  # Daily memory (7-day TTL)
""")


if __name__ == "__main__":
    asyncio.run(main())
