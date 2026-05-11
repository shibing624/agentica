# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Recommended memory management — LLM auto-save & auto-load

This example shows how Agentica handles long-term memory **automatically**:

1. **Session 1** — Agent with long-term memory (BuiltinMemoryTool auto-registered):
   - LLM autonomously decides what to remember (user preferences, personal info, etc.)
   - Memories are persisted as Markdown files by the Workspace storage layer
   - No manual `memory_store.save_memory()` calls needed
   - If LLM doesn't call save_memory, MemoryExtractHooks auto-extracts memories

2. **Session 2** — A fresh Agent loads the same long-term memory store:
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
from agentica.agent.config import WorkspaceMemoryConfig
from agentica.memory import WorkingMemory
from agentica.model.providers import create_provider
from agentica.workspace import Workspace


def _create_agent(workspace: Workspace, **kwargs) -> Agent:
    """Helper: create an Agent with long-term memory for auto-memory.

    When workspace is set:
    - BuiltinMemoryTool (save_memory, search_memory) is auto-registered
    - auto_archive: saves raw conversation (zero cost)
    - auto_extract_memory: LLM-based memory extraction fallback (one LLM call)
    - Long-term memory is auto-injected into system prompt
    """
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=workspace,
        enable_long_term_memory=True,
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,
            auto_extract_memory=True,
        ),
        add_history_to_context=True,
        num_history_turns=5,
        **kwargs,
    )


async def demo_auto_memory():
    """Demo 1: LLM automatically saves & loads long-term memory (core demo)"""
    print("=" * 60)
    print("Demo 1: LLM Auto-Save Memory -> New Agent Auto-Load")
    print("=" * 60)

    workspace_path = Path("outputs") / "auto_memory_workspace"
    # Clean up for a fresh demo
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    # -- Session 1: Tell the agent about yourself --
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
    memory_content = await ws1.get_relevant_memories()
    if memory_content:
        print(memory_content)
    else:
        print("(No memory saved — try a model with better tool-calling support)")

    print(f"\nMemory files location: {workspace_path}/users/alice/")

    # -- Session 2: A brand new Agent loads the same workspace --
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
        add_history_to_context=True,
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
        print("(Summary not yet generated -- run more turns)")


async def demo_combined():
    """Demo 4: Production-grade full stack -- long-term memory + auxiliary model
    for cheap+fast memory extraction + background extraction (non-blocking) +
    session summary.

    Three perf-critical knobs (all introduced in 1.4.4):
      1. ``auxiliary_model``: MemoryExtractHooks runs on this cheaper/faster
         model instead of the main one (e.g. DeepSeek Flash vs GPT-4o), so the
         extraction LLM call costs less and finishes ~5x faster.
      2. ``auto_extract_memory_background=True``: extraction is fire-and-forget
         via ``asyncio.create_task`` and does NOT block ``on_agent_end``. The
         user-visible response RT drops by the full extraction latency.
         ⚠️  Only enable under a long-running event loop (FastAPI, asyncio.run).
             Do NOT enable under ``run_sync()`` / ``run_stream_sync()``: the temp
             loop closes before the task completes and memories are lost.
      3. ``sync_memories_to_global_agent_md=True``: confirmed user-type memories
         get mirrored into ~/.agentica/AGENTS.md so future sessions inherit them.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Production Full Stack (auxiliary_model + background extract)")
    print("=" * 60)

    workspace_path = Path("outputs") / "best_practice_workspace"
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    workspace = Workspace(str(workspace_path), user_id="demo_user")
    workspace.initialize()

    # auxiliary_model: only used by MemoryExtractHooks for the extraction
    # sub-call. Falls back to the main model when DEEPSEEK_API_KEY is missing.
    auxiliary_model = (
        create_provider("deepseek") if os.getenv("DEEPSEEK_API_KEY") else None
    )

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        auxiliary_model=auxiliary_model,
        workspace=workspace,
        enable_long_term_memory=True,
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,
            auto_extract_memory=True,
            auto_extract_memory_background=True,
            sync_memories_to_global_agent_md=True,
        ),
        working_memory=WorkingMemory.with_summary(),
        add_history_to_context=True,
    )

    print(
        "Config: long-term memory + SessionSummary + auto memory "
        f"(auxiliary={'deepseek' if auxiliary_model else 'main-model fallback'}, "
        "background=True, sync_to_global_AGENTS.md=True)"
    )

    await agent.print_response(
        "I'm a senior engineer building a RAG system with LangChain and Qdrant. "
        "I prefer type-annotated Python code with detailed docstrings. Please remember."
    )

    await agent.print_response("What are the best practices for chunking strategies?")

    # background=True schedules extraction via asyncio.create_task. In a real
    # FastAPI server you simply return the response and the task finishes on
    # its own. In this synchronous demo we explicitly drain pending tasks so
    # we can observe the extracted memories in the same run.
    pending = [
        t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
    ]
    if pending:
        print(f"\n[demo] waiting for {len(pending)} background extraction task(s)...")
        await asyncio.gather(*pending, return_exceptions=True)

    print("\n--- Long-term memory ---")
    memory = await workspace.get_relevant_memories()
    print(memory or "(empty)")

    print(f"\n--- Session stats ---")
    print(f"  Messages: {len(agent.working_memory.messages)}")
    print(f"  Runs: {len(agent.working_memory.runs)}")
    if agent.working_memory.summary:
        print(f"  Summary: {agent.working_memory.summary.summary[:100]}...")


async def main():
    print("""
=================================================================
         Agentica: Automatic Long-Term Memory
=================================================================
  BuiltinMemoryTool -> LLM decides when to save memory
  MemoryExtractHooks -> auto-extracts if LLM didn't save
  Workspace storage -> Persistent Markdown files (Git-friendly)
  WorkingMemory      -> Session history & auto-summary

  Key: No manual save_memory() calls needed!
  The LLM autonomously saves important info as a tool call.
  If it doesn't, MemoryExtractHooks catches it as a fallback.
=================================================================
""")

    await demo_auto_memory()
    await demo_multi_user()
    await demo_session_summary()
    await demo_combined()

    print("\n" + "=" * 60)
    print("How It Works")
    print("=" * 60)
    print("""
  1. Agent with long-term memory -> BuiltinMemoryTool auto-registered
  2. LLM detects important info -> calls save_memory() tool
  3. Memory persisted to long_term_memory_root/users/{user}/memory/*.md
  4. If LLM didn't save -> MemoryExtractHooks auto-extracts
  5. Next session: long-term memory auto-injected into prompt
  6. Agent "remembers" without any manual code

Storage Layout:
  long_term_memory_root/
  +-- AGENTS.md            # Agent instructions
  +-- users/
  |   +-- {user_id}/
  |       +-- USER.md     # User profile
  |       +-- MEMORY.md   # Memory index (links to files)
  |       +-- memory/
  |       |   +-- user_alice_role.md
  |       |   +-- project_deploy_target.md
  |       +-- conversations/
  |           +-- 2025-01-01.md  # Conversation archive
""")


if __name__ == "__main__":
    asyncio.run(main())
