# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Recommended memory management approach

This example shows the RECOMMENDED way to manage agent memory:

1. **Workspace** (for persistent user memories)
   - Human-readable Markdown files
   - Version control friendly (Git)
   - Easy to edit and share
   - Multi-user support with user_id isolation

2. **AgentMemory.with_summary()** (for session summaries)
   - Track conversation context
   - Generate session summaries
   - No external database required

For multi-user web applications requiring database storage,
see examples/memory/02_long_term_memory.py (deprecated for new projects)
"""
import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.memory import AgentMemory
from agentica.workspace import Workspace


def demo_workspace_memory():
    """Demo: Using Workspace for persistent memory (RECOMMENDED)"""
    print("=" * 60)
    print("Demo 1: Workspace for Persistent Memory (RECOMMENDED)")
    print("=" * 60)

    # Create temporary workspace
    temp_dir = 'outputs'
    workspace_path = Path(temp_dir) / "demo_workspace"

    # Initialize workspace
    workspace = Workspace(str(workspace_path))
    workspace.initialize()

    # Customize user preferences
    workspace.write_file("USER.md", """# User Profile

## Preferences
- Language: English
- Style: Technical and concise
- Topics: Python, AI, Machine Learning
""")

    # Create agent with workspace
    agent = Agent(
        model=OpenAIChat(model="gpt-4o-mini"),
        workspace=workspace,
        load_workspace_context=True,
        load_workspace_memory=True,
    )

    print("\n--- First conversation ---")
    response = agent.run_sync("My name is Alice and I'm a data scientist")
    print(f"Agent: {response.content[:200]}...")

    # Save important information to workspace memory
    agent.save_memory("User name: Alice, Profession: Data Scientist")

    print("\n--- Workspace memory content ---")
    memory_content = workspace.get_memory_prompt(days=1)
    print(memory_content or "(empty)")

    # Memory persists in Markdown files - can be viewed/edited directly
    print("\n--- Memory file location ---")
    print(f"Memory saved to: {workspace_path}/memory/")


def demo_multi_user_workspace():
    """Demo: Multi-user Workspace for isolated user memories"""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-User Workspace (User Isolation)")
    print("=" * 60)

    # Create temporary workspace
    temp_dir = 'outputs'
    workspace_path = Path(temp_dir) / "multi_user_workspace"

    # User 1: Alice
    print("\n--- User 1: Alice ---")
    ws_alice = Workspace(str(workspace_path), user_id="alice@example.com")
    ws_alice.initialize()
    ws_alice.save_memory("Alice prefers Python and machine learning topics")
    print(f"Alice's memory: {ws_alice.get_memory_prompt(days=1)}")

    # User 2: Bob
    print("\n--- User 2: Bob ---")
    ws_bob = Workspace(str(workspace_path), user_id="bob@test.com")
    ws_bob.initialize()
    ws_bob.save_memory("Bob prefers JavaScript and web development")
    print(f"Bob's memory: {ws_bob.get_memory_prompt(days=1)}")

    # List all users
    print("\n--- All Users ---")
    ws = Workspace(str(workspace_path))
    users = ws.list_users()
    print(f"Registered users: {users}")

    for user in users:
        info = ws.get_user_info(user)
        print(f"  {user}: {info['memory_count']} memories")

    # Create agent for specific user
    print("\n--- Agent for Alice ---")
    agent = Agent(
        model=OpenAIChat(model="gpt-4o-mini"),
        workspace_path=str(workspace_path),
        user_id="alice@example.com",
    )
    # Agent will automatically load Alice's memories and preferences
    print(f"Agent workspace user: {agent.workspace.user_id if agent.workspace else 'None'}")


def demo_session_summary():
    """Demo: Using AgentMemory for session summaries"""
    print("\n" + "=" * 60)
    print("Demo 3: Session Summary (for conversation tracking)")
    print("=" * 60)

    # Create agent with session summary enabled
    memory = AgentMemory.with_summary()
    agent = Agent(
        model=OpenAIChat(model="gpt-4o-mini"),
        memory=memory,
        add_history_to_messages=True,
    )

    print("\n--- Conversation ---")
    agent.print_response_sync("Hi, I need help with Python decorators")
    agent.print_response_sync("Can you show me a simple example?")
    agent.print_response_sync("How about a decorator with arguments?")

    print("\n--- Session Summary ---")
    # Generate summary
    summary = agent.memory.update_summary()
    if summary:
        print(f"Summary: {summary.summary}")
        print(f"Topics: {summary.topics}")
    else:
        print("No summary generated yet")


def demo_combined_approach():
    """Demo: Combining Workspace + Session Summary (BEST PRACTICE)"""
    print("\n" + "=" * 60)
    print("Demo 4: Combined Approach (BEST PRACTICE)")
    print("=" * 60)

    # Create temporary workspace
    temp_dir = 'outputs'
    workspace_path = Path(temp_dir) / "combined_workspace"

    # Initialize workspace with user
    workspace = Workspace(str(workspace_path), user_id="demo_user")
    workspace.initialize()

    # Create agent with BOTH workspace and session summary
    agent = Agent(
        model=OpenAIChat(model="gpt-4o-mini"),
        workspace=workspace,
        memory=AgentMemory.with_summary(),
        add_history_to_messages=True,
    )

    print("Configuration:")
    print(f"  - Workspace user: {workspace.user_id}")
    print("  - Workspace: Persistent user memories (Markdown files)")
    print("  - AgentMemory: Session history and summaries")

    print("\n--- Conversation ---")
    response = agent.run_sync("I'm Bob, working on a FastAPI project")
    print(f"Agent: {response.content[:150]}...")

    # Save important info to workspace
    agent.save_memory("User: Bob, Project: FastAPI")

    print("\n--- Saved memories (workspace) ---")
    memory = workspace.get_memory_prompt(days=1)
    print(memory or "(empty)")

    print("\n--- Session info (AgentMemory) ---")
    print(f"Messages in session: {len(agent.memory.messages)}")
    print(f"Runs in session: {len(agent.memory.runs)}")


def main():
    """Run all demos"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       Recommended Memory Management in Agentica              ║
╠══════════════════════════════════════════════════════════════╣
║  Workspace         → Persistent user memories (Markdown)     ║
║  Workspace(user_id)→ Multi-user isolation                    ║
║  AgentMemory       → Session history & summaries             ║
║                                                              ║
║  The old `enable_user_memories` + database approach is       ║
║  DEPRECATED. Use Workspace for new projects.                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    demo_workspace_memory()
    demo_multi_user_workspace()
    demo_session_summary()
    demo_combined_approach()

    print("\n" + "=" * 60)
    print("Summary: Memory Management Best Practices")
    print("=" * 60)
    print("""
┌─────────────────────────┬───────────────────────────────────┐
│ Use Case                │ Recommended Approach              │
├─────────────────────────┼───────────────────────────────────┤
│ Single user preferences │ Workspace (MEMORY.md)             │
│ Multi-user isolation    │ Workspace(user_id="xxx")          │
│ Session tracking        │ AgentMemory (messages, runs)      │
│ Session summaries       │ AgentMemory.with_summary()        │
│ Multi-user web app [*]  │ AgentMemory.with_db() [deprecated]│
└─────────────────────────┴───────────────────────────────────┘

Directory Structure (Multi-User):
workspace/
├── AGENT.md          # Global agent instructions
├── PERSONA.md        # Global persona settings
├── TOOLS.md          # Global tool guidelines
├── USER.md           # Default user preferences
├── MEMORY.md         # Default long-term memory
├── memory/           # Default daily memories
├── skills/           # Custom skills
└── users/            # User-specific data
    ├── alice@example.com/
    │   ├── USER.md
    │   ├── MEMORY.md
    │   └── memory/
    └── bob@test.com/
        ├── USER.md
        ├── MEMORY.md
        └── memory/
""")


if __name__ == "__main__":
    main()
