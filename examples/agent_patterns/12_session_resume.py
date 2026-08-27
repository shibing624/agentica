# -*- coding: utf-8 -*-
"""
Session Resume Demo — CC-style JSONL session persistence.

Demonstrates:
1. session_id enables append-only JSONL logging at .sessions/{session_id}.jsonl
2. Each run() appends user input + assistant output to the log
3. On process restart, resume from the last compact boundary
4. Compact boundaries are written automatically during context compression

Usage:
    python 12_session_resume.py          # Run twice to see resume in action
    cat .sessions/demo-session.jsonl     # Inspect the JSONL log
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.model.openai import OpenAIChat
from agentica.memory.session_log import SessionLog


def demo_basic_session():
    """Demo 1: Basic session persistence — messages survive across runs."""
    print("=" * 60)
    print("Demo 1: Basic session persistence")
    print("=" * 60)

    session_id = "demo-session"

    # Check if session already exists
    log = SessionLog(session_id)
    if log.exists():
        print(f"\nFound existing session log: {log.path}")
        print(f"Entries: {log.entry_count()}")
        print(f"Resuming from last compact boundary...\n")
    else:
        print(f"\nNo existing session. Starting fresh.\n")

    # Create agent with session_id — auto-creates JSONL session log
    agent = Agent(
        name="session-agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        session_id=session_id,
        add_history_to_context=True,  # Include history in LLM context
    )

    # First run
    response = agent.run_sync("My name is Alice and I'm working on Project X.")
    print(f"User: My name is Alice and I'm working on Project X.")
    print(f"Assistant: {response.content}\n")

    # Second run — agent should remember context from first run
    response = agent.run_sync("What's my name and what am I working on?")
    print(f"User: What's my name and what am I working on?")
    print(f"Assistant: {response.content}\n")

    # Show session log contents
    print(f"\nSession log: {log.path}")
    print(f"Total entries: {log.entry_count()}")
    print(agent.session_log.format_trace())


def demo_resume_across_instances():
    """Demo 2: Resume works across separate Agent instances."""
    print("\n" + "=" * 60)
    print("Demo 2: Resume across Agent instances")
    print("=" * 60)

    session_id = "resume-demo"

    # Instance 1: Start a conversation
    agent1 = Agent(
        name="agent-v1",
        model=OpenAIChat(id="gpt-4o-mini"),
        session_id=session_id,
        add_history_to_context=True,
    )
    response = agent1.run_sync("Remember this: the secret code is 42.")
    print(f"\n[Instance 1] User: Remember this: the secret code is 42.")
    print(f"[Instance 1] Assistant: {response.content}")

    # Instance 2: Separate agent, same session_id — should resume
    agent2 = Agent(
        name="agent-v2",
        model=OpenAIChat(id="gpt-4o-mini"),
        session_id=session_id,
        add_history_to_context=True,
    )
    response = agent2.run_sync("What is the secret code?")
    print(f"\n[Instance 2] User: What is the secret code?")
    print(f"[Instance 2] Assistant: {response.content}")

    # Show the JSONL log
    log = SessionLog(session_id)
    print(f"\nSession log: {log.path}")
    print(f"Total entries: {log.entry_count()}")


def demo_compact_boundary():
    """Demo 3: Compact boundary — manual simulation."""
    print("\n" + "=" * 60)
    print("Demo 3: Compact boundary (simulated)")
    print("=" * 60)

    session_id = "compact-demo"
    log = SessionLog(session_id)

    # Simulate a long conversation with compaction
    log.append("user", "Tell me about Python.")
    log.append("assistant", "Python is a versatile programming language...")
    log.append("user", "What about async?")
    log.append("assistant", "Python supports async/await since 3.5...")
    log.append("user", "Show me an example.")
    log.append("assistant", "Here's an example: async def main(): ...")

    print(f"\nBefore compact: {log.entry_count()} entries")

    # Write compact boundary (normally done by CompressionManager.auto_compact)
    log.append_compact_boundary(
        "User asked about Python and async programming. "
        "Assistant explained Python basics and async/await with examples."
    )

    # New messages after compaction
    log.append("user", "Now tell me about type hints.")
    log.append("assistant", "Type hints in Python were introduced in PEP 484...")

    print(f"After compact: {log.entry_count()} entries")

    # Resume — should start from compact boundary
    resumed = log.load()
    print(f"\nResumed messages: {len(resumed)}")
    for msg in resumed:
        role = msg["role"]
        content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"  [{role}] {content}")


def cleanup_demo_sessions():
    """Clean up demo session files."""
    import shutil
    sessions_dir = ".sessions"
    if os.path.exists(sessions_dir):
        for f in ["demo-session.jsonl", "resume-demo.jsonl", "compact-demo.jsonl"]:
            path = os.path.join(sessions_dir, f)
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up: {path}")


if __name__ == "__main__":
    # Demo 3 works without API key (pure JSONL operations)
    demo_compact_boundary()

    # Demo 1 & 2 require OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        cleanup_demo_sessions()  # Start fresh for demo
        demo_basic_session()
        demo_resume_across_instances()
    else:
        print("\n[Skip Demo 1 & 2: Set OPENAI_API_KEY to run LLM demos]")

    # Cleanup
    cleanup_demo_sessions()
