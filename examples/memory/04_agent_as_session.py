# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent-as-Session demo - Agent instance IS the session

This example demonstrates Agentica's "Agent-as-Session" pattern:
- Agent instance holds its own AgentMemory, no separate Session object needed
- Each run() automatically records to memory.runs
- History injection via add_history_to_messages=True
- Compared to OpenAI Agents SDK's Runner+Session separation, this is more elegant

Key advantages over OpenAI Agents SDK:
1. No separate Session/Runner objects - Agent IS the session
2. Automatic history tracking via AgentMemory.runs
3. Fine-grained control: num_history_responses, max_tokens budget
4. Tool result truncation for older runs to save context
5. Message injection happens transparently in get_messages_for_run()
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.memory import AgentMemory


async def demo1_basic_session():
    """Demo 1: Basic Agent-as-Session - automatic multi-turn memory"""
    print("=" * 60)
    print("Demo 1: Basic Agent-as-Session")
    print("  Agent itself is the session, no extra Session object needed")
    print("=" * 60)

    # One line: create agent with session capability
    # Compare: OpenAI SDK needs Runner.run(agent, input, session=Session())
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        add_history_to_messages=True,
        num_history_responses=5,
        instructions="You are a helpful assistant. Remember what the user tells you.",
    )

    # Turn 1: introduce yourself
    print("\n[Turn 1] User: My name is Alice, I'm a machine learning engineer")
    r1 = await agent.run("My name is Alice, I'm a machine learning engineer")
    print(f"Agent: {r1.content[:200]}")

    # Turn 2: add more context
    print("\n[Turn 2] User: I'm working on a RAG system using LangChain")
    r2 = await agent.run("I'm working on a RAG system using LangChain")
    print(f"Agent: {r2.content[:200]}")

    # Turn 3: test recall - agent should remember both turns
    print("\n[Turn 3] User: What do you know about me and my project?")
    r3 = await agent.run("What do you know about me and my project?")
    print(f"Agent: {r3.content[:300]}")

    # Inspect internal state
    print(f"\n--- Internal State ---")
    print(f"  memory.runs count: {len(agent.memory.runs)}")
    print(f"  memory.messages count: {len(agent.memory.messages)}")

    # Get history messages that would be injected
    history = agent.memory.get_messages_from_last_n_runs(last_n=3)
    print(f"  History messages for next run: {len(history)}")
    for msg in history:
        role = msg.role
        content = str(msg.content)[:80] if msg.content else ""
        print(f"    [{role}] {content}...")


async def demo2_session_isolation():
    """Demo 2: Multiple independent sessions via separate Agent instances"""
    print("\n" + "=" * 60)
    print("Demo 2: Session Isolation - Each Agent = Independent Session")
    print("  No session ID management, no session store, just create agents")
    print("=" * 60)

    # Session A: Alice's conversation
    session_a = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        add_history_to_messages=True,
    )

    # Session B: Bob's conversation
    session_b = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        add_history_to_messages=True,
    )

    # Alice talks
    print("\n[Session A] Alice: I prefer Python and TensorFlow")
    await session_a.run("I prefer Python and TensorFlow")

    # Bob talks
    print("[Session B] Bob: I prefer Rust and low-level systems programming")
    await session_b.run("I prefer Rust and low-level systems programming")

    # Each session recalls its own context
    print("\n[Session A] Alice: What's my tech stack?")
    ra = await session_a.run("What's my tech stack?")
    print(f"  Agent -> Alice: {ra.content[:200]}")

    print("\n[Session B] Bob: What's my tech stack?")
    rb = await session_b.run("What's my tech stack?")
    print(f"  Agent -> Bob: {rb.content[:200]}")

    # Verify isolation
    print(f"\n--- Session Isolation Verified ---")
    print(f"  Session A runs: {len(session_a.memory.runs)}, messages: {len(session_a.memory.messages)}")
    print(f"  Session B runs: {len(session_b.memory.runs)}, messages: {len(session_b.memory.messages)}")


async def demo3_resume_conversation():
    """Demo 3: Resume/continue a conversation by reusing the same Agent"""
    print("\n" + "=" * 60)
    print("Demo 3: Resume Conversation - Agent memory persists across runs")
    print("  No need to pass previous messages or session tokens")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        add_history_to_messages=True,
        num_history_responses=10,
        instructions="You are a coding tutor. Track the student's learning progress.",
    )

    # Simulate a multi-step learning session
    conversations = [
        "I want to learn Python decorators",
        "Can you explain closures first? I think they're related",
        "Now show me a simple decorator example",
        "How do I make a decorator that accepts arguments?",
        "Summarize everything we've covered in this session",
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"\n[Turn {i}] Student: {msg}")
        r = await agent.run(msg)
        print(f"  Tutor: {r.content[:200]}...")

    # The agent automatically tracked the entire learning progression
    print(f"\n--- Learning Session Stats ---")
    print(f"  Total turns: {len(agent.memory.runs)}")

    # History retrieval with different window sizes
    recent_3 = agent.memory.get_messages_from_last_n_runs(last_n=3)
    recent_all = agent.memory.get_messages_from_last_n_runs()
    print(f"  Messages from last 3 runs: {len(recent_3)}")
    print(f"  Messages from all runs: {len(recent_all)}")


async def demo4_shared_memory():
    """Demo 4: Handoff between agents sharing the same AgentMemory"""
    print("\n" + "=" * 60)
    print("Demo 4: Agent Handoff via Shared Memory")
    print("  Transfer conversation context by sharing AgentMemory instance")
    print("=" * 60)

    # Shared memory - this is the session state
    shared_memory = AgentMemory()

    # Agent 1: Sales agent collects requirements
    sales_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        memory=shared_memory,
        add_history_to_messages=True,
        instructions="You are a sales agent. Collect customer requirements for a software project.",
    )

    print("\n[Sales Agent] Collecting requirements...")
    await sales_agent.run("I need a web app for managing my restaurant's orders and inventory")
    await sales_agent.run("It should support real-time order tracking and have a mobile-friendly interface")

    print(f"  Sales agent recorded {len(shared_memory.runs)} runs")

    # Agent 2: Technical architect picks up the conversation
    # Same memory = same session context, different agent personality
    tech_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        memory=shared_memory,
        add_history_to_messages=True,
        instructions="You are a technical architect. Based on conversation history, propose a technical solution.",
    )

    print("\n[Tech Agent] Proposing solution based on sales conversation...")
    r = await tech_agent.run(
        "Based on the customer requirements discussed earlier, propose a technical architecture"
    )
    print(f"  Tech Agent: {r.content[:400]}...")

    print(f"\n--- Shared Memory Stats ---")
    print(f"  Total runs across both agents: {len(shared_memory.runs)}")
    print(f"  Total messages: {len(shared_memory.messages)}")


async def demo5_history_control():
    """Demo 5: Fine-grained history control - sliding window, token budget"""
    print("\n" + "=" * 60)
    print("Demo 5: Fine-Grained History Control")
    print("  Control exactly how much history context the model sees")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        add_history_to_messages=True,
        num_history_responses=2,  # only last 2 runs
        instructions="Answer briefly.",
    )

    # Build up conversation history
    topics = [
        "The capital of France is Paris",
        "The capital of Japan is Tokyo",
        "The capital of Brazil is Brasilia",
        "The capital of Australia is Canberra",
    ]

    for topic in topics:
        await agent.run(f"Remember: {topic}")

    print(f"Total runs recorded: {len(agent.memory.runs)}")

    # With num_history_responses=2, only last 2 runs are visible
    print("\n[Query] What capitals have I told you about?")
    r = await agent.run("List ALL the capitals I've told you about")
    print(f"  Agent (window=2): {r.content[:300]}")
    print("  (Agent only sees last 2 runs, so it may miss earlier ones)")

    # Demonstrate get_messages_from_last_n_runs with different windows
    print(f"\n--- History Window Comparison ---")
    for n in [1, 2, 4, None]:
        msgs = agent.memory.get_messages_from_last_n_runs(last_n=n)
        label = f"last_{n}" if n else "all"
        print(f"  {label}: {len(msgs)} messages")


async def main():
    print("""
+==============================================================+
|          Agentica: Agent-as-Session Pattern Demo              |
+==============================================================+
| In Agentica, the Agent instance IS the session.              |
| No separate Session/Runner objects needed.                   |
|                                                              |
| OpenAI SDK:  result = Runner.run(agent, input, session=...)  |
| Agentica:    result = await agent.run(input)  # that's it!   |
+==============================================================+
""")

    await demo1_basic_session()
    await demo2_session_isolation()
    await demo3_resume_conversation()
    await demo4_shared_memory()
    await demo5_history_control()

    print("\n" + "=" * 60)
    print("Summary: Agent-as-Session vs OpenAI Runner+Session")
    print("=" * 60)
    print("""
+---------------------------+-------------------------------------+
| OpenAI Agents SDK         | Agentica                            |
+---------------------------+-------------------------------------+
| Runner.run(agent, input,  | agent.run(input)                    |
|   session=Session())      |   # Agent IS the session            |
+---------------------------+-------------------------------------+
| Session stores messages   | AgentMemory.runs stores all history |
| Need to pass session obj  | Memory is built into Agent          |
| Manual message injection  | Automatic via add_history_to_messages|
| No token budget control   | get_messages_from_last_n_runs       |
|                           |   supports max_tokens budget        |
| No tool result truncation | Auto-truncate old tool results      |
| Handoff needs session mgmt| Share AgentMemory between agents    |
+---------------------------+-------------------------------------+
""")


if __name__ == "__main__":
    asyncio.run(main())
