# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Session memory demo - Demonstrates persistent user memory with Agent

DEPRECATED: The `enable_user_memories` feature is deprecated.
For persistent user memories, use Workspace instead (see examples/workspace/):

    from agentica import Agent
    from agentica.workspace import Workspace

    workspace = Workspace("~/.agentica/workspace")
    agent = Agent(workspace=workspace, model=...)
    agent.save_memory("User prefers concise responses")

This example is preserved for:
1. Legacy support
2. Multi-user web applications where database-based storage is required
3. Understanding session history management (which is NOT deprecated)

What's NOT deprecated:
- Session history (add_history_to_messages)
- Session summaries (AgentMemory.with_summary())

What IS deprecated:
- enable_user_memories
- AgentMemory.with_db() for user memories
"""
import asyncio
import sys
import os
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, SqliteDb


async def main():
    # ============================================================
    # Part 1: Basic Agent (no memory persistence)
    # ============================================================
    print("=" * 60)
    print("Part 1: Basic Agent (no memory persistence)")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(),
        add_history_to_messages=False,
    )

    r = await agent.run("李四住在北京，一家三口住大别墅")
    print(r)
    r = await agent.run("我前面问了啥")
    print(r)

    # ============================================================
    # Part 2: Agent with Persistent User Memory (Simplified API)
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 2: Agent with Persistent User Memory (Simplified API)")
    print("=" * 60)

    # Setup database
    db_file = "outputs/memory.db"
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(db_file):
        os.remove(db_file)
    db = SqliteDb(db_file=db_file)

    # Create session and user IDs
    session_id = str(uuid4())
    user_id = "john_doe@example.com"

    # Create agent with persistent memory - Simple API!
    agent = Agent(
        model=OpenAIChat(),
        db=db,
        user_id=user_id,
        session_id=session_id,
        enable_user_memories=True,
        add_history_to_messages=True,
    )

    # First conversation - Agent will remember user info
    print("\n--- First conversation ---")
    r = await agent.run("My name is John Doe and I like to hike in the mountains on weekends.")
    print(r)

    # Query about hobbies - Agent should remember
    print("\n--- Query about hobbies ---")
    r = await agent.run("What are my hobbies?")
    print(r)

    # Get user memories
    print("\n--- User memories ---")
    memories = agent.get_user_memories(user_id=user_id)
    for mem in memories:
        print(f"  Memory: {mem.memory}")

    # Update preferences
    print("\n--- Update preferences ---")
    r = await agent.run("Ok I don't like hiking anymore, I like to play soccer instead.")
    print(r)

    # Check updated memories
    print("\n--- Updated memories ---")
    memories = agent.get_user_memories(user_id=user_id)
    for mem in memories:
        print(f"  Memory: {mem.memory}")

    # ============================================================
    # Part 3: Chinese Language Example
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 3: Chinese Language Example")
    print("=" * 60)

    chinese_user_id = "zhang_san@example.com"
    session_id_2 = str(uuid4())

    agent2 = Agent(
        model=OpenAIChat(),
        db=db,
        user_id=chinese_user_id,
        session_id=session_id_2,
        enable_user_memories=True,
        add_history_to_messages=True,
    )

    print("\n--- 中文对话 ---")
    print(await agent2.run("我叫张三，是一名软件工程师，住在北京"))
    print(await agent2.run("我最喜欢的电影是《花样年华》"))
    print(await agent2.run("我喜欢打篮球和游泳"))

    print("\n--- 查询用户信息 ---")
    print(await agent2.run("你还记得我的职业和爱好吗？"))

    print("\n--- 张三的记忆 ---")
    memories = agent2.get_user_memories(user_id=chinese_user_id)
    for mem in memories:
        print(f"  Memory: {mem.memory}")

    # ============================================================
    # Part 4: Clear memories
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 4: Clear memories")
    print("=" * 60)

    print(f"Before clear: {len(agent2.get_user_memories(user_id=chinese_user_id))} memories")
    agent2.clear_user_memories(user_id=chinese_user_id)
    print(f"After clear: {len(agent2.get_user_memories(user_id=chinese_user_id))} memories")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
