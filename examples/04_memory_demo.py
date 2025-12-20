# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory demo - demonstrates persistent user memory with Agent

This example shows how to use persistent memory with an Agent.
After each run, user memories are automatically created/updated.

Features:
    1. Simple API: Just set `enable_user_memories=True` and provide a `db`
    2. Agent and AgentMemory share the same database
    3. User memories persist across sessions
    4. Automatic memory extraction from conversations
"""
import sys
import os
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat, SqliteDb

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

r = agent.run("李四住在北京，一家三口住大别墅")
print(r)
r = agent.run("我前面问了啥")
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
# Just set enable_user_memories=True and provide db
agent = Agent(
    model=OpenAIChat(),
    db=db,  # Database for storing memories
    user_id=user_id,  # User ID for personalized memories
    session_id=session_id,  # Session ID for conversation history
    enable_user_memories=True,  # Enable automatic memory creation
    add_history_to_messages=True,  # Enable conversation history
)

# First conversation - Agent will remember user info
print("\n--- First conversation ---")
r = agent.run("My name is John Doe and I like to hike in the mountains on weekends.")
print(r)

# Query about hobbies - Agent should remember
print("\n--- Query about hobbies ---")
r = agent.run("What are my hobbies?")
print(r)

# Get user memories using the new API
print("\n--- User memories (via agent.get_user_memories) ---")
memories = agent.get_user_memories(user_id=user_id)
for mem in memories:
    print(f"  Memory: {mem.memory}")

# Update preferences
print("\n--- Update preferences ---")
r = agent.run("Ok I don't like hiking anymore, I like to play soccer instead.")
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

# New user
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
print(agent2.run("我叫张三，是一名软件工程师，住在北京"))
print(agent2.run("我最喜欢的电影是《花样年华》"))
print(agent2.run("我喜欢打篮球和游泳"))

print("\n--- 查询用户信息 ---")
print(agent2.run("你还记得我的职业和爱好吗？"))

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

# ============================================================
# Part 5: Advanced - Using AgentMemory directly (for more control)
# ============================================================
print("\n" + "=" * 60)
print("Part 5: Advanced - Using AgentMemory directly")
print("=" * 60)

from agentica import AgentMemory, MemoryManager, Memory

# Create a new database
advanced_db_file = "outputs/advanced_memory.db"
if os.path.exists(advanced_db_file):
    os.remove(advanced_db_file)
advanced_db = SqliteDb(db_file=advanced_db_file)

# Method 1: Use AgentMemory.with_db() factory method
memory = AgentMemory.with_db(
    db=advanced_db,
    user_id="advanced_user",
    num_memories=10,  # Limit number of memories to retrieve
)

agent3 = Agent(
    model=OpenAIChat(),
    memory=memory,
    add_history_to_messages=True,
)

print("\n--- Using AgentMemory.with_db() ---")
print(agent3.run("我是高级用户，喜欢深度学习和NLP"))

print("\n--- 高级用户的记忆 ---")
memories = agent3.get_user_memories(user_id="advanced_user")
for mem in memories:
    print(f"  Memory: {mem.memory}")

# Method 2: Use MemoryManager for programmatic memory management
print("\n--- Using MemoryManager for programmatic control ---")
memory_manager = MemoryManager(
    model=OpenAIChat(),
    db=advanced_db,
    user_id="programmatic_user",
)

# Add memories programmatically
memory_manager.add_user_memory(
    memory=Memory(memory="用户偏好 Python 编程"),
    user_id="programmatic_user"
)
memory_manager.add_user_memory(
    memory=Memory(memory="用户在google工作"),
    user_id="programmatic_user"
)

# Search memories
print("\n--- Search memories by keyword ---")
results = memory_manager.search_user_memories(
    query="Python",
    retrieval_method="keyword",
    user_id="programmatic_user"
)
for mem in results:
    print(f"  Memory: {mem.memory}")

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
