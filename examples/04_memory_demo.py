# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory demo, demonstrates how to use AgentMemory and MemoryManager with SqliteDb

Features:
    1. AgentMemory - Agent memory management with history and user memories
    2. MemoryManager - Intelligent memory retrieval with multiple methods:
        - last_n: Return the most recent memories
        - first_n: Return the oldest memories
        - keyword: Search memories by keyword matching
        - agentic: Use Agent for semantic similarity search
"""
import sys
import os
from rich.pretty import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, AgentMemory, MemoryManager, Memory
from agentica import OpenAIChat
from agentica.db.sqlite import SqliteDb

# ============================================================
# Part 1: Basic Agent Memory (no history)
# ============================================================
print("=" * 60)
print("Part 1: Basic Agent Memory (no history)")
print("=" * 60)

m = Agent(
    model=OpenAIChat(),
    add_history_to_messages=False,
    debug_mode=False
)

r = m.run("李四住在北京，一家三口住大别墅")
print(r)
r = m.run("我前面问了啥")
print(r)
pprint(m.memory.messages)
pprint(m.memory.memories)

# ============================================================
# Part 2: Agent Memory with history and user memories
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Agent Memory with history and user memories")
print("=" * 60)

db_file = "outputs/memory.db"
if os.path.exists(db_file):
    os.remove(db_file)
db = SqliteDb(db_file=db_file)

m = Agent(
    model=OpenAIChat(),
    memory=AgentMemory(db=db, create_user_memories=True),
    add_history_to_messages=True,
    debug_mode=False
)

# Test conversations
r = m.run("李四住在北京，一家三口住大别墅")
print(r)
r = m.run("李四住在北京，一家三口住大别墅，记住这个，你一句话介绍李四家庭情况")
print(r)
r = m.run("李四家里那边天气如何?是哪个气候带")
print(r)
r = m.run("我前面问了啥")
print(r)

# More test cases
print(m.run("你好，我在寻找一家位于什刹海的烤鸭店。"))
print(m.run("昨天，我和李明在下午三点见面，一起讨论了新项目。"))
print(m.run("我的名字是林瀚，我是一名软件工程师"))
print(m.run("我最喜欢的电影是《花样年华》"))
print(m.run("今天气温是18摄氏度。"))
print(m.run("hi"))
print(m.run("我前面问了啥"))

# Print memory
memory = m.memory
print("\n============ LLM Messages ============")
pprint(memory.messages)
print("\n============ Agent Memory ============")
pprint(memory.memories)

# ============================================================
# Part 3: MemoryManager - Intelligent Memory Retrieval
# ============================================================
print("\n" + "=" * 60)
print("Part 3: MemoryManager - Intelligent Memory Retrieval")
print("=" * 60)

# Create a new database for MemoryManager demo
manager_db_file = "outputs/memory_manager.db"
if os.path.exists(manager_db_file):
    os.remove(manager_db_file)
manager_db = SqliteDb(db_file=manager_db_file)

# Initialize MemoryManager
memory_manager = MemoryManager(
    model=OpenAIChat(),
    db=manager_db,
    user_id="user123",
    delete_memories=True,
    update_memories=True,
    add_memories=True,
    clear_memories=True,
)

# Add some test memories
print("\n--- Adding memories ---")
memory_manager.add_user_memory(
    memory=Memory(memory="用户喜欢 Python 编程，擅长机器学习"),
    user_id="user123"
)
memory_manager.add_user_memory(
    memory=Memory(memory="用户住在上海浦东新区"),
    user_id="user123"
)
memory_manager.add_user_memory(
    memory=Memory(memory="用户最喜欢的电影是《肖申克的救赎》"),
    user_id="user123"
)
memory_manager.add_user_memory(
    memory=Memory(memory="用户是一名高级算法工程师，在腾讯工作"),
    user_id="user123"
)
memory_manager.add_user_memory(
    memory=Memory(memory="用户喜欢打篮球和游泳"),
    user_id="user123"
)
print("Added 5 memories for user123")

# Get all memories
print("\n--- Get all user memories ---")
all_memories = memory_manager.get_user_memories(user_id="user123")
for mem in all_memories:
    print(f"  ID: {mem.id}, Memory: {mem.memory}")

# Test 1: last_n retrieval
print("\n--- Search: last_n (最近2条) ---")
recent_memories = memory_manager.search_user_memories(
    limit=2,
    retrieval_method="last_n",
    user_id="user123"
)
for mem in recent_memories:
    print(f"  Memory: {mem.memory}")

# Test 2: first_n retrieval
print("\n--- Search: first_n (最早2条) ---")
oldest_memories = memory_manager.search_user_memories(
    limit=2,
    retrieval_method="first_n",
    user_id="user123"
)
for mem in oldest_memories:
    print(f"  Memory: {mem.memory}")

# Test 3: keyword search
print("\n--- Search: keyword (关键词: Python) ---")
keyword_memories = memory_manager.search_user_memories(
    query="Python",
    retrieval_method="keyword",
    user_id="user123"
)
for mem in keyword_memories:
    print(f"  Memory: {mem.memory}")

print("\n--- Search: keyword (关键词: 电影) ---")
keyword_memories = memory_manager.search_user_memories(
    query="电影",
    retrieval_method="keyword",
    user_id="user123"
)
for mem in keyword_memories:
    print(f"  Memory: {mem.memory}")

# Test 4: agentic search (semantic similarity)
print("\n--- Search: agentic (语义搜索: 编程语言偏好) ---")
agentic_memories = memory_manager.search_user_memories(
    query="编程语言偏好",
    retrieval_method="agentic",
    user_id="user123"
)
for mem in agentic_memories:
    print(f"  Memory: {mem.memory}")

print("\n--- Search: agentic (语义搜索: 用户的工作和职业) ---")
agentic_memories = memory_manager.search_user_memories(
    query="用户的工作和职业",
    retrieval_method="agentic",
    user_id="user123"
)
for mem in agentic_memories:
    print(f"  Memory: {mem.memory}")

print("\n--- Search: agentic (语义搜索: 用户的兴趣爱好) ---")
agentic_memories = memory_manager.search_user_memories(
    query="用户的兴趣爱好",
    retrieval_method="agentic",
    user_id="user123"
)
for mem in agentic_memories:
    print(f"  Memory: {mem.memory}")

# Test 5: Delete a memory
print("\n--- Delete a memory ---")
if all_memories:
    memory_to_delete = all_memories[0]
    result = memory_manager.delete_user_memory(
        memory_id=memory_to_delete.id,
        user_id="user123"
    )
    print(f"Deleted memory: {memory_to_delete.memory}")
    print(f"Result: {result}")

# Verify deletion
print("\n--- Remaining memories after deletion ---")
remaining_memories = memory_manager.get_user_memories(user_id="user123")
for mem in remaining_memories:
    print(f"  Memory: {mem.memory}")

# Test 6: Clear all memories
print("\n--- Clear all memories ---")
memory_manager.clear_user_memories(user_id="user123")
final_memories = memory_manager.get_user_memories(user_id="user123")
print(f"Memories after clear: {len(final_memories)}")
