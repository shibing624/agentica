# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-turn conversation demo - Demonstrates conversation history management

This example shows how to:
1. Enable conversation history
2. Reference previous messages
3. Clear conversation history
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat

# Example 1: Without history (no context)
print("=" * 60)
print("Example 1: Without Conversation History")
print("=" * 60)

agent_no_history = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_messages=False,
)

print(agent_no_history.run("我叫张三，住在北京"))
print(agent_no_history.run("我叫什么名字?"))  # Won't remember

# Example 2: With history enabled
print("\n" + "=" * 60)
print("Example 2: With Conversation History")
print("=" * 60)

agent_with_history = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_messages=True,  # Enable conversation history
    read_chat_history=True,  # Allow agent to read chat history
)

print(agent_with_history.run("我叫李四，是一名软件工程师"))
print(agent_with_history.run("我叫什么名字？做什么工作?"))  # Will remember

# Example 3: Multi-turn with context
print("\n" + "=" * 60)
print("Example 3: Multi-turn Conversation")
print("=" * 60)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_messages=True,
    instructions="你是一个友好的助手，请记住用户告诉你的信息。",
)

# First turn
response1 = agent.run("我最喜欢的颜色是蓝色")
print(f"Turn 1: {response1}")

# Second turn
response2 = agent.run("我喜欢吃川菜")
print(f"Turn 2: {response2}")

# Third turn - reference previous context
response3 = agent.run("总结一下你了解到的关于我的信息")
print(f"Turn 3: {response3}")

# Example 4: Check conversation history
print("\n" + "=" * 60)
print("Example 4: View Conversation History")
print("=" * 60)

messages = agent.memory.get_messages()
print(f"Total messages in history: {len(messages)}")
for msg in messages[-4:]:  # Show last 4 messages
    role = msg.get("role", "unknown")
    content = msg.get("content", "")[:50]
    print(f"  [{role}]: {content}...")
