# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Context passing demo - Demonstrates how to pass context to agents and tools

This example shows:
1. Passing context to agent with add_context=True
2. Context is automatically added to user messages
3. Sharing context between multiple agents
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


def main():
    # Create user context as a dictionary (required format)
    user_context = {
        "user_id": "user_123",
        "user_name": "张三",
        "preferences": {"language": "zh", "theme": "dark", "notifications": True}
    }
    
    # Example 1: Agent with context (add_context=True adds context to user messages)
    print("=" * 60)
    print("Example 1: Context in User Messages")
    print("=" * 60)
    
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        context=user_context,
        add_context=True,  # This adds context to user messages automatically
        instructions="你是一个个性化助手。请用中文回复，并根据上下文中的用户信息提供个性化服务。",
    )
    
    agent.print_response("请问候我并告诉我我的偏好设置")
    
    # Example 2: Multiple agents sharing context
    print("\n" + "=" * 60)
    print("Example 2: Shared Context Between Agents")
    print("=" * 60)
    
    # Agent 1: Collects information
    collector = Agent(
        name="Information Collector",
        model=OpenAIChat(id="gpt-4o-mini"),
        context=user_context,
        add_context=True,
        instructions="你是一个信息收集助手。请用中文回复。",
    )
    
    # Agent 2: Processes information
    processor = Agent(
        name="Information Processor",
        model=OpenAIChat(id="gpt-4o-mini"),
        context=user_context,
        add_context=True,
        instructions="你是一个信息处理助手。请用中文回复，并根据用户偏好给出建议。",
    )
    
    # Collect information
    info = collector.run("根据上下文中的用户信息生成用户基本信息摘要")
    print(f"Collector: {info.content}")
    
    # Process information
    result = processor.run(f"处理以下信息并给出个性化建议: {info.content}")
    print(f"Processor: {result.content}")


if __name__ == "__main__":
    main()
