# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepSeek model demo - Demonstrates using DeepSeek models

This example shows how to use DeepSeek models with Agentica.
Requires: DEEPSEEK_API_KEY environment variable
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, Message, DeepSeekChat


async def main():
    # Example 1: Direct model usage
    print("=" * 60)
    print("Example 1: Direct DeepSeek Model Usage")
    print("=" * 60)
    
    model = DeepSeekChat()
    print(f"Model: {model}")
    
    messages = [Message(role="user", content="一句话介绍北京")]
    response = await model.response(messages)
    print(f"Response: {response}")
    
    # Example 2: Agent with DeepSeek
    print("\n" + "=" * 60)
    print("Example 2: Agent with DeepSeek")
    print("=" * 60)
    
    agent = Agent(
        model=DeepSeekChat(),
        instructions="You are a helpful assistant. Please respond in Chinese.",
    )
    
    await agent.print_response_stream("什么是深度学习?")
    
    # Example 3: DeepSeek Reasoner (if available)
    print("\n" + "=" * 60)
    print("Example 3: DeepSeek Reasoner")
    print("=" * 60)
    
    try:
        reasoner = DeepSeekChat(id="deepseek-reasoner")
        agent_reasoner = Agent(
            model=reasoner,
            instructions="You are a reasoning assistant. Think step by step.",
        )
        await agent_reasoner.print_response_stream("计算 15 * 23 + 47")
    except Exception as e:
        print(f"DeepSeek Reasoner not available: {e}")


if __name__ == "__main__":
    asyncio.run(main())
