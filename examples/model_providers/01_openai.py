# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OpenAI model demo - Demonstrates using OpenAI models

This example shows how to use OpenAI models with Agentica.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, Message
from agentica import OpenAIChat


def main():
    # Example 1: Direct model usage
    print("=" * 60)
    print("Example 1: Direct OpenAI Model Usage")
    print("=" * 60)
    
    model = OpenAIChat(id="gpt-4o-mini")
    print(f"Model: {model}")
    
    messages = [Message(role="user", content="一句话介绍北京")]
    response = model.response(messages)
    print(f"Response: {response}")
    
    # Example 2: With streaming
    print("\n" + "=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)
    
    print("Streaming: ", end="")
    for chunk in model.response_stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
    
    # Example 3: Agent with OpenAI
    print("\n" + "=" * 60)
    print("Example 3: Agent with OpenAI")
    print("=" * 60)
    
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions="You are a helpful assistant.",
    )
    
    agent.print_response("介绍一下人工智能的发展历史", stream=True)


if __name__ == "__main__":
    main()
