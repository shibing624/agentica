# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom LLM endpoint demo - Demonstrates using OpenAILike for custom endpoints

This example shows how to use custom LLM endpoints that are compatible with OpenAI API.
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, OpenAILike, Message


async def main():
    # Example 1: Custom endpoint with OpenAILike
    print("=" * 60)
    print("Example 1: Custom Endpoint with OpenAILike")
    print("=" * 60)
    
    model = OpenAILike(
        id='your-model-id',
        api_key='your_api_key',
        base_url='your_base_url'
    )
    
    print(f"Model: {model}")
    
    # Example 2: Agent with custom endpoint
    print("\n" + "=" * 60)
    print("Example 2: Agent with Custom Endpoint")
    print("=" * 60)
    
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        debug=True,
    )
    
    await agent.print_response_stream("你是谁？详细介绍自己")
    
    # Example 3: Local model (e.g., Ollama)
    print("\n" + "=" * 60)
    print("Example 3: Local Model (Ollama)")
    print("=" * 60)
    
    print("""
To use a local Ollama model:

    from agentica import OllamaChat
    
    model = OllamaChat(
        id="llama2",
        host="http://localhost:11434"
    )
    
    agent = Agent(model=model)
    await agent.print_response("Hello!")
    """)


if __name__ == "__main__":
    asyncio.run(main())
