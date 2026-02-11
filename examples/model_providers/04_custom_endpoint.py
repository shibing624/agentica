# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom LLM endpoint demo - Demonstrates using OpenAILike for custom endpoints

This example shows how to use custom LLM endpoints that are compatible with OpenAI API.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, OpenAILike, Message


def main():
    # Example 1: Custom endpoint with OpenAILike
    print("=" * 60)
    print("Example 1: Custom Endpoint with OpenAILike")
    print("=" * 60)
    
    # Replace with your actual API key and base URL
    model = OpenAILike(
        id='your-model-id',
        api_key='your_api_key',
        base_url='your_base_url'
    )
    
    print(f"Model: {model}")
    
    # Note: This will fail without valid credentials
    # Uncomment to test with your own endpoint
    # messages = [Message(role="user", content="一句话介绍林黛玉")]
    # response = model.response(messages)
    # print(f"Response: {response}")
    
    # Example 2: Agent with custom endpoint
    print("\n" + "=" * 60)
    print("Example 2: Agent with Custom Endpoint")
    print("=" * 60)
    
    # Using OpenAI as fallback for demo
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        debug_mode=True,
    )
    
    agent.print_response_sync("你是谁？详细介绍自己", stream=True)
    
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
    agent.print_response_sync("Hello!")
    """)


if __name__ == "__main__":
    main()
