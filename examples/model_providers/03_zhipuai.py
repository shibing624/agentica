# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ZhipuAI model demo - Demonstrates using ZhipuAI (GLM) models

This example shows how to use ZhipuAI models with Agentica.
Requires: ZHIPUAI_API_KEY environment variable
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAI, UserMessage


def main():
    # Example 1: Direct model usage
    print("=" * 60)
    print("Example 1: Direct ZhipuAI Model Usage")
    print("=" * 60)
    
    model = ZhipuAI()
    print(f"Model: {model}")
    
    messages = [UserMessage("一句话介绍北京")]
    response = model.response(messages)
    print(f"Response: {response}")
    
    # Example 2: Streaming
    print("\n" + "=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)
    
    stream_messages = [UserMessage("一句话介绍上海")]
    print("Streaming: ", end="", flush=True)
    for chunk in model.response_stream(stream_messages):
        if chunk.reasoning_content:
            print(chunk.reasoning_content, end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
    
    # Example 3: Agent with ZhipuAI
    print("\n" + "=" * 60)
    print("Example 3: Agent with ZhipuAI")
    print("=" * 60)
    
    agent = Agent(
        model=ZhipuAI(),
    )
    
    agent.print_response_sync("一句话简单介绍一下《红楼梦》", stream=True)


if __name__ == "__main__":
    main()
