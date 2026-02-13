# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Hello World demo - The simplest Agent example

This is the most basic example showing how to create and run an Agent.
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
t1 = time.time()
from agentica import Agent, OpenAIChat
t2 = time.time()
print(f"import time: {t2-t1}")


async def main():
    # Create a simple agent
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        debug_mode=True
    )

    # Run the agent with a simple query
    response = await agent.run("一句话介绍北京")
    print(response)

    # You can also use print_response_stream for formatted output with streaming
    await agent.print_response_stream("一句话介绍上海")


if __name__ == "__main__":
    asyncio.run(main())
