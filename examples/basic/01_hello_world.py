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
from agentica import Agent, OpenAIChat


async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    result = await agent.run("一句话介绍北京")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
