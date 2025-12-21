# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Hello World demo - The simplest Agent example

This is the most basic example showing how to create and run an Agent.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat

# Create a simple agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
)

# Run the agent with a simple query
response = agent.run("一句话介绍北京")
print(response)

# You can also use print_response for formatted output with streaming
agent.print_response("一句话介绍上海", stream=True)
