# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Hello World demo - The simplest Agent example

This is the most basic example showing how to create and run an Agent.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
t1 = time.time()
from agentica import Agent, OpenAIChat
t2 = time.time()
print(f"import time: {t2-t1}")

# Create a simple agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    debug=True
)

# Run the agent with a simple query
response = agent.run("一句话介绍北京")
print(response)

# You can also use print_response for formatted output with streaming
agent.print_response("一句话介绍上海", stream=True)
