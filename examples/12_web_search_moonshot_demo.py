# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo with Moonshot, demonstrates using SearchSerperTool with Moonshot model
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, Moonshot, BaiduSearchTool

model = Moonshot()

m = Agent(
    model=model,
    tools=[BaiduSearchTool()],
    add_datetime_to_instructions=True,
)

r = m.run("下一届奥运会在哪里举办")
print(r)
