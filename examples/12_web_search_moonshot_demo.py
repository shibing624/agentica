# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""
import sys

sys.path.append('..')
from agentica import Agent, Moonshot
from agentica.tools.search_serper_tool import SearchSerperTool

model = Moonshot()

m = Agent(
    model=model,
    tools=[SearchSerperTool()],
    add_datetime_to_instructions=True,
)

r = m.run("下一届奥运会在哪里举办")
print(r)
