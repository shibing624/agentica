# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""
import sys

sys.path.append('..')
from agentica import Message
from agentica import Agent, DeepSeekChat
from agentica.tools.search_serper_tool import SearchSerperTool

model = DeepSeekChat()
messages = [Message(role="user", content="一句话介绍林黛玉")]
r = model.response(messages)
print('model:', model)
print(r)

m = Agent(
    model=model,
    tools=[SearchSerperTool()],
    add_datetime_to_instructions=True,
)

r = m.run("下一届奥运会在哪里举办")
print(r)
