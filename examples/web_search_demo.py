# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""
import sys

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat
from agentica.tools.search_serper_tool import SearchSerperTool

m = Agent(
    llm=AzureOpenAIChat(model="gpt-4o"),
    tools=[SearchSerperTool()],
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    # Enable the assistant to read the chat history
    read_chat_history=True,
    debug_mode=True,
)
r = m.run("一句话介绍林黛玉")
print(r)
r = m.run("上海今天适合穿什么衣服")
print(r)
r = m.run("总结前面的问答")
print(r)
