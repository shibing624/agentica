# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""
import sys

sys.path.append('..')
from agentica.message import Message
from agentica import Assistant, MoonshotLLM
from agentica.tools.search_serper import SearchSerperTool
from agentica.tools.file import FileTool

llm = MoonshotLLM()

print('llm:', llm)
messages = [Message(role="user", content="一句话介绍林黛玉")]
llm_r = llm.response(messages)
print(llm_r)

m = Assistant(
    llm=llm,
    tools=[SearchSerperTool(), FileTool()],
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    add_chat_history_to_messages=True,
    debug_mode=True,
)

r = m.run("北京最近的top3新闻", stream=False, print_output=False)
print(r)
r = m.run("一句话介绍北京", stream=False, print_output=False)
print(r)
