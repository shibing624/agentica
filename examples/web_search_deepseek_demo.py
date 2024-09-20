# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""
import sys

sys.path.append('..')
from agentica.message import Message
from agentica import Assistant, DeepseekLLM
from agentica.tools.search_serper import SearchSerperTool
from agentica.tools.file import FileTool

llm = DeepseekLLM()

print('llm:', llm)
messages = [Message(role="user", content="一句话介绍林黛玉")]
llm_r = llm.response(messages)
print(llm_r)

m = Assistant(
    llm=llm,
    tools=[SearchSerperTool(), FileTool()],
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    read_chat_history=True,
    debug_mode=True,
)

r = m.run("北京最近的top3新闻")
print(r, "".join(r))
r = m.run("搜索下今天北京温度是多少度？")
print(r, "".join(r))
r = m.run("总结我们的对话。", stream=False, print_output=False)
print(r)
