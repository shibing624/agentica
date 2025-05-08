# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat, WeatherTool
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.edit_tool import EditTool
from agentica.tools.code_tool import CodeTool
from agentica.tools.workspace_tool import WorkspaceTool

m = Agent(
    model=OpenAIChat(model="gpt-4o"),
    tools=[SearchSerperTool(), WeatherTool(), EditTool(), CodeTool(), WorkspaceTool()],
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    read_chat_history=True,
    debug_mode=False,
)
r = m.run("一句话介绍林黛玉")
print(r)
r = m.run("上海今天适合穿什么衣服")
print(r)
r = m.run("总结前面的问答")
print(r)
m.print_response("当前文件路径是？新增a.py，并在里面加入冒泡排序的python代码")
m.print_response("查找当前路径下 a.py，并解释代码")
m.print_response("给出a.py的测试用例,并删除a.py文件")