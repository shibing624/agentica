# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using free zhipuai search to search the web.
"""
import sys

sys.path.append('..')
from agentica import Message
from agentica import Agent, ZhipuAI
from agentica.tools.web_search_pro_tool import WebSearchProTool

model = ZhipuAI(id='glm-4-flash')
messages = [Message(role="user", content="一句话介绍林黛玉")]
r = model.response(messages)
print('model:', model)
print(r)

m = Agent(
    model=model,
    tools=[WebSearchProTool()],
    add_datetime_to_instructions=True,
)

r = m.run("最新奥斯卡奖获奖电影是啥")
print(r)
