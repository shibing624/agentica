# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo with ZHIPUAI
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, ZhipuAI, BaiduSearchTool


m = Agent(
    model=ZhipuAI(),
    tools=[BaiduSearchTool()],
    add_datetime_to_instructions=True,
    debug=True,
)

r = m.run("下一届奥运会在哪里举办")
print(r)
