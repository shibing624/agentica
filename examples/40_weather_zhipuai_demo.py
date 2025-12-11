# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Weather demo with ZhipuAI, demonstrates using ZhipuAI model with WeatherTool
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Message
from agentica import Agent, ZhipuAI, WeatherTool

model = ZhipuAI(id='glm-4-plus')
messages = [Message(role="user", content="一句话介绍林黛玉")]
r = model.response(messages)
print('model:', model)
print(r)

m = Agent(
    model=model,
    tools=[WeatherTool()],
    add_datetime_to_instructions=True,
    debug_mode=True
)

m.print_response("明天北京天气咋样")
