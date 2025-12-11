# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DALL-E image generation demo, demonstrates how to create images using DalleTool
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat
from agentica.tools.dalle_tool import DalleTool

m = Agent(tools=[DalleTool()], show_tool_calls=True)
r = m.run("画一匹斑马在太空行走")
print(r)
