# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: User prompt demo, demonstrates how to use Agent with user messages
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica import OpenAIChat, AzureOpenAIChat

m = Agent(
    model=OpenAIChat(id="o4-mini"),
    debug=True,
)

m.print_response(message="你是谁？详细介绍自己，你的知识库到哪天",stream=True)
m.print_response(message="一句话介绍北京", stream=True)
