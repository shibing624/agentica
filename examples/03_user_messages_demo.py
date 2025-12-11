# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: User messages demo, demonstrates how to use Agent with message list input
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat

r = Agent(
    model=OpenAIChat(model="gpt-3.5-turbo", stop="</answer>"),
    debug_mode=True,
).run(
    messages=[
        {"role": "user", "content": "What is the color of a banana? Provide your answer in the xml tag <answer>."},
        {"role": "assistant", "content": "<answer>"},
    ],
)
print(r)
