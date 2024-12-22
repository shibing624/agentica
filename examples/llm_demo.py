# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: llm demo
"""

import sys

sys.path.append('..')
from agentica import Message, YiChat, AzureOpenAIChat, DeepSeekChat, OpenAIChat, MoonshotChat

query = "一句话介绍林黛玉"
model = AzureOpenAIChat()
print(model)
messages = [Message(role="user", content=query)]
r = model.response(messages)
print(r)
model = OpenAIChat()
print(model)
print(model.response(messages=messages))
model = DeepSeekChat()
print(model)
print(model.response(messages=messages))

