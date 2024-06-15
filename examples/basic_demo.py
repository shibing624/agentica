# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from actionflow import Assistant
from actionflow.llm.openai_llm import OpenAILLM
from actionflow.tools.search_serper import SearchSerperTool

m = Assistant(
    llm=OpenAILLM(model='gpt-3.5-turbo-1106'),
    description="You are a helpful ai assistant.",
    instructions=["食谱应该少于5种材料。"],
    show_tool_calls=True,
    # Enable the assistant to search the knowledge base
    search_knowledge=True,
    tools=[SearchSerperTool()],
    # Enable the assistant to read the chat history
    read_chat_history=True,
    markdown=True,
    debug_mode=True,
)
print("LLM:", m.llm)
# -*- Print a response to the console -*-
m.print_response("分享一个减肥早餐食谱")
m.print_response("当前最新最流行的燃脂减肥餐单")
m.print_response("我前面问了啥")
