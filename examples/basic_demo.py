# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.tools.search_serper import SearchSerperTool

m = Assistant(
    llm=AzureOpenAILLM(),
    description="You are a helpful ai assistant.",
    show_tool_calls=True,
    # Enable the assistant to search the knowledge base
    search_knowledge=False,
    tools=[SearchSerperTool()],
    # Enable the assistant to read the chat history
    read_chat_history=True,
    markdown=True,
    debug_mode=True,
)
print("LLM:", m.llm)
m.run("一句话介绍林黛玉")
m.run("北京最近的新闻", stream=True, print_output=True)
print(m.run("总结前面的问答", stream=False, print_output=False))
