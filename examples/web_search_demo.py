# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.tools.search_serper import SearchSerperTool

assistant = Assistant(
    llm=AzureOpenAILLM(model="gpt-4o"), tools=[SearchSerperTool()],
    add_datetime_to_instructions=True, show_tool_calls=True)
assistant.print_response("北京今天天气")
