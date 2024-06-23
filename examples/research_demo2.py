"""
The research Assistant searches for EXA for a topic
and writes an article in Markdown format.
"""

from datetime import datetime
from textwrap import dedent
import sys

sys.path.append('..')
from actionflow import Assistant
from actionflow.llm.openai_llm import OpenAILLM
from actionflow.tools.search_exa import SearchExaTool
from actionflow.tools.search_serper import SearchSerperTool

today = datetime.now().strftime("%Y-%m-%d")

m = Assistant(
    llm=OpenAILLM(model='gpt-4o'),
    tools=[SearchExaTool()],
    description="You are a senior NYT researcher writing an article on a topic.中文撰写报告",
    instructions=[
        "For the provided topic, run search.",
        "Read the results carefully and prepare a worthy article.",
        "Focus on facts and make sure to provide references. 中文撰写报告",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
    show_tool_calls=True,
    debug_mode=True,
)
m.print_response(
    "如果Eliud Kipchoge能够无限期地保持他创造记录的马拉松速度，那么他需要多少小时才能跑完地球和月球在最近接时之间的距离？请在进行计算时使用维基百科页面上的最小近地点值。将结果用中文回答")
