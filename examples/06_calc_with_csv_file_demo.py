# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CSV file calculation demo, demonstrates how to use Agent with RunPythonCodeTool for CSV data
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat, RunPythonCodeTool

m = Agent(
    name="Python Agent",
    model=OpenAIChat(),
    tools=[RunPythonCodeTool(save_and_run=True)],
    instructions=[
        "You are an expert Python programmer.",
        "Use the CSV file at: https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
        "This file contains information about movies from IMDB.",
        "Write Python code to download and analyze the data.",
    ],
    markdown=True,
)

r = m.run("What is the min rating of movies?")
print(r)
