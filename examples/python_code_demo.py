# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from agentica.file.csv import CsvFile
from agentica.llm.openai_llm import OpenAILLM
from agentica.python_assistant import PythonAssistant
from agentica.llm.azure_llm import AzureOpenAILLM
python_assistant = PythonAssistant(
    llm=AzureOpenAILLM(),
    files=[
        CsvFile(
            data_path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
            description="Contains information about movies from IMDB.",
        )
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

r = python_assistant.run("What is the min rating of movies?")
print("".join(r))
