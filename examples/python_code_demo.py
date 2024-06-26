# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from actionflow.file.csv import CsvFile
from actionflow.llm.openai_llm import OpenAILLM
from actionflow.python_assistant import PythonAssistant
from actionflow.llm.azure_llm import AzureOpenAILLM
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

python_assistant.print_response("What is the min rating of movies?")
