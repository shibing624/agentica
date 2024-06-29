# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自动调用OCR工具

pip install easyocr actionflow
"""

import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.tools.ocr import OcrTool

m = Assistant(
    llm=AzureOpenAILLM(),
    tools=[OcrTool()],
)
prompt = """
对图片`data/chinese.jpg`进行OCR识别，只给出ocr文字就可以。
"""
m.run(prompt)
