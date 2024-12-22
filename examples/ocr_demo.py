# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自动调用OCR工具

pip install easyocr agentica
"""

import sys

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat
from agentica.tools.ocr_tool import OcrTool

m = Agent(
    llm=AzureOpenAIChat(),
    tools=[OcrTool()],
)
prompt = """
对图片`data/chinese.jpg`进行OCR识别，只给出ocr文字就可以。
"""
r = m.run(prompt)
print(r)
