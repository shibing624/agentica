# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Remove image background demo, demonstrates automatic background removal using rembg

pip install rembg agentica
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import PythonAgent, OpenAIChat

m = PythonAgent(
    model=OpenAIChat(),
    pip_install=True,
    debug_mode=True,
)
prompt = """
This is a image, you need to use python toolkit rembg to remove the background of the image and save the result. 
image path: data/chinese.jpg ; save path: chinese_remove_bg.png
"""
r = m.run(prompt)
print(r)
