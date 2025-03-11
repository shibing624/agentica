# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自动去图片背景示例

pip install rembg agentica

自动安装rembg库，并执行去背景操作
"""

import sys

sys.path.append('..')
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
