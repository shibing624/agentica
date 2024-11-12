# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo, using SearchSerperTool(google) to search the web.
"""

import sys

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM, DeepseekLLM, YiLLM, MoonshotLLM

m = Assistant(
    llm=YiLLM(api_key='your_yi_api_key'),
    debug_mode=True,
)
r = m.run("一句话介绍林黛玉")
print(r, "".join(r))
