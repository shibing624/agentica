# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from agentica import PythonAgent, OpenAIChat
from agentica.tools.jina_tool import JinaTool

m = PythonAgent(
    model=OpenAIChat(),
    description="You are a helpful ai assistant.",
    show_tool_calls=True,
    tools=[JinaTool()],
    read_chat_history=True,
    debug_mode=True,
)

prompt = f"""从arxiv搜索论文的方法是: 
https://arxiv.org/search/?query=large+model&searchtype=all&abstracts=show&order=-announced_date_first&size=50

你构造类似的请求，要求：我需要拿到与 large language model 或者 agent 或者 llm 相关的论文，3个词分别检索，打印并保存前100篇，论文需要去重，
并把每篇论文的核心信息用csv格式呈现给我，方便我后续阅读学习。

大致步骤我理解是：分别爬取arxiv的搜索结果，保存各个爬取结果的前500字符到文件，阅读并理解文件内容，提取核心信息保存到csv文件，这时做论文标题字面去重（相似度卡0.8），检查csv文件中的第一篇论文的内容。
记得在每个步骤后，打印出来，方便我理解和检查。
"""
r = m.run(prompt)
print(r)
