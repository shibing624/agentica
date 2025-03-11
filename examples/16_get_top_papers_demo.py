# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from agentica import OpenAIChat, Agent, PythonAgent
from agentica.tools.jina_tool import JinaTool
from agentica.tools.file_tool import FileTool
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.text_analysis_tool import TextAnalysisTool

m = Agent(
    model=OpenAIChat(model='gpt-4o'),
    description="You are a helpful ai assistant.",
    tools=[JinaTool(), SearchSerperTool(), FileTool(), TextAnalysisTool()],
    debug_mode=True,
)

query = """
https://huggingface.co/papers?date=2024-11-15 解析该网页，理解内容，并提取所有论文的标题、作者、摘要、论文的链接、pdf链接等信息，保存到本地文件papers.json中，只要点赞超过10个的论文。
注意：pdf链接需要通过谷歌搜索论文题目获取，也保存到json中。

大概流程如下：

1. **解析网页内容**：
   - 使用 `jina_url_reader` 工具来解析给定的URL，获取网页的全部内容。

2. **提取论文信息**：
   - 解析得到的网页内容，使用大模型只提取点赞数大于10的论文的标题、作者、摘要、论文的链接、点赞数量等信息。

3. **获取PDF链接**：
   - 对于各论文，使用 `SearchSerperTool` 工具通过标题搜索PDF链接。

4. **保存信息到本地文件**：
   - 将最终的结果保存到 `papers.json` 文件中。

"""
r = m.run(query)
print(r)
