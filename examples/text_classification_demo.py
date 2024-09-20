# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自动调用LLM训练分类模型

pip install pytextclassifier agentica
"""

import sys

sys.path.append('..')
from agentica import PythonAssistant, AzureOpenAILLM
from agentica.tools.jina import JinaTool
from agentica.tools.url_crawler import UrlCrawlerTool
from agentica.tools.search_serper import SearchSerperTool
m = PythonAssistant(
    llm=AzureOpenAILLM(),
    description="You are a helpful ai assistant.",
    tools=[JinaTool(), SearchSerperTool(), UrlCrawlerTool()],
    debug_mode=True,
)
prompt = """
请你帮我完成以下任务：
训练fasttext分类模型，并预测看下模型训练是否成功。

我有文件`data/thucnews_train_1k.txt`，文件中的每一行包含一个文本记录和一个类别标签，类别标签是文本记录的分类结果，你能读取该文件，它就在本地。
用python库`pytextclassifier`训练分类模型，pytextclassifier的用法可以参考github里面shibing624/pytextclassifier 官方库的readme，
必须说明的是，pytextclassifier本地已经安装了，你可以直接调用。pytextclassifier的用法你谷歌搜索下github链接，再读一下readme，就能明白了。
训练5个epochs就可以了，训练模型后，试着跑下`data/thucnews_train_1k.txt`文件中前10条记录的分类结果，输出预测标签结果和原标签，方便我check对比效果。
"""
r = m.run(prompt)
print("".join(r))
