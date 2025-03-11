# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Translate a paper to another language.
"""

import sys

from loguru import logger

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.knowledge import Knowledge

knowledge_base = Knowledge(
    # data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    data_path=["data/paper_sample.pdf"],
)
# Comment after first run
knowledge_base.load(recreate=True)

docs = []
for document_list in knowledge_base.document_lists:
    docs.extend(document_list)
docs = docs[:2]
logger.info(f'docs size: {len(docs)}, top3: {docs[:3]}')

assistant = Agent(
    model=OpenAIChat(id='gpt-4o'),
    description="""现在你要解释一篇专业的技术文章成简体中文给大学生阅读。

规则：
- 翻译时要准确传达学术论文的事实和背景，同时风格上保持为通俗易懂并且严谨的科普文风格。
- 保留特定的英文术语、数字或名字，并在其前后加上空格，例如："中文 EN 中文"，"不超过 10 秒"。
- 即使上意译也要保留术语，例如 FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon 等。
- 保留引用的论文；同时也要保留针对图例的引用，例如保留 Figure 1 并翻译为图 1。
- 全角括号换成半角括号，并在左括号前面加半角空格，右括号后面加半角空格。
- 输入格式为Markdown格式，输出格式也必须保留原始Markdown格式


现在有两个角色：
- 英语老师，精通中英文，能精确的理解英文并用中文表达
- 中文老师，精通中文，擅长按照中文使用喜欢撰写通俗易懂的科普文

翻译文章，每一步都必须遵守以上规则，打印每一步的输出结果：
Step 1：现在你是英语翻译老师，精通中英文，对原文按照字面意思翻译为中文（直译），务必遵守原意，翻译时保持原始的段落结构，不要合并分段
Step 2：扮演中文老师，精通中文，擅长写通俗易懂的科普文章，对英语老师翻译的中文内容重新意译为中文，遵守原意的前提下让内容更通俗易懂，符合中文表达习惯，但不要增加和删减内容，保持原始分段

输出格式：
```
# Step 1: 英语老师翻译
中文直译结果

# Step 2: 中文老师翻译
中文意译结果
```

你翻译下面的文本。""",
)

full_translated = ""
for doc in docs:
    q = doc.content
    res = assistant.run(q)
    r = res.content
    r = r.split("Step 2: 中文老师翻译")[-1]
    logger.info(f"input: {q}")
    logger.info(f"output: {r}")
    full_translated += r

logger.info(f"full_translated: {full_translated}")
