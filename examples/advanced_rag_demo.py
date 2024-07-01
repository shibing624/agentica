# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Advanced RAG demo
实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）
"""

import sys

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM
from agentica.documents import TextDocuments
from agentica.vectordb.lancedb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge_base = TextDocuments(
    data_path="data/medical_corpus.txt",
    vector_db=LanceDb(
        embedder=Text2VecEmb(),
    ),
)
# Load the knowledge base
knowledge_base.load(recreate=True)

assistant = Assistant(
    llm=AzureOpenAILLM(),
    knowledge_base=knowledge_base,
    # The add_references_to_prompt will update the prompt with references from the knowledge base.
    add_references_to_prompt=True,
)
assistant.run("肛门病变可能是什么疾病的症状?")
