# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Advanced RAG demo

实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）

pip install similarities agentica transformers torch tantivy
"""
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.knowledge.base import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica import SearchType
from agentica.reranker.bge import BgeReranker
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge = Knowledge(
    data_path="data/paper_sample.pdf",
    vector_db=LanceDb(
        table_name='paper_sample',
        uri='tmp/lancedb',
        search_type=SearchType.vector,
        embedder=Text2VecEmb(model="shibing624/text2vec-base-multilingual"),
        reranker=BgeReranker(model="BAAI/bge-reranker-base"),  # Add a reranker
    )
)
# Load the knowledge base
knowledge.load()

m = Agent(
    model=OpenAIChat(),
    knowledge=knowledge,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)
m.print_response("Finetune LLM有啥好处?", stream=True)
