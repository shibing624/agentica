# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Use a custom function to generate references for RAG.

You can use the custom_references_function to generate references for the RAG model.
The function takes a query and returns a list of references from the knowledge base.

usage:
python memorydb_rag_demo.py
"""

import sys

sys.path.append('..')
from agentica import Agent
from agentica.knowledge.base import Knowledge
from agentica.vectordb.memory_vectordb import MemoryVectorDb
from agentica import SqlAgentStorage
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge = Knowledge(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=MemoryVectorDb(embedder=Text2VecEmb()),
)
# Comment after first run
knowledge.load()

# Create a SQL storage for the assistant's data
storage = SqlAgentStorage(table_name="ThaiRecipes", db_file="outputs/ThaiRecipes.db")
storage.create()

m = Agent(
    knowledge=knowledge,
    storage=storage,
    add_context=True,  # Use RAG by adding references from the knowledge base to the user prompt.
    description="中文回答",
)
m.print_response("如何做泰国咖喱")
