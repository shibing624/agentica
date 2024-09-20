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
from agentica import Assistant
from agentica.knowledge.knowledge_base import KnowledgeBase
from agentica.vectordb.memory_vectordb import MemoryVectorDb
from agentica.storage.sqlite_storage import SqlAssistantStorage
from agentica.emb.text2vec import Text2VecEmb

knowledge_base = KnowledgeBase(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=MemoryVectorDb(embedder=Text2VecEmb()),
)
# Comment after first run
knowledge_base.load()

# Create a SQL storage for the assistant's data
storage = SqlAssistantStorage(table_name="ThaiRecipes", db_file="outputs/ThaiRecipes.db")
storage.create()

assistant = Assistant(
    knowledge_base=knowledge_base,
    storage=storage,
    add_references_to_prompt=True,  # Use RAG
    description="中文回答",
    debug_mode=True,
)
r = assistant.run("如何做泰国咖喱")
print(r, "".join(r))
assistant.storage.export("outputs/ThaiRecipes_storage.csv")
