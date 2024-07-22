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
from agentica.vectordb.memorydb import MemoryDb
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge_base = KnowledgeBase(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=MemoryDb(embedder=Text2VecEmb()),
)
# Comment after first run
knowledge_base.load(recreate=True)

assistant = Assistant(
    knowledge_base=knowledge_base,
    # Adds references to the prompt.
    add_references_to_prompt=True,
    description="中文回答",
    debug_mode=False,
)
r = assistant.run("如何做泰国咖喱")
print(r, "".join(r))
