# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Reference in prompt RAG demo, demonstrates using add_context for RAG
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica.knowledge.base import Knowledge
from agentica.vectordb.memory_vectordb import InMemoryVectorDb
from agentica.db.sqlite import SqliteDb
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge = Knowledge(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=InMemoryVectorDb(embedder=Text2VecEmb()),
)
# Comment after first run
knowledge.load()

# Create a SQLite database for the assistant's data
db = SqliteDb(db_file="outputs/ThaiRecipes.db")

m = Agent(
    knowledge=knowledge,
    db=db,
    add_context=True,  # Use RAG by adding references from the knowledge base to the user prompt.
    description="中文回答",
)
m.print_response("如何做泰国咖喱")
