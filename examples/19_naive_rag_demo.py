# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Naive RAG demo, demonstrates basic retrieval-augmented generation with LanceDb
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica import SearchType
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge = Knowledge(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/recipes_lancedb",
        embedder=Text2VecEmb(),
        search_type=SearchType.vector,
    ),
)
# Load the knowledge base
knowledge.load(recreate=True)

m = Agent(
    model=OpenAIChat(),
    knowledge=knowledge,
    search_knowledge=True,  # enable agentic RAG
    show_tool_calls=True,
)
m.print_response("咋做冬阴功汤?", stream=True)
