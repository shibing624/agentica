# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Basic example of using the assistant with a naive retrieval-based model.
"""

import sys

sys.path.append('..')
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
