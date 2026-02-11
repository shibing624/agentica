# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Naive RAG demo - Demonstrates basic retrieval-augmented generation

This example shows how to:
1. Create a knowledge base from documents
2. Use vector search for retrieval
3. Answer questions based on retrieved content
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica import SearchType
from agentica.emb.text2vec_emb import Text2VecEmb

# Create knowledge base with PDF document
knowledge = Knowledge(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/recipes_lancedb",
        embedder=Text2VecEmb(),
        search_type=SearchType.vector,
    ),
)

# Load the knowledge base (recreate=True to rebuild index)
knowledge.load(recreate=True)

# Create agent with knowledge base
agent = Agent(
    model=OpenAIChat(),
    knowledge=knowledge,
    search_knowledge=True,  # Enable agentic RAG
)

# Ask questions about the document
print("=" * 60)
print("Naive RAG Demo: Thai Recipes")
print("=" * 60)

agent.print_response_sync("咋做冬阴功汤?", stream=True)

print("\n" + "=" * 60)
agent.print_response_sync("有哪些泰国甜点的做法?", stream=True)
