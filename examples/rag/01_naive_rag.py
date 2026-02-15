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

from agentica import Agent, OpenAIChat, OpenAIEmbedding
from agentica.agent.config import ToolConfig
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica import SearchType

# Create knowledge base with PDF document
knowledge = Knowledge(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/recipes_lancedb",
        embedding=OpenAIEmbedding(),
        search_type=SearchType.vector,
    ),
)

# Load the knowledge base (recreate=True to rebuild index)
knowledge.load(recreate=True)

# Create agent with knowledge base
agent = Agent(
    model=OpenAIChat(),
    knowledge=knowledge,
    tool_config=ToolConfig(search_knowledge=True),
)

# Ask questions about the document
print("=" * 60)
print("Naive RAG Demo: Thai Recipes")
print("=" * 60)

agent.print_response_stream_sync("咋做冬阴功汤?")
