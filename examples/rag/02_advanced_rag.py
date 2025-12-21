# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Advanced RAG demo with query rewriting, hybrid search and reranking

This example shows advanced RAG features:
1. PDF document parsing
2. Query rewriting
3. Hybrid search (vector + keyword)
4. Reranking for better results

pip install similarities agentica transformers torch tantivy
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.knowledge.base import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica import SearchType
from agentica.reranker.bge import BgeReranker
from agentica.emb.text2vec_emb import Text2VecEmb

# Get the correct path to data file
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(data_dir, "data", "paper_sample.pdf")
if os.path.exists(pdf_path):
    print(f"Found PDF file: {pdf_path}")
else:
    print(f"PDF file not found: {pdf_path}")
    sys.exit(1)

# Create knowledge base with advanced features
knowledge = Knowledge(
    data_path=pdf_path,
    vector_db=LanceDb(
        table_name='paper_sample',
        uri='tmp/paper_lancedb',
        search_type=SearchType.vector,
        embedder=Text2VecEmb(model="shibing624/text2vec-base-multilingual"),
        reranker=BgeReranker(model="BAAI/bge-reranker-base"),  # Add a reranker
    )
)

# Load the knowledge base
knowledge.load()

# Create agent with knowledge base
agent = Agent(
    model=OpenAIChat(),
    knowledge=knowledge,
    search_knowledge=True,
    show_tool_calls=True,
    add_history_to_messages=True,
    markdown=True,
)

print("=" * 60)
print("Advanced RAG Demo: Paper Q&A")
print("=" * 60)

agent.print_response("Finetune LLM有啥好处?", stream=True)

print("\n" + "=" * 60)
agent.print_response("这篇论文的主要贡献是什么?", stream=True)
