# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Chat PDF app demo - Demonstrates a CLI-based PDF Q&A application

This example shows how to build a complete PDF chat application:
1. Load multiple documents (PDF, TXT) into a knowledge base
2. Use vector search for retrieval-augmented generation
3. Multi-turn conversation with history
4. Interactive CLI interface

Usage:
    python 03_chat_pdf.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, OpenAIEmbedding
from agentica.agent.config import ToolConfig, PromptConfig
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb

pwd_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(pwd_path)
output_dir = os.path.join(pwd_path, "outputs")

files = [
    os.path.join(data_dir, 'data', 'medical_corpus.txt'),
    os.path.join(data_dir, 'data', 'paper_sample.pdf'),
]

# Create knowledge base with multiple documents
knowledge_base = Knowledge(
    data_path=files,
    vector_db=LanceDb(
        table_name="chat_pdf",
        embedding=OpenAIEmbedding(),
        uri=os.path.join(output_dir, "chat_pdf.lancedb"),
    ),
)

# Load knowledge base (comment out after first run)
knowledge_base.load(recreate=True)

# Create agent with knowledge base and multi-turn conversation
agent = Agent(
    model=OpenAIChat(),
    knowledge=knowledge_base,
    tool_config=ToolConfig(search_knowledge=True),
    prompt_config=PromptConfig(markdown=True),
    add_history_to_messages=True,
    history_window=5,
)

print("=" * 60)
print("Chat PDF Demo: Medical Corpus & Paper Q&A")
print("=" * 60)

# Example query
agent.print_response_stream_sync("糖尿病的常见症状有哪些?")

print("\n" + "=" * 60)
print("Entering interactive mode (type 'exit' to quit)")
print("=" * 60)

# Interactive CLI
agent.cli_app()
