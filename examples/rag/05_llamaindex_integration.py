# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: RAG integrated with LlamaIndex demo - Using LlamaIndex retriever

This example shows how to integrate Agentica with LlamaIndex's retrieval system
for retrieval-augmented generation.

pip install llama-index
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.knowledge.llamaindex_knowledge import LlamaIndexKnowledge

# Check if llama_index is installed
try:
    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:
    print("Please install llama-index: pip install llama-index")
    sys.exit(1)

pwd_path = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("RAG with LlamaIndex Integration Demo")
    print("=" * 60)

    # Define path to document
    file_path = os.path.join(pwd_path, "../data/news_docs.txt")
    # Check if data file exists
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}")
        print("Please ensure the data file exists.")
        return

    print(f"\n1. Loading document: {file_path}")
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    print(f"   Loaded {len(documents)} documents")

    print("2. Splitting documents into nodes...")
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   Created {len(nodes)} nodes")

    print("3. Creating vector store index...")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    print("4. Creating retriever...")
    retriever = VectorIndexRetriever(index)

    print("5. Creating LlamaIndexKnowledge and Agent...")
    knowledge = LlamaIndexKnowledge(retriever=retriever)
    agent = Agent(
        knowledge=knowledge,
        search_knowledge=True,
        debug_mode=True,
        show_tool_calls=True
    )

    print("\n" + "=" * 60)
    print("Asking question: 2023年全国田径锦标赛在哪里举办的?")
    print("=" * 60)

    response = agent.run("2023年全国田径锦标赛在哪里举办的?")
    print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
