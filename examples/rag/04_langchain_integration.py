# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: RAG integrated with LangChain demo - Using LangChain vector store

This example shows how to integrate Agentica with LangChain's vector stores
for retrieval-augmented generation.

pip install langchain langchain-community langchain-openai langchain-chroma langchain-text-splitters chromadb
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.knowledge.langchain_knowledge import LangChainKnowledge

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

pwd_path = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("RAG with LangChain Integration Demo")
    print("=" * 60)

    # Define paths
    chroma_db_dir = "outputs/chroma_db"
    file_path = os.path.join(pwd_path, "../data/news_docs.txt")

    # Check if data file exists
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}")
        print("Please ensure the data file exists.")
        return

    print(f"\n1. Loading document: {file_path}")
    raw_documents = TextLoader(file_path).load()

    print("2. Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    print(f"   Created {len(documents)} chunks")

    print("3. Creating embeddings and storing in Chroma...")
    Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=chroma_db_dir)

    print("4. Creating retriever from vector store...")
    db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=chroma_db_dir)
    retriever = db.as_retriever()

    print("5. Creating LangChainKnowledge and Agent...")
    knowledge = LangChainKnowledge(retriever=retriever)
    agent = Agent(knowledge=knowledge)

    print("\n" + "=" * 60)
    print("Asking question: 2023年全国田径锦标赛在哪里举办的?")
    print("=" * 60)

    response = agent.run("2023年全国田径锦标赛在哪里举办的?")
    print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
