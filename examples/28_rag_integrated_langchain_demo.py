# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: agentica integrated langchain db demo

pip install langchain
"""

import sys

sys.path.append('..')
from agentica import Agent

from agentica.knowledge.langchain_knowledge import LangChainKnowledge
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Define the directory where the Chroma database is located
chroma_db_dir = "outputs/chroma_db"

# Define the path to the document to be loaded into the knowledge base
file_path = "data/news_docs.txt"

# Load the document
raw_documents = TextLoader(file_path).load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Embed each chunk and load it into the vector store
Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=chroma_db_dir)

# Get the vector database
db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=chroma_db_dir)

# Create a retriever from the vector store
retriever = db.as_retriever()

# Create a knowledge base from the vector store
knowledge = LangChainKnowledge(retriever=retriever)

# Create an agent with the knowledge base
agent = Agent(knowledge=knowledge)

# Use the agent to ask a question and print a response.
r = agent.run("2023年全国田径锦标赛在哪里举办的?")
print(r)
