# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Chat PDF app demo - Demonstrates a CLI-based PDF Q&A application

This example shows how to build a complete PDF chat application:
1. Load multiple documents (PDF, TXT)
2. Persistent session storage
3. Interactive CLI interface
"""
import sys
import os
from typing import Optional, List
import typer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, OpenAIEmb
from agentica.agent.config import ToolConfig
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.db.sqlite import SqliteDb

pwd_path = os.path.dirname(os.path.abspath(__file__))

output_dir = "outputs"
db_file = f"{output_dir}/medical_corpus.db"

files = [os.path.join(pwd_path, '../data/medical_corpus.txt'), os.path.join(pwd_path, '../data/paper_sample.pdf')]
# Create knowledge base with multiple documents
knowledge_base = Knowledge(
    data_path=files,
    vector_db=LanceDb(
        embedder=OpenAIEmb(),
        uri=f"{output_dir}/medical_corpus.lancedb",
    ),
)

# Load knowledge base (comment out after first run)
knowledge_base.load(recreate=True)

db = SqliteDb(db_file=db_file)


def pdf_app(new: bool = False, user: str = "user"):
    """Run the PDF chat application.

    Args:
        new: Start a new session
        user: User identifier
    """
    # TODO: Session persistence (session_id, user_id, db) moved to SessionManager in V2

    agent = Agent(
        model=OpenAIChat(),
        knowledge=knowledge_base,
        search_knowledge=True,
        tool_config=ToolConfig(read_chat_history=True),
        debug_mode=True,
    )

    print(f"User: {user}\nRun ID: {agent.run_id}\n")
    agent.cli_app()


if __name__ == "__main__":
    typer.run_sync(pdf_app)
