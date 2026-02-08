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

from agentica import Agent, OpenAIChat
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica.db.sqlite import SqliteDb

pwd_path = os.path.dirname(os.path.abspath(__file__))


llm = OpenAIChat()
emb = Text2VecEmb()
output_dir = "outputs"
db_file = f"{output_dir}/medical_corpus.db"

files = [os.path.join(pwd_path, '../data/medical_corpus.txt'), os.path.join(pwd_path, '../data/paper_sample.pdf')]
# Create knowledge base with multiple documents
knowledge_base = Knowledge(
    data_path=files,
    vector_db=LanceDb(
        embedder=emb,
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
    sess_id: Optional[str] = None

    if not new:
        session_ids: List[str] = db.get_all_session_ids(user_id=user)
        if len(session_ids) > 0:
            sess_id = session_ids[0]
    
    print(f"User: {user}\nSession ID: {sess_id}\n")
    
    agent = Agent(
        model=llm,
        session_id=sess_id,
        user_id=user,
        knowledge_base=knowledge_base,
        db=db,
        search_knowledge=True,
        read_chat_history=True,
        debug_mode=True,
    )
    
    if sess_id is None:
        sess_id = agent.run_id
        print(f"Started Run: {sess_id}\n")
    else:
        print(f"Continuing Run: {sess_id}\n")
    
    agent.cli_app()


if __name__ == "__main__":
    typer.run(pdf_app)
