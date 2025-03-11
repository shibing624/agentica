import sys
from typing import Optional, List

import typer

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica import SqlAgentStorage

llm = OpenAIChat()
# llm = OllamaChat(model="qwen:0.5b")
print(llm)
emb = Text2VecEmb()
print(emb)
output_dir = "outputs"
db_file = f"{output_dir}/medical_corpus.db"
table_name = 'medical_corpus'
knowledge_base = Knowledge(
    data_path=["data/medical_corpus.txt", "data/paper_sample.pdf"],  # PDF files also works
    vector_db=LanceDb(
        embedder=emb,
        uri=f"{output_dir}/medical_corpus.lancedb",
    ),
)
# Comment out after first run
knowledge_base.load(recreate=True)

storage = SqlAgentStorage(table_name=table_name, db_file=db_file)


def pdf_app(new: bool = False, user: str = "user"):
    sess_id: Optional[str] = None

    if not new:
        session_ids: List[str] = storage.get_all_session_ids(user)
        if len(session_ids) > 0:
            sess_id = session_ids[0]
    print(f"User: {user}\nrun_id: {sess_id}\n")
    m = Agent(
        model=llm,
        session_id=sess_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
        debug_mode=True,
    )
    if sess_id is None:
        sess_id = m.run_id
        print(f"Started Run: {sess_id}\n")
    else:
        print(f"Continuing Run: {sess_id}\n")
    m.cli_app()


if __name__ == "__main__":
    typer.run(pdf_app)
