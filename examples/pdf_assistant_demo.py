import sys
from typing import Optional, List

import typer

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.llm.ollama_llm import OllamaLLM
from actionflow.documents import TextDocuments
from actionflow.vectordb.lancedb import LanceDb  # noqa
from actionflow.emb.text2vec_emb import Text2VecEmb
from actionflow.sqlite_storage import SqliteStorage

llm = AzureOpenAILLM()
# llm = OllamaLLM(model="qwen:0.5b")
print(llm)
emb = Text2VecEmb()
print(emb)
output_dir = "outputs"
db_file = f"{output_dir}/medical_corpus.db"
table_name = 'medical_corpus'
knowledge_base = TextDocuments(
    data_path="data/medical_corpus.txt",
    vector_db=LanceDb(
        embedder=emb,
        uri=f"{output_dir}/medical_corpus.lancedb",
    ),
)
# Comment out after first run
knowledge_base.load(recreate=True)

storage = SqliteStorage(table_name=table_name, db_file=db_file)


def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
    print(f"User: {user}\nrun_id: {run_id}\n")
    assistant = Assistant(
        llm=llm,
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
        output_dir=output_dir,
        debug_mode=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")
    assistant.cli(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_assistant)
