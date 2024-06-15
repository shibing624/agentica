import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.documents import TextDocuments
from actionflow.vectordb.lancedb import LanceDb

knowledge_base = TextDocuments(
    path="data/medical_corpus.txt",
    emb_db=LanceDb(),
)
# Load the knowledge base
knowledge_base.load(recreate=False)

assistant = Assistant(
    llm=AzureOpenAILLM(),
    knowledge_base=knowledge_base,
    # The add_references_to_prompt will update the prompt with references from the knowledge base.
    add_references_to_prompt=True,
)
assistant.print_response("How do I make pad thai?", markdown=True)
