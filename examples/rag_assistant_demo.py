import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.documents import TextDocuments
from actionflow.vectordb.lancedb import LanceDb
from actionflow.emb.text2vec_emb import Text2VecEmb

knowledge_base = TextDocuments(
    data_path="data/medical_corpus.txt",
    vector_db=LanceDb(
        embedder=Text2VecEmb(),
    ),
)
# Load the knowledge base
knowledge_base.load(recreate=True)

assistant = Assistant(
    llm=AzureOpenAILLM(),
    knowledge_base=knowledge_base,
    # The add_references_to_prompt will update the prompt with references from the knowledge base.
    add_references_to_prompt=True,
)
assistant.print_response("肛门病变可能是什么疾病的症状?", markdown=True)
