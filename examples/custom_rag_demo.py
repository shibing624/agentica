# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Use a custom function to generate references for RAG.

You can use the custom_references_function to generate references for the RAG model.
The function takes a query and returns a list of references from the knowledge base.

usage:
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16

python custom_rag_demo.py
"""

import json
import sys
from typing import Optional

sys.path.append('..')
from agentica import Assistant
from agentica.knowledge.knowledge_base import KnowledgeBase
from agentica.vectordb.pgvector import PgVector
from agentica.emb.text2vec import Text2VecEmb

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = KnowledgeBase(
    data_path=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(collection="ThaiRecipes", db_url=db_url, embedder=Text2VecEmb()),
)
# Comment after first run
knowledge_base.load(recreate=True)


def custom_references_function(query: str, **kwargs) -> Optional[str]:
    """Return a list of references from the knowledge base"""
    print(f"-*- Searching for references for query: {query}")
    relevant_docs = knowledge_base.search(query=query, num_documents=5)
    if len(relevant_docs) == 0:
        return None
    content = json.dumps([doc.to_dict() for doc in relevant_docs], indent=2, ensure_ascii=False)
    print('references:', content)
    return content


assistant = Assistant(
    knowledge_base=knowledge_base,
    # Generate references using a custom function.
    references_function=custom_references_function,
    # Adds references to the prompt.
    add_references_to_prompt=True,
    debug_mode=False,
)
r = assistant.run("How to make Thai curry?")
print("".join(r))
