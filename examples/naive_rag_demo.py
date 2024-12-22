# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Basic example of using the assistant with a naive retrieval-based model.
"""

import sys

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge_base = Knowledge(
    data_path="data/medical_corpus.txt",
    vector_db=LanceDb(
        table_name="medical",
        embedder=Text2VecEmb(),
    ),
)
# Load the knowledge base
knowledge_base.load(recreate=True)

m = Agent(
    model=AzureOpenAIChat(),
    knowledge_base=knowledge_base,
    # The add_references_to_prompt will update the prompt with references from the knowledge base.
    add_references=True,
    debug_mode=True
)
m.print_response("肛门病变可能是什么疾病的症状?", stream=True)
