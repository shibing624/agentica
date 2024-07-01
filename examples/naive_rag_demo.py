# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Basic example of using the assistant with a naive retrieval-based model.
"""

import sys

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM
from agentica.knowledge_base import KnowledgeBase
from agentica.vectordb.lancedb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge_base = KnowledgeBase(
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
assistant.run("肛门病变可能是什么疾病的症状?")
