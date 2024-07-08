# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Advanced RAG demo

实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）

pip install rank_bm25 jieba agentica
"""
import json
import sys

import jieba
from rank_bm25 import BM25Okapi

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM
from agentica.knowledge.knowledge_base import KnowledgeBase
from agentica.vectordb.lancedb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb

knowledge_base = KnowledgeBase(
    data_path="data/paper_sample.pdf",
    vector_db=LanceDb(
        embedder=Text2VecEmb(),
    ),
)
# Load the knowledge base
knowledge_base.load(recreate=False)


def merge_references_function(query: str, **kwargs) -> str:
    """Return a list of references from the knowledge base"""
    print(f"-*- Searching for references for query: {query}")
    # 向量检索
    num_documents = 3
    emb_relevant_docs = knowledge_base.search(query=query, num_documents=num_documents)
    print('emb_relevant_docs:', emb_relevant_docs)
    # 关键词检索
    all_documents = []
    for document_list in knowledge_base.document_lists:
        all_documents.extend(document_list)
    tokenized_corpus = [jieba.lcut(doc.content) for doc in all_documents if doc]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = jieba.lcut(query)
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_documents]
    keyword_relevant_docs = [all_documents[i] for i in top_n_indices]
    print('keyword_relevant_docs:', keyword_relevant_docs)
    # 合并两路召回结果
    relevant_docs = emb_relevant_docs + keyword_relevant_docs
    # 召回排序, todo: rerank

    if len(relevant_docs) == 0:
        return ""
    content = json.dumps([doc.to_dict() for doc in relevant_docs], indent=2, ensure_ascii=False)
    print('references:', content)
    return content


assistant = Assistant(
    llm=AzureOpenAILLM(model='gpt-4o'),
    knowledge_base=knowledge_base,
    references_function=merge_references_function,
    # The add_references_to_prompt will update the prompt with references from the knowledge base.
    add_references_to_prompt=True,
    debug_mode=True,
)
r = assistant.run("Finetune LLM有啥好处?")
print(r, "".join(r))
