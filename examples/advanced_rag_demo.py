# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Advanced RAG demo
实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）
"""
import json
import sys

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM
from agentica.knowledge_base import KnowledgeBase
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
    num_documents = 5
    emb_relevant_docs = knowledge_base.search(query=query, num_documents=num_documents)
    # 关键词检索
    all_documents = []
    for document_list in knowledge_base.document_lists:
        all_documents.extend(document_list)
    keyword_relevant_docs = [doc for doc in all_documents if query.lower() in doc.content.lower()]
    # 按单词出现的词频排序
    keyword_relevant_docs = sorted(keyword_relevant_docs, key=lambda x: x.content.lower().count(query.lower()),
                                   reverse=True)
    # 合并两路召回结果
    relevant_docs = emb_relevant_docs + keyword_relevant_docs[:num_documents]
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
