# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Advanced RAG demo

实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）

pip install similarities agentica transformers torch
"""
import sys
import torch
from typing import List, Union, Set
from similarities import BM25Similarity
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append('..')
from agentica import Assistant, AzureOpenAILLM
from agentica.knowledge.knowledge_base import KnowledgeBase
from agentica.vectordb.lancedb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

knowledge_base = KnowledgeBase(
    data_path="data/paper_sample.pdf",
    vector_db=LanceDb(embedder=Text2VecEmb())
)
# Load the knowledge base
knowledge_base.load(recreate=False)

# Initialize the BM25 similarity
all_documents = []
for document_list in knowledge_base.document_lists:
    all_documents.extend(document_list)
corpus = [doc.content for doc in all_documents if doc]
bm25_model = BM25Similarity(corpus)

# Initialize rerank model
rerank_model_name_or_path = "BAAI/bge-reranker-base"
rerank_top_k = 3
if rerank_model_name_or_path:
    rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)
    rerank_model.to(device)
    rerank_model.eval()
    logger.info(f"rerank_model: {rerank_model}")


def get_reranker_score(rerank_model, query: str, reference_results: Union[Set, List[str]]):
    """Get reranker score."""
    pairs = []
    for reference in reference_results:
        pairs.append([query, reference])
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs_on_device = {k: v.to(rerank_model.device) for k, v in inputs.items()}
        scores = rerank_model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()
    return scores


def merge_references_function(query: str, **kwargs) -> str:
    """Return a list of references from the knowledge base"""
    logger.info(f"-*- Searching for references for query: {query}")
    # 向量检索
    num_documents = 3
    emb_res = knowledge_base.search(query=query, num_documents=num_documents)
    emb_relevant_docs = [i.content for i in emb_res]
    logger.debug(f'emb_relevant_docs: {emb_relevant_docs}')
    # 关键词检索(基于BM25)
    res = bm25_model.search(query, num_documents)[0]
    keyword_relevant_docs = [i.get('corpus_doc') for i in res]
    logger.debug(f'keyword_relevant_docs: {keyword_relevant_docs}')
    # 合并两路召回结果
    relevant_docs = set(emb_relevant_docs + keyword_relevant_docs)
    if len(relevant_docs) == 0:
        return ""
    # 召回排序，Rerank 模型排序
    if rerank_model_name_or_path and rerank_model:
        # Rerank reference results
        rerank_scores = get_reranker_score(rerank_model, query, relevant_docs)
        logger.debug(f"rerank_scores: {rerank_scores}")
        # Get rerank top k chunks
        relevant_docs = [reference for reference, score in sorted(
            zip(relevant_docs, rerank_scores), key=lambda x: x[1], reverse=True)][:rerank_top_k]
    content = "\n".join(relevant_docs)
    logger.info(f'references: {content}')
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
