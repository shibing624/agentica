"""
This is a simple example of how to use the PythonAssistant class to interact with the Assistant.
usage:
    python python_assistant_demo.py

运行两次，第一次会基于谷歌搜索回答问题，并生成一个person1.csv文件，第二次会读取这个文件（不再进行搜索），然后进行问题回答。
"""
import os.path
import sys

sys.path.append('..')
from agentica import PythonAssistant, AzureOpenAILLM, SqlAssistantStorage
from agentica.tools.search_serper import SearchSerperTool
from agentica.vectordb.lancedb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica import KnowledgeBase, AssistantMemory, CsvMemoryDb


def main():
    memory_file = "person1.csv"
    knowledge_base = KnowledgeBase(
        data_path=memory_file if os.path.exists(memory_file) else [],
        vector_db=LanceDb(embedder=Text2VecEmb())
    )
    knowledge_base.load()
    m = PythonAssistant(
        llm=AzureOpenAILLM(model='gpt-4o'),
        tools=[SearchSerperTool()],
        show_tool_calls=True,
        debug_mode=True,
        knowledge_base=knowledge_base,
        storage=SqlAssistantStorage(table_name="person1", db_file="outputs/person1.db"),
        update_knowledge=True,
        search_knowledge=True,
        memory=AssistantMemory(db=CsvMemoryDb(memory_file)),
        create_memories=True,
        force_update_memory_after_run=True,
    )
    r = m.run(
        "如果Eliud Kipchoge能够无限期地保持他创造记录的马拉松速度，那么他需要多少小时才能跑完地球和月球在最近接时之间的距离？"
        "请在进行计算时使用维基百科页面上的最小近地点值。将结果用中文回答"
    )
    print("".join(r))
    m.storage.export("outputs/person1_storage.csv")


if __name__ == '__main__':
    main()
