"""
This is a simple example of how to use the PythonAgent class to interact with the Assistant.
usage:
    python python_assistant_demo.py

运行两次，第一次会基于谷歌搜索回答问题，并生成一个person1.csv文件，第二次会读取这个文件（不再进行搜索），然后进行问题回答。
"""
import os.path
import sys

sys.path.append('..')
from agentica import PythonAgent, OpenAIChat, SqlAgentStorage
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica import Knowledge, AgentMemory
from agentica.memorydb import CsvMemoryDb


def main():
    memory_file = "outputs/person1.csv"
    knowledge_base = Knowledge(
        data_path=memory_file if os.path.exists(memory_file) else [],
        vector_db=LanceDb(embedder=Text2VecEmb())
    )
    knowledge_base.load()
    m = PythonAgent(
        model=OpenAIChat(model='gpt-4o'),
        tools=[BaiduSearchTool()],
        show_tool_calls=True,
        debug_mode=True,
        knowledge=knowledge_base,
        storage=SqlAgentStorage(table_name="person1", db_file="outputs/person1.db"),
        update_knowledge=True,
        search_knowledge=True,
        memory=AgentMemory(db=CsvMemoryDb(memory_file), create_user_memories=True),
    )
    r = m.run(
        "如果Eliud Kipchoge能够无限期地保持他创造记录的马拉松速度，那么他需要多少小时才能跑完地球和月球在最近接时之间的距离？"
        "请在进行计算时使用维基百科页面上的最小近地点值。将结果用中文回答"
    )
    print(r)


if __name__ == '__main__':
    main()
