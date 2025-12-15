# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Python agent memory demo, demonstrates Agent with RunPythonCodeTool, knowledge base and memory
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat, RunPythonCodeTool
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica import Knowledge, AgentMemory
from agentica.db.sqlite import SqliteDb


def main():
    db_file = "outputs/person1.db"
    db = SqliteDb(db_file=db_file)
    
    knowledge_base = Knowledge(
        data_path=[],
        vector_db=LanceDb(embedder=Text2VecEmb())
    )
    knowledge_base.load()
    m = Agent(
        name="Python Agent",
        model=OpenAIChat(model='gpt-4o'),
        tools=[BaiduSearchTool(), RunPythonCodeTool(save_and_run=True)],
        instructions=["You are an expert Python programmer."],
        show_tool_calls=True,
        markdown=True,
        debug_mode=True,
        knowledge=knowledge_base,
        db=db,
        update_knowledge=True,
        search_knowledge=True,
        memory=AgentMemory(db=db, create_user_memories=True),
    )
    r = m.run(
        "如果Eliud Kipchoge能够无限期地保持他创造记录的马拉松速度，那么他需要多少小时才能跑完地球和月球在最近接时之间的距离？请在进行计算时使用维基百科页面上的最小近地点值。将结果用中文回答"
    )
    print(r.content)


if __name__ == '__main__':
    main()
